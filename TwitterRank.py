#!/usr/bin/python2
# -*-coding:utf-8-*-

# Copyright (c) 2014 lufo <lufo816@gmail.com>
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import re
import StopWords
import scipy.stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import networkx as nx


stop_word_list = StopWords.stop_word_list


def text_parse(big_string):
    """
    从字符串中提取处它的所有单词
    :param big_string:字符串
    :return:列表，所有的出现过的单词，可重复
    """
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def create_vocab_list(df_in=None):
    """
    获得词汇表
    :return:列表，每个元素是一个词汇
    """
    vocab_list = []
    if df_in is None: 
        with open('dict.txt') as dict:
            vocab_list = [word.lower().strip() for word in dict if (word.lower().strip() + ' ' not in stop_word_list)]
    else:
        for index, row in df_in.iterrows():
            vocab_list.extend(re.findall('\w+',row['Clean Tweet']))


    return list(set(vocab_list))

def normalize(mat):
    '''
    将矩阵每一行归一化(一范数为1)
    :param mat: 矩阵
    :return: list,行归一化的矩阵
    '''
    row_normalized_mat = []
    for row_mat in mat:
        normalized_row = []
        row = np.array(row_mat).reshape(-1, ).tolist()
        row_sum = sum(row)
        for item in row:
            if row_sum != 0:
                normalized_row.append(float(item) / float(row_sum))
            else:
                normalized_row.append(0)
        row_normalized_mat.append(normalized_row)
    return row_normalized_mat


def get_sim(t, i, j, row_normalized_dt):
    '''
    获得sim(i,j)
    '''
    sim = 1.0 - abs(row_normalized_dt[i][t] - row_normalized_dt[j][t])
    # 下列三行代码为使用 KL 散度衡量相似度
    # pk = [row_normalized_dt[i][t]]
    # qk = [row_normalized_dt[j][t]]
    # sim = 1 - (scipy.stats.entropy(pk, qk) + scipy.stats.entropy(qk, pk)) / 2
    return sim


def get_Pt(t, samples, tweets_list, friends_tweets_list, row_normalized_dt, relationship):
    '''
    获得Pt,Pt[i][j]表示i关注j，在主题t下i受到j影响的概率
    '''
    Pt = []
    for i in xrange(samples):
        friends_tweets = friends_tweets_list[i]
        temp = []
        for j in xrange(samples):
            if relationship[j][i] == 1:
                if friends_tweets != 0:
                    temp.append(float(tweets_list[j]) / float(friends_tweets) * get_sim(t, i, j, row_normalized_dt))
                else:
                    temp.append(0.0)
            else:
                temp.append(0.0)
        Pt.append(temp)
    return Pt


def get_TRt(gamma, Pt, Et, iter=1000, tolerance=1e-16):
    '''
    获得TRt，在t topic下每个用户的影响力矩阵
    :param gamma: 获得 TRt 的公式中的调节参数
    :param Pt: Pt 矩阵,Pt[i][j]表示i关注j，在主题t下i受到j影响的概率
    :param Et: Et 矩阵,Et[i]代表用户 i 对主题 t 的关注度,已经归一化,所有元素相加为1
    :param iter: 最大迭代数
    :param tolerance: TRt迭代后 与迭代前欧氏距离小于tolerance时停止迭代
    :return: TRt,TRt[i]代表在主题 t 下用户 i 的影响力
    '''
    TRt = np.mat(Et).transpose()
    old_TRt = TRt
    i = 0
    # np.linalg.norm(old_TRt,new_TRt)
    while i < iter:
        TRt = gamma * (np.dot(np.mat(Pt), TRt)) + (1 - gamma) * np.mat(Et).transpose()
        euclidean_dis = np.linalg.norm(TRt - old_TRt)
        # print 'dis', dis
        if euclidean_dis < tolerance:
            break
        old_TRt = TRt
        i += 1
    return TRt


def get_doc_list(samples, df_in=None):
    """
    得到一个列表,每个元素为一片文档
    :param samples: 文档的个数
    :return: list,每个元素为一篇文档
    """
    doc_list = []
    if df_in is None:
        for i in range(1, samples + 1):
            with open('tweet_cont/tweet_cont_%d.txt' % i) as fr:
                temp = text_parse(fr.read())
            word_list = [word.lower() for word in temp if (word + ' ' not in stop_word_list and not word.isspace())]
            doc_list.append(word_list)
    else:
        doc_list = df_in.groupby('Author')['Clean Tweet'].apply(lambda x:' '.join(x)).values
    return doc_list


def get_feature_matrix(doc_list):
    """
    Get the feature matrix of each document, each word as a feature
    :param doc_list: list,Each element is a document
    :return: i row and j column list, i is the number of samples, j is the number of features, and feature_matrix_ij represents the number of times that feature j appears in the i-th sample

    """
#    feature_matrix = []
    # word_index 为字典,每个 key 为单词,value 为该单词在 vocab_list 中的下标
#    word_index = {}
#    for i in range(len(vocab_list)):
#        word_index[vocab_list[i]] = i
#    for doc in doc_list:
#        temp = [0 for i in range(len(vocab_list))]
#        for word in doc:
#            if word in word_index:
#                temp[word_index[word]] += 1
#        feature_matrix.append(temp)

    vectorizer = CountVectorizer()
    feature_matrix = vectorizer.fit_transform(raw_documents=doc_list)
    return feature_matrix, vectorizer


def get_num_tweets_list(nx_graph,df_in):
    """
    Get the number of tweets per user
    :return: list,The i element is the number of tweets from the i user
    """

    return [df_in.groupby('Author').get_group(i)['Clean Tweet'].count() for i in list(G.nodes)]

def get_relationship(nx_graph):
    """
    Get user relationship matrix
    :param samples: Number of Users
    :return: i row and j column, relationship [i] [j] = 1 means j follows i
    """
    return nx.to_scipy_sparse_matrix(nx_graph)


def get_friends_tweets_list(relationship, tweets_list):
    """
    Get the sum of the number of tweets that each user has followed
    :param relationship: User relationship matrix, i rows and j columns, relationship [i] [j] = 1 means j follows i
    :param tweets_list: list,The i element is the number of tweets from the i user
    :return: list,The i element is the sum of the tweets from everyone i followed
    """
    friends_tweets_list = [0 for i in xrange(samples)]
    for j in xrange(samples):
        for i in xrange(samples):
            if relationship[i][j] == 1:
                friends_tweets_list[j] += tweets_list[i]
    return friends_tweets_list


def get_user_list():
    """
    Get list of user ids
    :return: list,The i element is the id of user i

    """
    user = []
    with open('user_id.txt') as fr:
        for line in fr.readlines():
            user.append(line)
    return user


def get_TR(topics, samples, tweets_list, friends_tweets_list, row_normalized_dt, col_normalized_dt, relationship,
           gamma=0.2, tolerance=1e-16):
    """
    Get a TR matrix that represents the influence of each user on each topic
    :param topics: Number of topics
    :param samples: User number
    :param tweets_list: list,The i element is the number of tweets from the i user
    :param friends_tweets_list: list,The i element is the sum of the tweets from everyone i followed
    :param row_normalized_dt: dt Row normalization matrix
    :param col_normalized_dt: dt Column normalization matrix
    :param relationship: i row and j column, relationship [i] [j] = 1 means j follows i
    :param gamma: Get the tuning parameters in the formula for TRt
    :param tolerance: Stop iteration after TRt iteration when Euclidean distance from iteration is less than tolerance
    :return: list,TR[i][j]Is the influence of user j on topic i
    """
    TR = []
    for i in xrange(topics):
        Pt = get_Pt(i, samples, tweets_list, friends_tweets_list, row_normalized_dt, relationship)
        Et = col_normalized_dt[i]
        TR.append(np.array(get_TRt(gamma, Pt, Et, tolerance)).reshape(-1, ).tolist())
    return TR

def get_graph_object(df_in,source='Retweet of',target='Author',filter_column=None):
    """
    Get the network in the form of a networkx graph object
    :param df_in: The raw dataframe with authors and tweets
    return: network of authors as a networkx object with retweets as edges.
    """

    return nx.from_pandas_edgelist(df_in[df_in[filter_column].notna()],source,target)

def get_TR_sum(TR, samples, topics):
    """
    获取总的 TR 矩阵,有 i 个元素,TR_sum[i]为用户 i 在所有主题下影响力之和
    :param TR: list,TR[i][j]为第 i 个主题下用户 j 的影响力
    :param samples: 用户数
    :param topics: 主题数
    :return: list,有 i 个元素,TR_sum[i]为用户 i 在所有主题下影响力之和
    """
    TR_sum = [0 for i in xrange(samples)]
    for i in xrange(topics):
        for j in xrange(samples):
            TR_sum[j] += TR[i][j]
    TR_sum.sort()
    return TR_sum


def get_lda_model(samples, topics, n_iter, df_in=None,):
    """
    获得训练后的 LDA 模型
    :param samples: 文档数
    :param topics: 主题数
    :param n_iter: 迭代数
    :return: model,训练后的 LDA 模型
             vocab_list,列表，表示这些文档出现过的所有词汇，每个元素是一个词汇
    """
    doc_list = get_doc_list(samples,df_in)
    #vocab_list = create_vocab_list(df_in)
    term_frequency, vectorizer = get_feature_matrix(doc_list)
    #feature_matrix = term_frequency.toarray()
    vocab_list = vectorizer.get_feature_names()
    model = LatentDirichletAllocation(n_components=topics,max_iter=n_iter)
    model.fit(term_frequency)
    return model, vocab_list, term_frequency


def print_topics_as_df(model, vocab_list, n_top_words=5):
    """
    输出模型中每个 topic 对应的前 n 个单词
    :param model:  lda 模型
    :param vocab_list: 列表，表示这些文档出现过的所有词汇，每个元素是一个词汇
    """
#    topic_word = model.topic_word_
    # print 'topic_word',topic_word
#    for i, topic_dist in enumerate(topic_word):
#        topic_words = np.array(vocab_list)[np.argsort(topic_dist)][:-n_top_words:-1]
#        print('Topic {}: {}'.format(i + 1, ' '.join(topic_words)))

    topic_word_df = pd.DataFrame(model.components_,columns=vocab_list)
    sorted_topic_words = pd.DataFrame()
    for index, row in topic_word_df.iterrows():
        row_df = pd.DataFrame({'topic_'+str(index): row.sort_values(ascending=False).index[:5].values})
        sorted_topic_words = pd.concat([sorted_topic_words,row_df],axis=1)
    return sorted_topic_words


def get_TR_using_DT(dt, df_in, samples, topics=5, gamma=0.2, tolerance=1e-16):
    """
    Knowing the DT matrix gives the TR matrix
    :param dt: dt The matrix represents the topic distribution of the document, and dt [i] [j] represents the proportion of the topic j in the document i
    :param samples: Number of documents
    :param topics:  Number of topics
    :param gamma: Get the tuning parameters in the formula for TRt
    :param tolerance: Stop iteration after TRt iteration when Euclidean distance from iteration is less than tolerance
    :return TR: list,TR[i][j]Is the influence of user j on topic i
    :return TR_sum: list,There are i elements, TR_sum [i] is the sum of influence of user i under all topics
    """
    row_normalized_dt = dt/np.sum(dt,axis=0)
    col_normalized_dt = dt/np.sum(dt,axis=1)
    nx_graph = get_graph_object(df_in,filter_column='Retweet of')
    relationship = get_relationship(nx_graph)
    tweets_list = get_num_tweets_list(nx_graph,df_in)
    friends_tweets_list = get_friends_tweets_list(samples, relationship, tweets_list)
    user = get_user_list()
    TR = get_TR(topics, samples, tweets_list, friends_tweets_list, row_normalized_dt, col_normalized_dt, relationship,
                gamma, tolerance)
    for i in xrange(topics):
        print(TR[i])
        print(user[TR[i].index(max(TR[i]))])
    TR_sum = get_TR_sum(TR, samples, topics)
    return TR, TR_sum


def get_doc_topic_distribution_using_lda_model(model, feature_matrix):
    """
    使用训练好的 LDA 模型得到新文档的主题分布
    :param model: lda 模型
    :param feature_matrix: i行j列list，i为样本数，j为特征数，feature_matrix[i][j]表示第i个样本中特征j出现的次数
    :return:
    """
    return model.transform(np.array(feature_matrix), max_iter=100, tol=0)


def using_lda_model_test_other_data(topics=5, n_iter=100, num_of_train_data=10, num_of_test_data=5, gamma=0.2,
                                    tolerance=1e-16):
    """
    训练 LDA 模型然后用训练好的 LDA 模型得到新文档的主题然后找到在该文档所对应的主题中最有影响力的用户
    :param topics:  LDA 主题数
    :param n_iter:  LDA 模型训练迭代数
    :param num_of_train_data: 训练集数据量
    :param num_of_test_data: 测试集数据量
    :param gamma: 获得 TRt 的公式中调节参数
    :param tolerance: TRt迭代后 与迭代前欧氏距离小于tolerance时停止迭代
    """
    model, vocab_list = get_lda_model(samples=num_of_train_data, topics=topics, n_iter=n_iter)
    dt = model.self._unnormalized_transform(X)
    print_topics(model, vocab_list, n_top_words=5)
    TR, TR_sum = get_TR_using_DT(dt, samples=num_of_train_data, topics=topics, gamma=gamma, tolerance=tolerance)
    doc_list = get_doc_list(samples=num_of_test_data)
    feature_matrix = get_feature_matrix(doc_list)
    dt = get_doc_topic_distribution_using_lda_model(model, feature_matrix)
    # doc_user[i][j]表示第 i 个文本与第 j 个用户的相似度
    doc_user = np.dot(dt, TR)
    user = get_user_list()
    for i, doc in enumerate(doc_user):
        print(user[i], user[list(doc).index(max(doc))])


def twitter_rank(raw_df, topics=5, n_iter=100, samples=30, gamma=0.2, tolerance=1e-16):
    """
    对文档做twitter rank
    :param topics: 主题数
    :param n_iter: 迭代数
    :param samples: 文档数
    :param gamma: 获得 TRt 的公式中调节参数
    :param tolerance: TRt迭代后 与迭代前欧氏距离小于tolerance时停止迭代
    :return:
    """
    model, vocab_list, term_frequency = get_lda_model(samples, topics, n_iter)
    # topic_word为i行j列array，i为主题数，j为特征数，topic_word_ij表示第i个主题中特征j出现的比例
    print_topics_as_df(model, vocab_list, n_top_words=5)
    # dt 矩阵代表文档的主题分布,dt[i][j]代表文档 i 中属于主题 j 的比重
    dt = np.mat(model._unnormalized_transform(term_frequency))
    TR, TR_sum = get_TR_using_DT(dt, raw_df, samples, topics, gamma, tolerance)
