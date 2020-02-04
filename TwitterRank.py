#!/usr/bin/python3
# -*-coding:utf-8-*-

from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pandas as pd
import re
import scipy.stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import networkx as nx
import time, sys
from IPython.display import clear_output
import ipywidgets as widgets


def update_progress(progress):
    bar_length = 80
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))
    

    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    out = widgets.Output()
    out.append_stdout(text)


def create_vocab_list(df_in=None):
    """
    Get the glossary

    :return:List, each element is a word
    """
    for index, row in df_in.iterrows():
        vocab_list.extend(re.findall('\w+',row['Clean Tweet']))

    return list(set(vocab_list))


def get_sim(t, i, j, row_normalized_dt):
    '''
    Get sim (i, j)
    '''
    return 1.0 - abs(row_normalized_dt[i,t] - row_normalized_dt[j,t])

def get_Pt(t, tweets_list, friends_tweets_list, row_normalized_dt, relationship):
    '''
    Get Pt, Pt [i] [j] is the probability that i follows j and i is affected by j under topic t
    '''
    print('Creating transition probability for topic {}'.format(t))
    Pt = scipy.sparse.lil_matrix((relationship.get_shape()))
    #rows,cols = relationship.nonzero()
    #for row,col in zip(rows,cols):
    #    Pt[row,col] = tweets_list[col]/friends_tweets_list[row] * get_sim(t, row, col, row_normalized_dt)
    rel_c = relationship.tocoo()    
    for i,j in zip(rel_c.row, rel_c.col):
        Pt[i,j] = tweets_list[j]/friends_tweets_list[i] * get_sim(t, i, j, row_normalized_dt)
    return Pt


def get_TRt(gamma, topic_number, Pt, Et, iter=1000, tolerance=1e-16):
    '''
    Get TRt, the influence matrix of each user under t topic
    :param gamma: Get tuning parameters in TRt's formula
    :param Pt: Pt Matrix, Pt [i] [j] represents the probability that i follows j, and i is affected by j under topic t
    :param Et: Et Matrix, Et [i] represents user i's attention to topic t, has been normalized, all elements are added to 1
    :param iter: Maximum number of iterations
    :param tolerance: Stop iteration after TRt iteration when Euclidean distance from iteration is less than tolerance
    :return: TRt,TRt[i]Represents the influence of user i under topic t
    '''
    TRt = Et[:,topic_number]
    old_TRt = TRt
    i = 0
    print('Calculating influence scores for users under topic {}'.format(topic_number))
    while i < iter:
        TRt = gamma * (Pt*TRt) + (1 - gamma) * Et[:,topic_number]
        euclidean_dis = np.linalg.norm(TRt - old_TRt)
        if euclidean_dis < tolerance:
            break
        old_TRt = TRt
        i += 1
    return TRt


def get_doc_list(df_in):
    """
    Get a list, each element is a piece of document
    :param samples: Number of documents
    :return: np.array,Each element is a document
    """
    print('Collecting grouped tweets by user')
    doc_list = df_in.groupby('Author')['Clean Tweet'].apply(lambda x:' '.join(x)).values
    return doc_list

def get_feature_matrix(doc_list):
    """
    Get the feature matrix of each document, each word as a feature
    :param doc_list: list,Each element is a document
    :return: i row and j column list, i is the number of samples, j is the number of features, and feature_matrix_ij represents the number of times that feature j appears in the i-th sample

    """
    print('Creating term-author matrix')
    vectorizer = CountVectorizer()
    feature_matrix = vectorizer.fit_transform(raw_documents=doc_list)
    return feature_matrix, vectorizer


def get_num_tweets_list(nx_graph,df_in):
    """
    Get the number of tweets per user
    :return: list,The i element is the number of tweets from the i user
    """
    print('Gathering tweet count for all users')
    authors = df_in['Author'].value_counts()
    num_nodes = len(nx_graph)
    num_tweets_list = np.zeros(shape=(num_nodes))
    for ind,node in enumerate(nx_graph):
        num_tweets_list[ind] = authors[node] 
        update_progress(ind/num_nodes)
    return num_tweets_list

def get_relationship(nx_graph):
    """
    Get user relationship matrix
    :param samples: Number of Users
    :return: i row and j column, relationship [i] [j] = 1 means j follows i
    """
    print('Creating relationship matrix')
    return nx.to_scipy_sparse_matrix(nx_graph)


def get_friends_tweets_list(relationship, tweets_list):
    """
    Get the sum of the number of tweets that each user has followed
    :param relationship: User relationship matrix, i rows and j columns, relationship [i] [j] = 1 means j follows i
    :param tweets_list: list,The i element is the number of tweets from the i user
    :return: list,The i element is the sum of the tweets from everyone i followed
    """
    print('Gathering tweet counts for friends of each user')
    friends_tweets_list = np.zeros(shape=(relationship.get_shape()[0]))
    for i in range(relationship.get_shape()[0]):
        friends_tweets_list[i] = tweets_list[relationship[i].nonzero()[1]].sum()
        update_progress(i / relationship.get_shape()[0])
    return friends_tweets_list

def get_top_topic_influencers(TR, nx_graph, num_topics=5, num_influencers=10):
    top_influencer_list = pd.DataFrame()
    for i, TRt in enumerate(TR):
        top_influencer_list['Topic' + str(i) + 'Influncers'] = pd.Series(TRt,index=nx_graph.nodes()).sort_values(ascending=False).head(num_influencers).index
    return top_influencer_list

def get_TR(num_topics, tweets_list, friends_tweets_list, row_normalized_dt, col_normalized_dt, relationship,
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
    print('Ranking users by ')
    TR = np.zeros(shape=(num_topics,tweets_list.shape[0]))
    for i in range(num_topics):
        print('topic number {}'.format(i))
        Pt = get_Pt(i, tweets_list, friends_tweets_list, row_normalized_dt, relationship)
        Et = col_normalized_dt
        TR[i] = get_TRt(gamma, i, Pt, Et, tolerance).flatten()
    return TR

def get_graph_object(df_in,source='Author',target='Retweet of',filter_column=None):
    """
    Get the network in the form of a networkx graph object
    :param df_in: The raw dataframe with authors and tweets
    return: network of authors as a networkx object with retweets as edges.
    """
    print('Creating graph object')
    G = nx.convert_matrix.from_pandas_edgelist(df_in,source,target)
    print('Removing nodes where author doesn\'t exist in raw df')
    G.remove_nodes_from(list(df_in['Retweet of'][~df_in['Retweet of'].isin(df_in['Author'].value_counts().index)].unique()))
    return G

def get_lda_model(topics, n_iter, df_in=None):
    """
    Get the trained LDA model
    :param topics: Number of topics
    :param n_iter: Number of iterations
    :return: model,LDA model after training
             vocab_list,A list of all words that have appeared in these documents, each element is a word
    """
    doc_list = get_doc_list(df_in)
    #vocab_list = create_vocab_list(df_in)
    term_frequency, vectorizer = get_feature_matrix(doc_list)
    #feature_matrix = term_frequency.toarray()
    vocab_list = vectorizer.get_feature_names()
    print('Fitting LDA model to discover topics')
    model = LatentDirichletAllocation(n_components=topics,max_iter=n_iter)
    model.fit(term_frequency)
    return model, vocab_list, term_frequency


def print_topics_as_df(model, vocab_list, n_top_words=5):

    topic_word_df = pd.DataFrame(model.components_,columns=vocab_list)
    sorted_topic_words = pd.DataFrame()
    for index, row in topic_word_df.iterrows():
        row_df = pd.DataFrame({'topic_'+str(index): row.sort_values(ascending=False).index[:5].values})
        sorted_topic_words = pd.concat([sorted_topic_words,row_df],axis=1)
    return sorted_topic_words


def get_TR_using_DT(dt, df_in, num_topics=5, gamma=0.2, tolerance=1e-16):
    """
    Knowing the DT matrix gives the TR matrix
    :param dt: dt The matrix represents the topic distribution of the document, and dt [i] [j] represents the proportion of the topic j in the document i
    :param samples: Number of documents
    :param topics:  Number of topics
    :param gamma: Get the tuning parameters in the formula for TRt
    :param tolerance: Stop iteration after TRt iteration when Euclidean distance from iteration is less than tolerance
    :return TR: matrix,TR[i][j]Is the influence of user j on topic i
    :return nx_graph: A Networkx graph object where the nodes are authors and edges are retweets
    """
    row_normalized_dt = dt/(dt.sum(axis=1).reshape(-1,1))
    col_normalized_dt = dt/dt.sum(axis=0)
    nx_graph = get_graph_object(df_in,filter_column='Retweet of')
    relationship = get_relationship(nx_graph)
    tweets_list = get_num_tweets_list(nx_graph,df_in)
    friends_tweets_list = get_friends_tweets_list(relationship, tweets_list)
    TR = get_TR(num_topics, tweets_list, friends_tweets_list, row_normalized_dt, col_normalized_dt, relationship,
                gamma, tolerance)
    return TR, nx_graph

def twitter_rank(raw_df, topics=5, n_iter=100, gamma=0.2, tolerance=1e-16):
    """
    Twitter rank of documents
    :param topics: number of topics to discover
    :param n_iter: max number of iterations for LDA model
    :param gamma: The tuning parameters in the formula for TRt
    :param tolerance: Tolerance for early stopping
    :return:TR: A matrix of author influence scores for each topic. Rows are authors, columns are topics
    :return:graph object: A Networkx graph object where the nodes are authors and edges are retweets 
    """
    model, vocab_list, term_frequency = get_lda_model(topics, n_iter, raw_df)
    print_topics_as_df(model, vocab_list, n_top_words=5)
    #dt matrix represents the topic distribution of the document, 
    #dt [i] [j] represents the proportion of the subject j in the document i
    dt = model._unnormalized_transform(term_frequency)
    TR, graph_object = get_TR_using_DT(dt, raw_df, topics, gamma, tolerance)

    return TR, graph_object, model, vocab_list
