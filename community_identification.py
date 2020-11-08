# -*- coding: utf-8 -*-

import os 
from collections import Counter 

import networkx as nx 
import networkx.algorithms.community as nx_comm
import community as lv_comm
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns



''' create graph '''
def create_graph(file_name):
    graph = nx.Graph()
    # create graph from edges
    with open(file_name) as f:
        f.readline()
        while True:
            line = f.readline().strip()
            if line:
                line = line.split(',')
                graph.add_edge(int(line[0]), int(line[1]))
            else:
                break 
    # remove rings
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


''' basic information of graph '''
def describe_graph(graph):
    # number of nodes
    n_nodes = graph.number_of_nodes()
    print(format('Number of nodes:', '35s'), n_nodes)
    # number of edges
    n_edges = graph.number_of_edges()
    print(format('Number of edges:', '35s'), n_edges)
    # density
    density = nx.density(graph)
    print(format('Density:', '35s'), density)
    # assortativity
    assortativity = nx.degree_assortativity_coefficient(graph)
    print(format('Assortativity:', '35s'), assortativity)
    # network transitivity
    transitivity = nx.transitivity(graph)    
    print(format('Transitivity:', '35s'), transitivity)
    # average clustering index
    average_clustering = nx.average_clustering(graph)
    print(format('Average clustering:', '35s'), average_clustering)
    # average shortest path length
    average_shortest_path_length = nx.average_shortest_path_length(graph)
    print(format('Average shortest path length:', '35s'), average_shortest_path_length)
    # degree centrality
    centrality_degree = np.array([c for n , c in nx.degree_centrality(graph).items()])
    print(format('Max degree centrality:', '35s'), centrality_degree.max())
    print(format('Min degree centrality:', '35s'), centrality_degree.min())
    print(format('Average degree centrality:', '35s'), centrality_degree.mean())
    # eigenvector centrality
    centrality_eigenvector = np.array([c for n, c in nx.eigenvector_centrality(graph).items()])
    print(format('Max eigenvector centrality:', '35s'), centrality_eigenvector.max())
    print(format('Min eigenvector centrality:', '35s'), centrality_eigenvector.min())
    print(format('Average eigenvector centrality:', '35s'), centrality_eigenvector.mean())
    # betweenness centrality
    centrality_betweenness = np.array([c for n, c in nx.betweenness_centrality(graph).items()])
    print(format('Max betweenness centrality:', '35s'), centrality_betweenness.max())
    print(format('Min betweenness centrality:', '35s'), centrality_betweenness.min())
    print(format('Average betweenness centrality:', '35s'), centrality_betweenness.mean())
    # degree
    degree_array = np.array([d for n, d in graph.degree()])
    print(format('Max degree:', '35s'), degree_array.max())
    print(format('Min degree:', '35s'), degree_array.min())
    print(format('Average degree:', '35s'), degree_array.mean())
    # triangle
    triangle_array = np.array([t for n, t in nx.triangles(graph).items()])
    print(format('Number of triangles:', '35s'), triangle_array.sum() / 3)
    print(format('Max number of triangles:', '35s'), triangle_array.max())
    print(format('Average number of triangles:', '35s'), triangle_array.mean())
    # core
    kcore_array = np.array([c for n, c in nx.core_number(graph).items()])
    print(format('Max k-core number:', '35s'), kcore_array.max())
    
    description = {
        'n_nodes':n_nodes,
        'n_edges':n_edges,
        'density':density,
        'assortativity':assortativity,
        'transitivity':transitivity,
        'average_clustering':average_clustering,
        'average_shortest_path_length':average_shortest_path_length,
        'centrality_degree':centrality_degree,
        'centrality_eigenvector':centrality_eigenvector,
        'centrality_betweenness':centrality_betweenness,
        'df':pd.DataFrame({'degree':degree_array, 'triangle':triangle_array, 'kcore':kcore_array, 'centrality_degree':centrality_degree, 'centrality_eigenvector':centrality_eigenvector, 'centrality_betweenness':centrality_betweenness})
    }
    return description


''' pairplot of (degree, triangle, kcore) '''
def plot_pairplot(description, save=False):
    sns.pairplot(description['df'])
    if save:
        plt.savefig('figures/pairplot.png')
    plt.show()
    return
    

''' distribution plot and cdf plot of feature '''
def plot_feature(description, feature, log=False, save=False, figsize=(12, 10)):
    # get feature
    feature_count = np.c_[sorted(Counter(description['df'][feature]).items(), key=lambda x: x[0])]
    feature_count = feature_count.astype(np.float64)
    if log:
        xlabel = ''.join(['log_of_', feature]) 
        ylabel = 'log_of_counts'
        feature_count[:, 0] = np.log(feature_count[:, 0])
        feature_count[:, 1] = np.log(feature_count[:, 1])
    else:
        xlabel = feature 
        ylabel = 'counts'

    # dist
    plt.figure()
    f, ax = plt.subplots(figsize=figsize)
    plt.scatter(feature_count[:, 0], feature_count[:, 1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(''.join([xlabel, '_distribution']))
    if save:
        plt.savefig(''.join(['figures/', xlabel, '_dist.png']))
    plt.show()

    # cdf 
    plt.figure()
    f, ax = plt.subplots(figsize=figsize)
    plt.scatter(feature_count[:, 0], feature_count[:, 1].cumsum() / feature_count[:, 1].sum())
    plt.xlabel(xlabel)
    plt.title(''.join([xlabel, '_cdf']))
    if save:
        plt.savefig(''.join(['figures/', xlabel, '_cdf.png']))
    plt.show()
 
    return 


''' Modularity algorithm '''
def func_modularity(graph):
    cm = nx_comm.greedy_modularity_communities(graph)
    cm = tuple(set(e) for e in cm)
    return cm 


''' Label propagation algorithm '''
def func_propagation(graph):
    # cm = nx_comm.label_propagation_communities(graph)
    cm = nx_comm.asyn_lpa_communities(graph)
    cm = tuple(set(e) for e in cm)
    return cm 


''' Girvan-Newman algorithm '''
def func_newman(graph):
    cm = nx_comm.centrality.girvan_newman(graph)
    cm = tuple(set(e) for e in next(cm))
    return cm


''' Louvain algorithm '''
def func_louvain(graph):
    cm = lv_comm.best_partition(graph)
    cm_dict = {}
    for n, c in cm.items():
        if c not in cm_dict:
            cm_dict[c] = [n]
        else:
            cm_dict[c].append(n)
    cm = tuple(set(e) for e in cm_dict.values())
    return cm 


''' measuring identification performance by unsupervised methods '''
def measure_performance(graph, cm):
    print(format('Community numbers:', '35s'), len(cm))
    comm_len = []
    for c in cm:
        comm_len.append(len(c))
    comm_len.sort(reverse=True)
    print(format('', '35s'), comm_len)
    coverage = nx_comm.quality.coverage(graph, cm)
    modularity = nx_comm.modularity(graph, cm)
    performance = nx_comm.quality.performance(graph, cm)
    print(format('Coverage:', '35s'), coverage)
    print(format('Modularity:', '35s'), modularity)
    print(format('Performance:', '35s'), performance)
    return {'coverage': coverage, 'modularity': modularity, 'performance': performance}


''' node2label raw '''
def node2label_raw(comm):
    label_num = len(comm)
    nodes_num = 0
    for i in range(label_num):
        nodes_num = max(nodes_num, max(comm[i]))
    nodes_num += 1
    node_label = np.zeros(nodes_num, dtype=np.int32)
    for i in range(label_num):
        for node in comm[i]:
            node_label[node] = i
    return list(node_label)


''' node2label adjust '''
def node2label_adjust(comm):
    node_label = []
    comm_nums = len(comm)
    for i in range(comm_nums):
        for _ in range(len(comm[i])):
            node_label.append(i)
    return node_label


''' visualize the original graph '''
def plot_graph(graph, save=False, figsize=(20, 15)):
    f, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(graph)
    nx.draw(graph, with_labels=False, node_size=7, width=0.1)
    if save:
        plt.savefig('figures/network.png')
    plt.show()
    return 


''' visualize community (raw) '''
def plot_community_raw(graph, cm, comm_method, save=False, figsize=(20, 15)):
    node_label = node2label_raw(cm)
    f, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=False, node_size=7, width=0.1, node_color=node_label)
    if save:
        plt.savefig(''.join(['figures/', comm_method, '_raw.png']))
    plt.show()
    return 


''' visualize community (adjust nodes' location) '''
def plot_community_adjust(graph, cm, comm_method, save=False, figsize=(20, 15)):
    node_label = node2label_adjust(cm)
    G = nx.Graph()
    for c in cm:
        G.update(nx.subgraph(graph, c))
    f, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G)
    # pos = nx.spring_layout(G, seed=4, k=0.33)
    nx.draw(graph, pos, with_labels=False, node_size=7, width=0.1, alpha=0.2)
    nx.draw(G, pos, with_labels=False, node_size=7, width=0.1, node_color=node_label)
    if save:
        plt.savefig(''.join(['figures/', comm_method, '_adjust.png']))
    plt.show()
    return



if __name__ == '__main__':

    # random graph
    # graph = nx.erdos_renyi_graph(100, 0.3)

    # create graph
    os.chdir('community_identification')
    file_name = 'lastfm_asia_edges.csv'
    graph = create_graph(file_name)

    # visualize original graph
    plot_graph(graph, save=True)

    # features of graph
    description = describe_graph(graph)

    # pairplot
    plot_pairplot(description, save=True)

    # distribution plot and cdf plot
    # log_of_degree
    plot_feature(description, 'degree', True, save=True)
    # degree
    plot_feature(description, 'degree', save=True)
    # log_of_triangle
    plot_feature(description, 'triangle', True, save=True)
    # triangle
    plot_feature(description, 'triangle', save=True)
    # lof_of_kcore
    plot_feature(description, 'kcore', True, save=True)
    # kcore
    plot_feature(description, 'kcore', save=True)

    # community identifying: modularity method
    cm_modularity = func_modularity(graph)
    print('--------------- Quality of Modularity ---------------')
    quality_modularity = measure_performance(graph, cm_modularity)
    plot_community_raw(graph, cm_modularity, 'modularity', save=True)
    plot_community_adjust(graph, cm_modularity, 'modularity', save=True)

    # community identifying: label propagation method
    cm_propagation = func_propagation(graph)
    print('--------------- Quality of propagation ---------------')
    quality_propagation = measure_performance(graph, cm_propagation)
    plot_community_raw(graph, cm_propagation, 'propagation', save=True)
    plot_community_adjust(graph, cm_propagation, 'propagation', save=True)

    # community identifying: Girvan-Newman method
    # cm_newman = func_newman(graph)
    # print('--------------- Quality of Girvan-Newman ---------------')
    # quality_newman = measure_performance(graph, cm_newman)
    # plot_community_raw(graph, cm_newman, 'newman', save=True)
    # plot_community_adjust(graph, cm_newman, 'newman', save=True)

    # community identifying: Louvain method
    cm_louvain = func_louvain(graph)
    print('--------------- Quality of Louvain ---------------')
    quality_louvain = measure_performance(graph, cm_louvain)
    plot_community_raw(graph, cm_louvain, 'louvain', save=True)
    plot_community_adjust(graph, cm_louvain, 'louvain', save=True)