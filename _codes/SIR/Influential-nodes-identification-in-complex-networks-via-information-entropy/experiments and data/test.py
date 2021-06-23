# %%
import pickle
from tqdm import tqdm
import numpy as np
# import matplotlib.pyplot as plt
# from algorithms import *
import networkx as nx
import sys
import random
sys.path.append(r'c:\\Users\\rotim\\OneDrive\\Documents\\Reading\\graph-code\\Skoltech-PhD-Thesis\\_codes\\SIR\\Influential-nodes-identification-in-complex-networks-via-information-entropy')

# %%


def compute_probability(Source_G):
    """compute the infection probability
    # Arguments
        Source_G: a graph as networkx Graph
    Returns
        the infection probability computed by  formula: <k> / (<k^2> - <k>)
    """
    G = nx.Graph()
    G = Source_G
    degree_dict = G.degree()
    k = 0.0
    k_pow = 0.0
    for i, v in degree_dict:
        k = k + v
        k_pow = k + v * v

    k = k / G.number_of_nodes()
    k_pow = k_pow / G.number_of_nodes()
    pro = k / (k_pow - k)
    return pro


# %%
data_file = 'CEnew'  # 'HepPh'
G = nx.read_adjlist(data_file)
G.remove_edges_from(nx.selfloop_edges(G))
nodes = list(nx.nodes(G))
for node in nodes:
    if G.degree(node) == 0:
        G.remove_node(node)

# %%
infect_prob = compute_probability(G) * 1.5
print(infect_prob)

# %%
degree_dict = G.degree()
k = 0.0
k_pow = 0.0
# deg_list = list(degree_dict)[:5]
for i, v in degree_dict:
    k = int(k) + v
    print(i, v, k)

#     k = k + degree_dict[i]
#     k_pow = k_pow + degree_dict[i] * degree_dict[i]

# k = k / G.number_of_nodes()
# k_pow = k_pow / G.number_of_nodes()
# pro = k / (k_pow - k)

# %%


def voterank(G, topk):
    """use the voterank to get topk nodes
     # Arguments
         g: a graph as networkx Graph
         topk: how many nodes will be returned
     Returns
         return the topk nodes by voterank, [(node1, score), (node2, score ), ...]
     """

    k_ = 1 / (nx.number_of_edges(G) * 2 / nx.number_of_nodes(G))
    result_rank = []
    voting_ability = {}

    node_scores = {}
    for node in nx.nodes(G):
        voting_ability[node] = 1
        node_scores[node] = G.degree(node)

    for i in range(topk):
        selected_node, score = max(node_scores.items(), key=lambda x: x[1])
        result_rank.append((selected_node, score))

        weaky = voting_ability[selected_node]
        node_scores.pop(selected_node)
        voting_ability[selected_node] = 0

        for nbr in nx.neighbors(G, selected_node):
            weaky2 = k_
            voting_ability[nbr] -= k_
            if voting_ability[nbr] < 0:
                weaky2 = abs([voting_ability[nbr]])
                voting_ability[nbr] = 0
            if nbr in node_scores[nbr]:
                node_scores[nbr] -= weaky
            for nbr2 in nx.neighbors(G, nbr):
                if nbr2 in node_scores:
                    node_scores[nbr2] -= weaky2
    return result_rank


# %%
data_file = 'router'
G = nx.read_adjlist(data_file)
G.remove_edges_from(nx.selfloop_edges(G))
for node in nx.nodes(G):
    if G.degree(node) == 0:
        G.remove_node(node)
# %%
print(nx.number_of_nodes(G), nx.number_of_edges(G))

# %%


def degree(g, topk):
    """use the degree to get topk nodes
     # Arguments
         g: a graph as networkx Graph
         topk: how many nodes will be returned
     Returns
         return the topk nodes by degree, [(node1, ' '), (node2, '' ), ...]
     """
    degree_rank = nx.degree_centrality(g)
    degree_rank = sorted(degree_rank.items(), key=lambda x: x[1], reverse=True)
    rank1 = []
    for node, score in degree_rank:
        rank1.append(node)
        if len(rank1) == topk:
            for i in range(len(rank1)):
                rank1[i] = (rank1[i], ' ')
            return rank1
    return rank1

# %%


def get_topk(result, topk):
    """return the topk nodes
    # Arguments
        result: a list of result, [(node1, centrality), (node1, centrality), ...]
        topk: how much node will be returned

    Returns
        topk nodes as a list, [node1, node2, ...]
    """
    result_topk = []
    for i in range(topk):
        result_topk.append(result[i][0])
    return result_topk


def degree_non(g, topk):
    """use the degree_non to get topk nodes
     # Arguments
         g: a graph as networkx Graph
         topk: how many nodes will be returned
     Returns
         return the topk nodes by degree_non, [(node1, ' '), (node2, '' ), ...]
     """
    degree_rank = nx.degree_centrality(g)
    degree_rank = sorted(degree_rank.items(), key=lambda x: x[1], reverse=True)
    rank1 = []
    rank2 = []
    for node, score in degree_rank:
        nbrs = nx.neighbors(g, node)
        if len(set(nbrs) & set(rank1)) == 0:
            rank1.append(node)
            if len(rank1) == topk:
                for i in range(len(rank1)):
                    rank1[i] = (rank1[i], ' ')
                return rank1
        else:
            rank2.append(node)
    rank1.extend(rank2)
    for i in range(len(rank1)):
        rank1[i] = (rank1[i], ' ')
    return rank1


def degree(g, topk):
    """use the degree to get topk nodes
     # Arguments
         g: a graph as networkx Graph
         topk: how many nodes will be returned
     Returns
         return the topk nodes by degree, [(node1, ' '), (node2, '' ), ...]
     """
    degree_rank = nx.degree_centrality(g)
    degree_rank = sorted(degree_rank.items(), key=lambda x: x[1], reverse=True)
    rank1 = []
    for node, score in degree_rank:
        rank1.append(node)
        if len(rank1) == topk:
            for i in range(len(rank1)):
                rank1[i] = (rank1[i], ' ')
            return rank1
    return rank1


# %%


def SIR(g, infeacted_set, infect_prob, cover_prob, max_iter):
    """Perform once simulation
    # Arguments
        g: a graph as networkx Graph
        infeacted_set: the initial node set to simulate, [node1, node2, ...]
        infect_prob: the infection probability
        cover_prob: the cover probability,
        max_iter: maximum number of simulation steps
    Returns
        time: the max time step in this simulation
        time_count_dict: record the scale of infection at each step, {1:5, 2:20, ..., time: scale, ...}
    """
    time = 0
    time_count_dict = {}
    time_count_dict[time] = len(infeacted_set)
    # infeacted_set = infeacted_set
    node_state = {}
    covered_set = set()

    for node in nx.nodes(g):
        if node in infeacted_set:
            node_state[node] = 'i'
        else:
            node_state[node] = 's'

    while len(infeacted_set) != 0 and max_iter != 0:
        ready_to_cover = []
        ready_to_infeact = []
        for node in infeacted_set:
            nbrs = nx.neighbors(g, node)
            nbr = np.random.choice(nbrs)
            if random.uniform(0, 1) <= infect_prob and node_state[nbr] == 's':
                node_state[nbr] = 'i'
                ready_to_infeact.append(nbr)
            if random.uniform(0, 1) <= cover_prob:
                ready_to_cover.append(node)
        for node in ready_to_cover:
            node_state[node] = 'r'
            infeacted_set.remove(node)
            covered_set.add(node)
        for node in ready_to_infeact:
            infeacted_set.append(node)
        max_iter -= 1
        time += 1
        time_count_dict[time] = len(covered_set) + len(infeacted_set)
    return time, time_count_dict


def get_sir_result(G, rank, topk, avg, infect_prob, cover_prob, max_iter):
    """perform SIR simulation
    # Arguments
        G: a graph as networkx Graph
        rank: the initial node set to simulate, [(node1, centrality), (node1, centrality), ...]
        topk: use the topk nodes in rank to simulate
        avg: simulation times, multiple simulation to averaging
        infect_prob: the infection probability
        cover_prob: the cover probability,
        max_iter: maximum number of simulation steps

    Returns
        average simulation result, a 1-D array, indicates the scale of the spread of each step
    """
    time_num_dict_list = []
    time_list = []

    for i in list(range(avg)):
        time, time_num_dict = SIR(G, get_topk(
            rank, topk), infect_prob, cover_prob, max_iter)
        time_num_dict_list.append(list(time_num_dict.values()))
        time_list.append(time)

    max_time = max(time_list) + 1
    result_matrix = np.zeros((len(time_num_dict_list), max_time))
    for index, (row, time_num_dict) in enumerate(zip(result_matrix, time_num_dict_list)):
        row[:] = time_num_dict[-1]
        row[0:len(time_num_dict)] = time_num_dict
        result_matrix[index] = row
    return np.mean(result_matrix, axis=0)


# %%
max_ = 0.03
max_topk = round(max_ * nx.number_of_nodes(G))
print(max_topk)
degreerank = degree(G, max_topk)
print('done!')
degreerank_non = degree_non(G, max_topk)
print('done!')

# vote = voterank(G, max_topk)
# print('done!')
# vote_non = voterank_non(G, max_topk)
# print('done!')
infect_prob = compute_probability(G) * 1.5
avg = 100
max_iter = 200000
atio = 1.5
cover_prob = infect_prob / atio
topk = max_topk

# %%
degreerank_result = get_sir_result(
    G, degreerank, topk, avg, infect_prob, cover_prob, max_iter) / nx.number_of_nodes(G)
print('done!')

# %%
