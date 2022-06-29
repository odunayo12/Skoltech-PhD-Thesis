# Copyright (c) 2019 Chungu Guo. All rights reserved.
import collections
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from networkx.algorithms.shortest_paths import weighted
import numpy as np
from tqdm import tqdm
import copy
import random
from math import cos, asin, sqrt, pi, log
import pandas as pd
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


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

    for i in range(avg):
        time, time_num_dict = SIR(G, get_topk(
            rank, topk), infect_prob, cover_prob, max_iter)
        time_num_dict_list.append((time_num_dict.values()))
        time_list.append(time)
    # time_num_dict = time_num_dict[0]
    max_time = max(time_list) + 1
    result_matrix = np.zeros((len(time_num_dict_list), max_time))
    for index, (row, time_num_dict) in enumerate(zip(result_matrix, time_num_dict_list)):
        row[:] = list(time_num_dict)[-1]
        row[0:len(time_num_dict)] = list(time_num_dict)
        result_matrix[index] = row
    return np.mean(result_matrix, axis=0)


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
    for i, v in list(degree_dict):
        k = k + v  # degree_dict[v]
        k_pow = k_pow + v * v  # degree_dict[v] * degree_dict[v]

    k = k / G.number_of_nodes()
    k_pow = k_pow / G.number_of_nodes()
    pro = k / (k_pow - k)
    return pro


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
    infeacted_set = infeacted_set
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
            nbrs = list(nx.neighbors(g, node))
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


def degree_non(g, topk):
    """use the degree_non to get topk nodes
     # Arguments
         g: a graph as networkx Graph
         topk: how many nodes will be returned
     Returns
         return the topk nodes by degree_non, [(node1, ' '), (node2, ' ' ), ...]
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


def voterank_non(G, topk):
    """use the voterank_non to get topk nodes
     # Arguments
         g: a graph as networkx Graph
         topk: how many nodes will be returned
     Returns
         return the topk nodes by voterank_non, [(node1, ' '), (node2, '' ), ...]
     """
    k_ = 1 / (nx.number_of_edges(G) * 2 / nx.number_of_nodes(G))
    result_rank = []
    voting_ability = {}

    node_scores = {}
    for node in nx.nodes(G):
        voting_ability[node] = 1
        node_scores[node] = G.degree(node)

    rank2 = []
    for index in range(nx.number_of_nodes(G)):
        selected_node, score = max(node_scores.items(), key=lambda x: x[1])
        nbrs = nx.neighbors(G, selected_node)
        if len(set(nbrs) & set(result_rank)) == 0:
            result_rank.append(selected_node)
            if len(result_rank) == topk:
                for i in range(len(result_rank)):
                    result_rank[i] = (result_rank[i], ' ')
                return result_rank
        else:
            rank2.append(selected_node)
        weaky = voting_ability[selected_node]
        node_scores.pop(selected_node)
        voting_ability[selected_node] = 0

        for nbr in nx.neighbors(G, selected_node):
            weaky2 = k_
            voting_ability[nbr] -= k_
            if voting_ability[nbr] < 0:
                weaky2 = abs(voting_ability[nbr])
                voting_ability[nbr] = 0
            if nbr in node_scores:
                node_scores[nbr] -= weaky
            for nbr2 in nx.neighbors(G, nbr):
                if nbr2 in node_scores:
                    node_scores[nbr2] -= weaky2
    result_rank.extend(rank2)
    for i in range(len(result_rank)):
        result_rank[i] = (result_rank[i], ' ')
    return result_rank


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

        for nbr in list(nx.neighbors(G, selected_node)):
            weaky2 = k_
            voting_ability[nbr] -= k_
            if voting_ability[nbr] < 0:
                weaky2 = abs([voting_ability[nbr]])
                voting_ability[nbr] = 0
            if nbr in [node_scores[nbr]]:
                node_scores[nbr] -= weaky
            for nbr2 in nx.neighbors(G, nbr):
                if nbr2 in node_scores:
                    node_scores[nbr2] -= weaky2
    return result_rank


def kshell_non(G, topk):
    """use the kshell_non to get topk nodes
     # Arguments
         g: a graph as networkx Graph
         topk: how many nodes will be returned
     Returns
         return the topk nodes by kshell_non, [(node1, ' '), (node2, ' '), ...]
     """
    node_core = nx.core_number(G)
    core_node_list = {}
    for node in node_core:
        if node_core[node] not in core_node_list:
            core_node_list[node_core[node]] = []
        core_node_list[node_core[node]].append((node, nx.degree(G, node)))

    for core in core_node_list:
        core_node_list[core] = sorted(
            core_node_list[core], key=lambda x: x[1], reverse=True)
    core_node_list = sorted(core_node_list.items(),
                            key=lambda x: x[0], reverse=True)
    kshellrank = []
    for core, node_list in core_node_list:
        kshellrank.extend([n[0] for n in node_list])

    rank = []
    rank2 = []
    for node in kshellrank:
        nbrs = nx.neighbors(G, node)
        if len(set(nbrs) & set(rank)) == 0:
            rank.append(node)
            if len(rank) == topk:
                for i in range(len(rank)):
                    rank[i] = (rank[i], ' ')
                return rank
        else:
            rank2.append(node)
    rank.extend(rank2)
    for i in range(len(rank)):
        rank[i] = (rank[i], ' ')
    return rank


def kshell(G, topk):
    """use the kshell to get topk nodes
     # Arguments
         g: a graph as networkx Graph
         topk: how many nodes will be returned
     Returns
         return the topk nodes by kshell, [(node1, ' '), (node2, ' '), ...]
     """
    node_core = nx.core_number(G)
    core_node_list = {}
    for node in node_core:
        if node_core[node] not in core_node_list:
            core_node_list[node_core[node]] = []
        core_node_list[node_core[node]].append((node, nx.degree(G, node)))

    for core in core_node_list:
        core_node_list[core] = sorted(
            core_node_list[core], key=lambda x: x[1], reverse=True)
    core_node_list = sorted(core_node_list.items(),
                            key=lambda x: x[0], reverse=True)
    kshellrank = []
    for core, node_list in core_node_list:
        kshellrank.extend([n[0] for n in node_list])

    rank = []
    for node in kshellrank:
        rank.append((node, ' '))
        if len(rank) == topk:
            return rank


def get_ls(g, infeacted_set):
    """compute the average shortest path in the initial node set
     # Arguments
         g: a graph as networkx Graph
         infeacted_set: the initial node set
     Returns
         return the average shortest path
     """
    dis_sum = 0
    path_num = 0
    for u in infeacted_set:
        for v in infeacted_set:
            if u != v:
                try:
                    dis_sum += nx.shortest_path_length(g, u, v)
                    path_num += 1
                except:
                    dis_sum += 0
                    path_num -= 1
    return dis_sum / path_num


def EnRenewRank(G, topk, order):
    """use the our method to get topk nodes
     # Arguments
         g: a graph as networkx Graph
         topk: how many nodes will be returned
     Returns
         return the topk nodes by EnRenewRank, [(node1, score), (node2, score), ...]
     """

    all_degree = nx.number_of_nodes(G) - 1
    k_ = nx.number_of_edges(G) * 2 / nx.number_of_nodes(G)
    k_entropy = - k_ * ((k_ / all_degree) * log((k_ / all_degree)))

    # node's information pi
    node_information = {}
    for node in nx.nodes(G):
        information = (G.degree(node) / all_degree)
        node_information[node] = - information * log(information)

    # node's entropy Ei
    node_entropy = {}
    for node in nx.nodes(G):
        node_entropy[node] = 0
        for nbr in nx.neighbors(G, node):
            node_entropy[node] += node_information[nbr]

    rank = []
    for i in range(topk):
        # choose the max entropy node
        max_entropy_node, entropy = max(
            node_entropy.items(), key=lambda x: x[1])
        rank.append((max_entropy_node, entropy))

        cur_nbrs = nx.neighbors(G, max_entropy_node)
        for o in range(order):
            for nbr in cur_nbrs:
                if nbr in node_entropy:
                    node_entropy[nbr] -= (node_information[max_entropy_node] /
                                          k_entropy) / (2**o)
            next_nbrs = []
            for node in cur_nbrs:
                nbrs = nx.neighbors(G, node)
                next_nbrs.extend(nbrs)
            cur_nbrs = next_nbrs

        # set the information quantity of selected nodes to 0
        node_information[max_entropy_node] = 0
        # delete max_entropy_node
        node_entropy.pop(max_entropy_node)
    return rank


def mini_maxi(a):
    """returns the minimum and maximum value of a list
    """
    min_, max_ = min(a), max(a)
    return f"min: {min_}, max: {max_}"


def DSCombination(Dic1, Dic2):
    """
    # Arguments
        Dic1={'a':0.5, 'ab':0.5}
        Dic2={'b':0.3, 'abc':0.6, 'c':0.1}
    Returns
        returns the D-S combination rule of Dic1 of Dic2 {'a': 0.39999999999999997,'ab': 0.39999999999999997,'abc': 0.0,'b': 0.19999999999999998,'c': 0.0}
    """
    # extract the frame dicernment
    sets = set(Dic1.keys()).union(set(Dic2.keys()))
    Result = dict.fromkeys(sets, 0)
    # Combination process
    for i in Dic1.keys():
        for j in Dic2.keys():
            if set(str(i)).intersection(set(str(j))) == set(str(i)):
                Result[i] += Dic1[i]*Dic2[j]
            elif set(str(i)).intersection(set(str(j))) == set(str(j)):
                Result[j] += Dic1[i]*Dic2[j]

     # normalize the results
    f = sum(list(Result.values()))
    for i in Result.keys():
        Result[i] /= f
    return Result


def covert_to_dict(h, l, t):
    """converts tuple to dictionary

    Args:
        h ([list]): [contains low weights]
        l ([type]): [description]
        t ([type]): [description]

    Returns:
        [list]: [description]
    """
    l = dict(l)
    t = dict(t)
    weight_dict = {}
    weight_dict_list = []
    for i, v in h:
        weight_dict[i] = {"h": v, "l": l.get(i, ""), "t": t.get(i, "")}
    weight_dict_list.append(weight_dict)
    return weight_dict_list


def n_neighbor(g, node, dist):
    dist_range = list(range(1, dist+1))
    dist_range = sorted(dist_range, reverse=True)
    neighbors_set = set()

    for i in dist_range:
        neighbors_set |= nx.descendants_at_distance(g, node, distance=i)
    neighbors_set = list(neighbors_set)
    return sorted(neighbors_set, key=lambda x: int(x[0]))


def hub_information(G, node, dist):
    """gets the sum of weigths of edges within a specified dist = 1, 2, 3, ...

    Args:
        G (graph): graph of networkx
        dist (int): lenght sought

    Returns:
        node_hub_information: dictionary of each node with corresponding nth length weight
    """
    node_information = n_neighbor(
        G, node, dist)
    node_hub_information = [nx.shortest_path_length(
        G, node, k, weight='weight') for k in node_information]
    return node_hub_information


def distance(lat1, lon1, lat2, lon2):
    """returns the geo-locational distance between two nodes.

    Args:
        lat1 (float): latitude of node 1.
        lon1 (float): longitude of node 1.
        lat2 (float): latitude of node 2.
        lon2 (float): longitude of node 2.

    Returns:
        dist_: a number distance between two nodes given their coordinates.
    """
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * \
        cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    dist_ = round(12742 * asin(sqrt(a))/0.62137/200, 3)
    return dist_


def plot_degree_dist(G, n):
    """plots degree distribution by given nth-length: n = 1, 2, 3,...

    Args:
        G (Graph):  graph as networkx Graph
        n (int): nth-length
    """
    node = list(G.degree())
    degrees = [len(n_neighbor(G, i, n)) for (i, j) in node]
    plt.hist(degrees, bins=50)
    plt.show()


def evidence(w_d_h, w_d_l, w_d_t, w_d_2_h, w_d_2_l, w_d_2_t):
    k = (w_d_h*w_d_2_l) + (w_d_l*w_d_2_h)
    h = ((w_d_h*w_d_2_h)+(w_d_h*w_d_2_t)+(w_d_2_h*w_d_t))/(1-k)
    l = ((w_d_l*w_d_2_l)+(w_d_l*w_d_2_t)+(w_d_2_l*w_d_t))/(1-k)
    t = (w_d_t*w_d_2_t)/(1-k)
    evi_result = dict(zip(("h", "l", "t"), (h, l, t)))
    return evi_result


def get_geo_data(data_):
    df = pd.read_csv(data_)
    df = df.reset_index()
    df = df.rename({"index": "id", "Latitude ": "lat",
                    "Longitude ": "long"}, axis=1)
    df = df.drop('Unnamed: 0', 1)
    df['id'] = df.id + 1
# df.head()
    geo_loc_data = df.set_index("id").to_dict(orient="index")
    return geo_loc_data


def assign_location(G, geo_loc_data):
    node_attribute = {str(k): v for k, v in geo_loc_data.items()}

    edge_geo_data_from = {k: {k[0]: v2} for k in G.edges(
    ) for k2, v2 in node_attribute.items() if k[0] == k2}

    edge_geo_data_to = {k: {k[1]: v2} for k in G.edges(
    ) for k2, v2 in node_attribute.items() if k[1] == k2}
    edge_geo_data_combined = {
        k: (edge_geo_data_from[k], edge_geo_data_to[k]) for k in edge_geo_data_from}

    return edge_geo_data_combined


def set_edge_attr(G, edge_geo_data_combined):
    attr = {k: {'weight': distance(f['lat'], f['long'], t['lat'], t['long']) for f in v[0].values() for t in v[1].values()}
            for k, v in edge_geo_data_combined.items()}
    # set edge attributes
    set_attr = nx.set_edge_attributes(G, attr)

    return set_attr  # list(G.nodes(data=True))


def clean_data(data_file):
    G = nx.read_adjlist(data_file)
    G.remove_edges_from(nx.selfloop_edges(G))
    nodes = list(nx.nodes(G))
    for node in nodes:
        if G.degree(node) == 0:
            G.remove_node(node)
    return G


# def maxi_mini(a, b, epsilon=0.15, mu=0.15):
#     k_max, k_min, k_2_max, k_2_min = max([j for i, j in a]), min(
#         [j for i, j in a]), max([j for i, j in b]), min([j for i, j in b])  # two_SN
#     sigma = k_max-k_min+(2*mu)
#     delta = k_2_max-k_2_min+(2*epsilon)
#     return k_max, k_min, k_2_max, k_2_min, sigma, delta

def maxi_mini(a=[], b=[], c=[], d=[],  epsilon=0.15, mu=0.15, no_of_evidences=2):
    def maxi_(l):
        return max([j for i, j in l])

    def mini_(k):
        return min([j for i, j in k])

    sigma = maxi_(a)-mini_(a) + (2*mu)
    delta = maxi_(b)-mini_(b) + (2*epsilon)
    if (no_of_evidences == 2):
        return maxi_(a), mini_(a), maxi_(b), mini_(b), sigma, delta
    elif (no_of_evidences == 3):
        return maxi_(a), mini_(a), maxi_(b), mini_(b), maxi_(c), mini_(c), sigma, delta
    elif (no_of_evidences == 4):
        return maxi_(a), mini_(a), maxi_(b), mini_(b), maxi_(c), mini_(c),  maxi_(d), mini_(d), sigma, delta


def probability_weights(d, two_SN, k_max, k_min, k_2_max, k_2_min, sigma, delta):
    w_d_h, w_d_2_h = [(i, abs(k-k_min)/sigma) for (i, k)
                      in d], [(i, abs(k-k_2_min)/delta) for (i, k) in two_SN]
    w_d_l, w_d_2_l = [(i, abs(k-k_max)/sigma) for (i, k)
                      in d], [(i, abs(k-k_2_max)/delta) for (i, k) in two_SN]
    w_d_t, w_d_2_t = [(i, 1-(abs(k-k_min)/sigma + abs(k-k_max)/sigma))
                      for i, k in d], [(i, 1-(abs(k-k_2_min)/delta + abs(k-k_2_max)/delta))
                                       for i, k in two_SN]

    return w_d_h, w_d_2_h, w_d_l, w_d_2_l, w_d_t, w_d_2_t


def probability_weights_multi(*maxi_mini_results, e_1=[], e_2=[], e_3=[], e_4=[], number_of_evidences=2):
    """Assign probability weights based on each variable and source of evidence.

    Args:
        e_1 (list, optional): a list of tuples containing rsults of rating of individual evidence. Defaults to [].
        e_2 (list, optional): a list of tuples containing rsults of rating of individual evidence. Defaults to [].
        e_3 (list, optional): a list of tuples containing rsults of rating of individual evidence. Defaults to [].
        e_4 (list, optional): a list of tuples containing rsults of rating of individual evidence. Defaults to [].
        number_of_evidences (int, optional): number of sources of evdences. Defaults to 2.
    """
    def pby_weights(a, b, c):
        return [(i, abs(k-a)/b) if b != 0 else (i, 0) for (i, k) in c]
    maxi_mini_results = list(*maxi_mini_results)
    w_d_h, w_d_2_h = pby_weights(maxi_mini_results[1], maxi_mini_results[4], e_1), pby_weights(
        maxi_mini_results[3], maxi_mini_results[5], e_2)
    w_d_l, w_d_2_l = pby_weights(maxi_mini_results[0], maxi_mini_results[4], e_1), pby_weights(
        maxi_mini_results[2], maxi_mini_results[5], e_2)
    w_d_t, w_d_2_t = [(i[0], 1-(i[1]+j[1])) for i, j in zip(w_d_h, w_d_l)
                      ], [(i[0], 1-(i[1]+j[1])) for i, j in zip(w_d_2_h, w_d_2_l)]
    # w_d_3_h = pby_weights(maxi_mini_results[5], maxi_mini_results[6], e_3)

    if (number_of_evidences == 2):
        return w_d_h, w_d_l, w_d_t, w_d_2_h, w_d_2_l, w_d_2_t
    elif (number_of_evidences == 3):
        w_d_3_h = pby_weights(maxi_mini_results[5], maxi_mini_results[6], e_3)
        w_d_3_l = pby_weights(maxi_mini_results[4], maxi_mini_results[6], e_3)
        w_d_3_t = [(i[0], 1-(i[1]+j[1])) for i, j in zip(w_d_3_h, w_d_3_l)]
        return w_d_h, w_d_l, w_d_t, w_d_2_h, w_d_2_l, w_d_2_t, w_d_3_h, w_d_3_l, w_d_3_t
    elif (number_of_evidences == 4):
        w_d_3_h, w_d_4_h = pby_weights(maxi_mini_results[5], maxi_mini_results[6], e_3), pby_weights(
            maxi_mini_results[7], maxi_mini_results[9], e_4)
        w_d_3_l, w_d_4_l = pby_weights(maxi_mini_results[4], maxi_mini_results[6], e_3), pby_weights(
            maxi_mini_results[6], maxi_mini_results[9], e_4)
        w_d_3_t, w_d_4_t = [(i[0], 1-(i[1]+j[1])) for i, j in zip(w_d_3_h, w_d_3_l)], [
            (i[0], 1-(i[1]+j[1])) for i, j in zip(w_d_4_h, w_d_4_l)]
        return w_d_h, w_d_l, w_d_t, w_d_2_h, w_d_2_l, w_d_2_t, w_d_3_h, w_d_3_l, w_d_3_t, w_d_4_h, w_d_4_l, w_d_4_t


def evidence_multi(w_d_h, w_d_l, w_d_t, w_d_2_h, w_d_2_l, w_d_2_t, w_d_3_h=0, w_d_3_l=0, w_d_3_t=0,  w_d_4_h=0, w_d_4_l=0, w_d_4_t=0, no_of_evidences=2):
    """

    Args:
        w_d_h (float): weight of evidence 1 associated with decison variable high (h)
        w_d_l (float): weight of evidence 1 associated with decison variable low (l)
        w_d_t (float): weight of evidence 1 associated with decison variable theta (t)
        w_d_2_h (float): weight of evidence 2 associated with decison variable high (h)
        w_d_2_l (float): weight of evidence 2 associated with decison variable low (l)
        w_d_2_t (float): weight of evidence 2 associated with decison variable theta (t)
        w_d_3_h (int, optional): weight of evidence 3 associated with decison variable high (h). Defaults to 0.
        w_d_3_l (int, optional): weight of evidence 3 associated with decison variable low (l). Defaults to 0.
        w_d_3_t (int, optional): weight of evidence 3 associated with decison variable theta (t). Defaults to 0.
        w_d_4_h (int, optional): weight of evidence 4 associated with decison variable high (h). Defaults to 0.
        w_d_4_l (int, optional): weight of evidence 4 associated with decison variable low (l). Defaults to 0.
        w_d_4_t (int, optional): weight of evidence 4 associated with decison variable theta (t). Defaults to 0.
        no_of_evidences (int, optional): number of evidences to be combined. Defaults to 2.

    Returns:
        float: a dictionary of basic assinged probability relative to the decision variables high (h), low (l) and theta (t)
    """
    k_1, k_2 = (w_d_h*w_d_2_l), (w_d_l*w_d_2_h)
    k = k_1 + k_2
    e_3_k_1, e_3_k_2 = k_1 * w_d_3_l, k_2 * w_d_3_h
    e_3_k = e_3_k_1 + e_3_k_2
    e_4_k_1, e_4_k_2 = w_d_4_h * e_3_k_1, w_d_4_l * e_3_k_2
    e_4_k = e_4_k_1 + e_4_k_2
    h_1, h_2, h_3 = (w_d_h*w_d_2_h), (w_d_h*w_d_2_t), (w_d_t*w_d_2_h)
    h = 0 if (1-k) == 0 else (h_1+h_2+h_3)/(1-k)
    e_3_h_1, e_3_h_2, e_3_h_3 = h_1*w_d_3_h, h_2*w_d_3_t, h_3*w_d_3_h
    e_3_h = (e_3_h_1+e_3_h_2+e_3_h_3)/(1-e_3_k)
    e_4_h_1, e_4_h_2, e_4_h_3 = e_3_h_1 * \
        w_d_4_h,  e_3_h_2 * w_d_4_h, e_3_h_3 * w_d_4_t
    e_4_h = (e_4_h_1 + e_4_h_2 + e_4_h_3)/(1-e_4_k)
    l_1, l_2, l_3 = (w_d_l*w_d_2_l), (w_d_l*w_d_2_t), (w_d_2_l*w_d_t)
    l = 0 if (1-k) == 0 else (l_1 + l_2 + l_3)/(1-k)
    e_3_l_1, e_3_l_2, e_3_l_3 = l_1*w_d_3_l, l_2*w_d_3_t, l_3*w_d_3_l
    e_3_l = (e_3_l_1 + e_3_l_2+e_3_l_3)/(1-e_3_k)
    e_4_l_1, e_4_l_2, e_4_l_3 = e_3_l_1*w_d_4_l, e_3_l_2*w_d_4_l, e_3_l_3*w_d_4_t
    e_4_l = (e_4_l_1 + e_4_l_2 + e_4_l_3)/(1-e_4_k)
    t_1 = (w_d_t*w_d_2_t)
    t = 0 if (1-k) == 0 else t_1/(1-k)
    e_3_t_1 = t_1*w_d_3_t
    e_3_t = e_3_t_1/(1-e_3_k)
    e_4_t_1 = e_3_t_1 * w_d_4_t
    e_4_t = e_4_t_1/(1-e_4_k)
    if no_of_evidences == 2:
        evi_result = dict(zip(("h", "l", "t"), (h, l, t)))
        return evi_result
    elif no_of_evidences == 3:
        evi_result = dict(zip(("h", "l", "t"), (e_3_h, e_3_l, e_3_t)))
        return evi_result
    elif no_of_evidences == 4:
        evi_result = dict(zip(("h", "l", "t"), (e_4_h, e_4_l, e_4_t)))
        return evi_result


def hubs_SN_NS(G, tmp_t):
    tmp_t_SN = [{k: [(i, len(n_neighbor(G, i, k))) for (i, j) in sorted(list(G.degree()),
                                                                        key=lambda item: int(item[0]))]}
                for k in tmp_t]
    tmp_t_hub = [{k: [(i, sum(hub_information(G, i, k))) for (i, j) in sorted(list(G.degree()),
                                                                              key=lambda item: int(item[0]))]}
                 for k in tmp_t]

    return tmp_t_SN, tmp_t_hub


def convert_dict_multi(*probability_weights_multi_res):
    """converts weights obtained to labelled dictionary of each node.

    Returns:
        dict: a dictionary of of aggregated probability weights.
    """
    w_1, w_2, w_3, w_4, w_5, w_6, * \
        others = list(*probability_weights_multi_res)
    combined_dict_evidence_1, combined_dict_evidence_2 = covert_to_dict(
        w_1, w_2, w_3), covert_to_dict(w_4, w_5, w_6)
    # return combined_dict_evidence_1
    if len(others) == 0:
        return combined_dict_evidence_1, combined_dict_evidence_2
    elif len(others) == 3:
        combined_dict_evidence_3 = covert_to_dict(
            others[0], others[1], others[2])
        return combined_dict_evidence_1, combined_dict_evidence_2, combined_dict_evidence_3
    elif len(others) == 6:
        combined_dict_evidence_3, combined_dict_evidence_4 = covert_to_dict(
            others[0], others[1], others[2]), covert_to_dict(others[3], others[4], others[5])
        return combined_dict_evidence_1, combined_dict_evidence_2, combined_dict_evidence_3, combined_dict_evidence_4


def varying_examples(tmp_t_SN_1, tmp_t_hub_2):
    k_max, k_min, k_2_max, k_2_min, sigma, delta = maxi_mini(
        tmp_t_SN_1, tmp_t_hub_2)

    w_d_h, w_d_2_h, w_d_l, w_d_2_l, w_d_t, w_d_2_t = probability_weights(
        tmp_t_SN_1, tmp_t_hub_2, k_max, k_min, k_2_max, k_2_min, sigma, delta)
    combined_dict, combined_dict_k_2 = covert_to_dict(
        w_d_h, w_d_l, w_d_t), covert_to_dict(w_d_2_h, w_d_2_l, w_d_2_t)
    return combined_dict, combined_dict_k_2


def varying_examples_multi(s_1, s_2, s_3, s_4, evi):
    maxi_mini_result = maxi_mini(s_1, s_2, s_3, s_4, no_of_evidences=evi)
    probability_weights_multi_res = probability_weights_multi(
        maxi_mini_result, e_1=s_1, e_2=s_2, e_3=s_3, e_4=s_4, number_of_evidences=evi)
    convert_dict_multi_results = convert_dict_multi(
        probability_weights_multi_res)
    # , len(rank_result_multi(convert_dict_multi_results))
    return rank_result_multi(convert_dict_multi_results)


def rank_result(combined_dict, combined_dict_k_2):
    evidence_result_D_2SN = [{k: evidence(v['h'], v['l'], v['t'], v2['h'], v2['l'], v2['t']) for k, v in x.items() for k2, v2 in y.items() if k2 == k}
                             for x in combined_dict for y in combined_dict_k_2]
    ranked_nodes = [{k: {'l': v['l'], 'h': v['h'], 'D_2SN': v['h']-v['l']} for k, v in x.items()}
                    for x in evidence_result_D_2SN]
    ranked_nodes = sorted([(k, v['D_2SN']) for x in ranked_nodes for k, v in x.items(
    )], key=lambda elem: elem[1], reverse=True)
    opti_rank = [(k, v) for k, v in ranked_nodes if v > 0]
    return opti_rank, ranked_nodes


def rank_result_multi(*convert_dict_multi_results):
    """Generates ranked nodes according to predefined rankings.

    Returns:
        dict: dict of ranked nodes based on length (number) of evidences provided.
    """
    convert_dict_multi_results = list(*convert_dict_multi_results)

    def sorter(evidence_results_input):
        ranked_nodes = [{k: {'l': v['l'], 'h': v['h'], 'D_2SN': v['h']-v['l']} for k, v in x.items()}
                        for x in evidence_results_input]
        ranked_nodes = sorted([(k, v['D_2SN']) for x in ranked_nodes for k, v in x.items(
        )], key=lambda elem: elem[1], reverse=True)
        opti_rank = [(k, v) for k, v in ranked_nodes if v > 0]
        return opti_rank  # , ranked_nodes
    if len(convert_dict_multi_results) == 2:
        evidence_result_D_2SN = [{k: evidence_multi(v['h'], v['l'], v['t'], v2['h'], v2['l'], v2['t']) for (k, v), (k2, v2) in zip(x.items(), y.items()) if k2 == k}
                                 for x, y in zip(*convert_dict_multi_results[0:2])]
        return sorter(evidence_result_D_2SN)
    elif len(convert_dict_multi_results) == 3:
        evidence_result_D_2SN = [{k: evidence_multi(v['h'], v['l'], v['t'], v2['h'], v2['l'], v2['t'], v3['h'], v3['l'], v3['t'], no_of_evidences=3)
                                  for (k, v), (k2, v2), (k3, v3) in zip(x.items(), y.items(), z.items()) if {k, k2, k3}}
                                 for x, y, z in zip(*convert_dict_multi_results[0:3])]
        return sorter(evidence_result_D_2SN)
    elif len(convert_dict_multi_results) == 4:
        evidence_result_D_2SN = [{k: evidence_multi(v['h'], v['l'], v['t'], v2['h'], v2['l'], v2['t'], v3['h'], v3['l'], v3['t'], v4['h'], v4['l'], v4['t'], no_of_evidences=4)
                                  for (k, v), (k2, v2), (k3, v3), (k4, v4) in zip(w.items(), x.items(), y.items(), z.items()) if {k, k2, k3, k4}}
                                 for w, x, y, z in zip(*convert_dict_multi_results[0:4])]
        return sorter(evidence_result_D_2SN)


def cluster_optimal_nodes(G, opti_rank, b=1):
    node_list = G.nodes()
    opti_rank_nodes = [i for i, j in opti_rank]
    remainder_nodes = set(node_list) - set(opti_rank_nodes)
    current_set_result_all = []

    def rank_loop(G, opti_rank, b, remainder_nodes, current_set_result_all):
        while remainder_nodes != set():
            chosen_list = [(i, n_neighbor(G, i, b)) for (i, j) in opti_rank]
            for x, y in chosen_list:
                chosen_set = set(y)
                current_set = chosen_set.intersection(remainder_nodes)
                current_set_result = (x, current_set)
                current_set_result_all.append((current_set_result))
                remainder_nodes -= chosen_set
            b += 1

    rank_loop(G, opti_rank, b, remainder_nodes, current_set_result_all)

    coll_ = collections.defaultdict(list)
    for d, e in current_set_result_all:
        coll_[d].extend(e)  # add to existing list or create a new one

    current_set_result_all = list(coll_.items())
    ranked_output = {i: set(j) for i, j in current_set_result_all}
    return ranked_output


def cluster_optimal_nodes_test(G, opti_rank, b=1, is_filtered=False, filter_rank=1000):
    """Clusters the optimal set of nodes provided by the ranking algorithms.

    Args:
        G (Graph): A graph as networkx Graph
        opti_rank (list): a list of tuples, with each tupple cotaining optimal node and their ranking
        b (int, optional): depth of cluster. Defaults to 1.

    Returns:
        list: returns a list of nodes with attached clusters.
    """
    node_list = G.nodes()
    opti_rank_nodes = [i for i, j in opti_rank]
    remainder_nodes = set(node_list) - set(opti_rank_nodes)
    current_set_result_all = []

    coll_ = collections.defaultdict(list)

    def rank_loop(G, opti_rank, b, remainder_nodes, current_set_result_all):
        while remainder_nodes != set():
            chosen_list = [(i, n_neighbor(G, i, b)) for (i, j) in opti_rank]
            for x, y in chosen_list:
                chosen_set = set(y)
                current_set = chosen_set.intersection(remainder_nodes)
                current_set_result = (x, current_set)
                current_set_result_all.append((current_set_result))
                remainder_nodes -= chosen_set
            b += 1

    rank_loop(G, opti_rank, b, remainder_nodes, current_set_result_all)
    non_collated_current_set_result_all = [
        (i, k) for i, k in current_set_result_all if k != set()]

    new_opti_rank = list(
        set([i for i, k in non_collated_current_set_result_all]))
    new_opti_rank_2 = [(i, j)
                       for k, j in opti_rank for i in new_opti_rank if i == k]
    empty_controllers = set(opti_rank_nodes) - set(new_opti_rank)

    rank_loop(G, new_opti_rank_2, b, empty_controllers,
              non_collated_current_set_result_all)

    def factor_results(factor_set):
        for d, e in factor_set:
            coll_[d].extend(e)

        factor_set = list(coll_.items())
        ranked_output = {i: set(j) for i, j in factor_set}
        return ranked_output

    ranked_output = factor_results(non_collated_current_set_result_all)

    # return ranked_output
    if is_filtered == False or filter_rank >= len(ranked_output):
        return ranked_output
    else:
        # pass
        filtered_controllers = {k: v for k, v in [x for x in sorted(
            ranked_output.items(), key=lambda item: len(item[1]), reverse=True)][:filter_rank]}
        current_controller_result_all = [
            (i, j) for i, j in filtered_controllers.items()]
        opti_rank_filtered = [(i, j) for i, j in filtered_controllers.items()]
        empty_controllers = set().union(
            *[(set([i]).union(j)) for i, j in ranked_output.items() if i not in filtered_controllers.keys()])

        rank_loop(G, opti_rank_filtered, b, empty_controllers,
                  current_controller_result_all)
        ranked_output = factor_results(current_controller_result_all)
        return {k: v for k, v in [x for x in ranked_output.items()][:filter_rank]}


def plot_optimal_cluster_test(graph_clusters, g, title, evi, filter_rank=1000):

    len_renadom = len(graph_clusters)
    random_colors = color_generator(len_renadom)
    respective_colors = {
        k: v for k, v in random_colors.items() if k in set(graph_clusters.values())}
    handles__ = [patches.Patch(color=v, label=k)
                 for k, v in respective_colors.items()]
    graph_clusters = dict(
        sorted(graph_clusters.items(), key=lambda x: int(x[0])))
    # print(len(graph_clusters))
    values = [random_colors.get(v, "#000000")
              for k, v in graph_clusters.items()]
    pos = nx.spring_layout(g)
    nx.draw(g, cmap=plt.get_cmap('viridis'), pos=pos, node_color=values,
            with_labels=True, font_color='white')

    title = title.split(".")[0]
    plt.suptitle(f"Graph: {title} \n Sources of Evidences: {evi}, #Nodes: {nx.number_of_nodes(g)}, #Edges: {nx.number_of_edges(g)}, #Controllers: {len(set(graph_clusters.values()))}",
                 fontsize=10, y=0.95)  # f"{title}_{evi}"
    plt.legend(handles=handles__)
    dirpath = Path(f"{Path().absolute()}\images\{title}") if (
        filter_rank == 1000) else Path(f"{Path().absolute()}\images_{filter_rank}\{title}")
    os.makedirs(dirpath, exist_ok=True)
    plt.savefig(Path(f"{dirpath}\{title.lower()}_{evi}.png"))
    # plt.show()
    plt.close()
    return "Done"


def read_graph(file_directory):
    def renameNode(G):
        mapping = dict([(i, str(j))
                        for i, j in zip(G, range(0, len(G.nodes())))])
        G = nx.relabel_nodes(G, mapping)
        return G

    file_path = [str(x) for x in Path(file_directory).iterdir() if x.is_file()]

    graph_list = [nx.Graph(nx.read_graphml(x)) if Path(
        x).suffix == ".graphml" else renameNode(nx.Graph(nx.read_gml(x))) for x in file_path]

    graph_name = [Path(x).stem.upper() for x in file_path]

    graph_summary = {g[0]: {"index": i, "nodes": nx.number_of_nodes(g[1]), "edges": nx.number_of_edges(
        g[1])} for i, g in enumerate(zip(graph_name, graph_list))}
    print(graph_summary)
    return graph_list, graph_name


def selection__from_graph(graph_coll, graph_name_list, sel_=[]):
    """_summary_

    Args:
        graph_coll (_type_): _description_
        sel_ (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    if sel_ != []:
        return [(j, graph_coll[k]) for i, j in enumerate(graph_name_list)
                for k in sel_ if i == k]
    else:
        return [(i, j) for i, j in zip(graph_name_list, graph_coll)]


def closeness_centrality(g):
    return sorted(nx.closeness_centrality(g).items(), key=lambda item: item[1], reverse=True)


def degree_centrality(g):
    return sorted(nx.degree_centrality(g).items(), key=lambda item: item[1], reverse=True)

# eigenvector_centrality


def eccentricity(g):
    return sorted(nx.eccentricity(g).items(), key=lambda item: item[1], reverse=True)


def eigenvector_centrality(g, weight=None):
    return sorted(nx.eigenvector_centrality(g, weight=weight).items(), key=lambda item: item[1], reverse=True)


def load_centrality(g, weight=None):
    return sorted(nx.load_centrality(g, weight=weight).items(), key=lambda item: item[1], reverse=True)


def betweenness_centrality(g, weight=None):
    return sorted(nx.betweenness_centrality(g, weight=weight).items(), key=lambda item: item[1], reverse=True)


def current_flow_betweenness_centrality(g, weight=None):
    return sorted(nx.current_flow_betweenness_centrality(g, weight=weight).items(), key=lambda item: item[1], reverse=True)


def approximate_current_flow_betweenness_centrality(g, weight=None):
    return sorted(nx.approximate_current_flow_betweenness_centrality(g, weight=weight).items(), key=lambda item: item[1], reverse=True)


def k_core(g):
    return sorted(nx.core_number(g).items(), key=lambda item: item[1], reverse=True)


def color_generator(no_colors):
    colors = []
    while len(colors) < no_colors:
        random_number = np.random.randint(0, 16777215)
        hex_number = format(random_number, 'x')
        if len(hex_number) == 6:
            hex_number = '#' + hex_number
            colors.append(hex_number)
    return {str(i): v for i, v in enumerate(colors)}
