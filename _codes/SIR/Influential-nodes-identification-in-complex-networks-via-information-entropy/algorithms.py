# Copyright (c) 2019 Chungu Guo. All rights reserved.
import collections
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.shortest_paths import weighted
import numpy as np
from tqdm import tqdm
import copy
import random
from math import cos, asin, sqrt, pi, log
import pandas as pd
import os
import sys
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
    # all_ = [x[1:]+y[1:]+z[1:]
    #         for x in h for y in l for z in t if x[0] == y[0] == z[0]]
    # all_ = [(x[0], x[1:]+y[1:]+z[1:])
    #         for x in h for y in l for z in t if x[0] == y[0] == z[0]]
    # keys = ['h', 'l', 't']  # ['key', 'value', 'id']
    # com_all_ = [
    #     {x: {key: val for key, val in zip(keys, sub)}} for x, sub in all_]
    # # return com_all_

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
    """gets the sum of weigths of edges within a specified dist = 1, 2, 3, ..., ...

    Args:
        G (graph): graph of networkx
        dist (int): lenght sought

    Returns:
        node_hub_information: dictionary of each node with corresponding nth length weight
    """
    node_information = node_information = n_neighbor(
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
    nx.set_edge_attributes(G, attr)


def clean_data(data_file):
    G = nx.read_adjlist(data_file)
    G.remove_edges_from(nx.selfloop_edges(G))
    nodes = list(nx.nodes(G))
    for node in nodes:
        if G.degree(node) == 0:
            G.remove_node(node)
    return G


def probability_weights(d, two_SN, k_max, k_min, k_2_max, k_2_min, sigma, delta):
    w_d_h, w_d_2_h = [(i, abs(k-k_min)/sigma) for (i, k)
                      in d], [(i, abs(k-k_2_min)/delta) for (i, k) in two_SN]
    w_d_l, w_d_2_l = [(i, abs(k-k_max)/sigma) for (i, k)
                      in d], [(i, abs(k-k_2_max)/delta) for (i, k) in two_SN]
    w_d_t, w_d_2_t = [(i, 1-(abs(k-k_min)/sigma + abs(k-k_max)/sigma))
                      for i, k in d], [(i, 1-(abs(k-k_2_min)/delta + abs(k-k_2_max)/delta))
                                       for i, k in two_SN]

    return w_d_h, w_d_2_h, w_d_l, w_d_2_l, w_d_t, w_d_2_t


def maxi_mini(a, b, epsilon=0.15, mu=0.15):
    k_max, k_min, k_2_max, k_2_min = max([j for i, j in a]), min(
        [j for i, j in a]), max([j for i, j in b]), min([j for i, j in b])  # two_SN
    sigma = k_max-k_min+(2*mu)
    delta = k_2_max-k_2_min+(2*epsilon)
    return k_max, k_min, k_2_max, k_2_min, sigma, delta


def hubs_SN_NS(G, tmp_t):
    tmp_t_SN = [{k: [(i, len(n_neighbor(G, i, k))) for (i, j) in sorted(list(G.degree()),
                                                                        key=lambda item: int(item[0]))]}
                for k in tmp_t]
    tmp_t_hub = [{k: [(i, sum(hub_information(G, i, k))) for (i, j) in sorted(list(G.degree()),
                                                                              key=lambda item: int(item[0]))]}
                 for k in tmp_t]

    return tmp_t_SN, tmp_t_hub


def rank_result(combined_dict, combined_dict_k_2):
    evidence_result_D_2SN = [{k: evidence(v['h'], v['l'], v['t'], v2['h'], v2['l'], v2['t']) for k, v in x.items() for k2, v2 in y.items() if k2 == k}
                             for x in combined_dict for y in combined_dict_k_2]
    ranked_nodes = [{k: {'l': v['l'], 'h': v['h'], 'D_2SN': v['h']-v['l']} for k, v in x.items()}
                    for x in evidence_result_D_2SN]
    ranked_nodes = sorted([(k, v['D_2SN']) for x in ranked_nodes for k, v in x.items(
    )], key=lambda elem: elem[1], reverse=True)
    opti_rank = [(k, v) for k, v in ranked_nodes if v > 0]
    return opti_rank, ranked_nodes


def varying_examples(tmp_t_SN_1, tmp_t_hub_2):
    k_max, k_min, k_2_max, k_2_min, sigma, delta = maxi_mini(
        tmp_t_SN_1, tmp_t_hub_2)

    w_d_h, w_d_2_h, w_d_l, w_d_2_l, w_d_t, w_d_2_t = probability_weights(
        tmp_t_SN_1, tmp_t_hub_2, k_max, k_min, k_2_max, k_2_min, sigma, delta)
    combined_dict, combined_dict_k_2 = covert_to_dict(
        w_d_h, w_d_l, w_d_t), covert_to_dict(w_d_2_h, w_d_2_l, w_d_2_t)
    return combined_dict, combined_dict_k_2


def cluster_optimal_nodes(G, opti_rank, b=1):
    node_list = G.nodes()
    opti_rank_nodes = [i for i, j in opti_rank]
    remainder_nodes = set(node_list) - set(opti_rank_nodes)
    current_set_result_all = []

    while remainder_nodes != set():
        chosen_list = [(i, n_neighbor(G, i, b)) for (i, j) in opti_rank]
        for x, y in chosen_list:
            chosen_set = set(y)
            current_set = chosen_set.intersection(remainder_nodes)
            current_set_result = (x, current_set)
            current_set_result_all.append((current_set_result))
            remainder_nodes -= chosen_set
        b += 1

    coll_ = collections.defaultdict(list)
    for d, e in current_set_result_all:
        coll_[d].extend(e)  # add to existing list or create a new one

    current_set_result_all = list(coll_.items())
    return current_set_result_all
