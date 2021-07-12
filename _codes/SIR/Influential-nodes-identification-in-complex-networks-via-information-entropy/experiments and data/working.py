# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
import pickle
from typing import Dict
import random
from algorithms import *
import sys

import pandas
dir_path = r'c:\\Users\\rotim\\OneDrive\\Documents\\Reading\\graph-code\\Skoltech-PhD-Thesis\\_codes\\SIR\\Influential-nodes-identification-in-complex-networks-via-information-entropy'
sys.path.append(dir_path)
# %%
# %%
data_file = 'topo'  # 'CEnew' # pd.read_csv("topo.txt", sep=" ") #   # 'HepPh'
G = nx.read_adjlist(data_file)
G.remove_edges_from(nx.selfloop_edges(G))
nodes = list(nx.nodes(G))
for node in nodes:
    if G.degree(node) == 0:
        G.remove_node(node)
# print(G)

# %%
print(nx.number_of_nodes(G), nx.number_of_edges(G))


# %%
a, b = {'a': 0.5, 'ab': 0.5}, {'b': 0.3, 'abc': 0.6, 'c': 0.1}


# %%
combi_res = DSCombination(a, b)
combi_res


# %%
node = list(G.degree())[:5]
node_k_2 = [(i, len(n_neighbor(G, i, 1))) for (i, j) in node]
# print(node, node_k_2)


# %%
k_max, k_min, k_2_max, k_2_min = max([j for i, j in node]), min(
    [j for i, j in node]), max([j for i, j in node_k_2]), min([j for i, j in node_k_2])
mu, epsilon = 0.15, 0.15
sigma, sigma_k_2 = k_max-k_min+(2*mu), k_2_max-k_2_min+(2*mu)
delta, delta_k_2 = k_max-k_min+(2*epsilon), k_2_max-k_2_min+(2*epsilon)


# %%
w_d_h, w_d_2_h = [(i, abs(k-k_min)/sigma) for (i, k)
                  in node], [(i, abs(k-k_2_min)/sigma) for (i, k) in node_k_2]
# print(w_d_h)
w_d_l, w_d_2_l = [(i, abs(k-k_max)/delta) for (i, k)
                  in node], [(i, abs(k-k_2_max)/sigma) for (i, k) in node_k_2]
# print(w_d_i)
w_d_t, w_d_2_t = [(i, (1-(y + z))) for i, y in w_d_h for k, z in w_d_l if i[0] ==
                  k[0]], [(i, (1-(y + z))) for i, y in w_d_2_h for k, z in w_d_2_l if i[0] == k[0]]
# w_d_t = [(i,1-(abs(k-k_min)/sigma + abs(k-k_max)/delta)) for i,k in node]
# print(w_d_t, w_d_2_l)


# %%
combined_dict, combined_dict_k_2 = covert_to_dict(
    w_d_h, w_d_l, w_d_t), covert_to_dict(w_d_2_h, w_d_2_l, w_d_2_t)
print((combined_dict), "\n ...... \n", combined_dict_k_2)


# %%
# Dic1 = {'a': {'a': 0.5, 'ab': 0.5}, 'ab': {'a': 0.5, 'ab': 0.5}}
# Dic1a, Dic1b = {'a': {'a': 0.5, 'ab': 0.5}}, {'a': {'a': 0.5, 'ab': 0.5}}
# Dic2 = {'b': 0.3, 'abc': 0.6, 'c': 0.1}
# Dic3 = {'a': 0.59, 'ab': 0.5}
# Dic4 = {'a': {'a': 0.5, 'ab': 0.5}, 'ab': {'b': 0.3, 'abc': 0.6, 'c': 0.1}}

# %%

def Dict_DSCombination(d1, d2):
    set_dict = set(d1.keys()).union(set(d2.keys()))
    Result_dict = dict.fromkeys(set_dict, 0)
    print(set_dict, Result_dict)
    print(d1.values())
    # d1_values  = list(d1.values())
    # d2_values = list(d2.values())
    # d1__keys = [d.keys() for d in d1_values]
    # d2__keys = [d.keys() for d in d2_values]
    set_inner_dict_1_val = [item for item in d1.values()]
    set_inner_dict_2_val = [item for item in d2.values()]
    set_inner_dict_1 = [key for item in d1.values() for key in item.keys()]
    set_inner_dict_2 = [key for item in d2.values() for key in item.keys()]
    set_inner_dict = set(set_inner_dict_1).union(set(set_inner_dict_2))
    Result = dict.fromkeys(set_inner_dict, 0)
    print(set_inner_dict, Result)
    # [d for d in d1__values[1].keys()]
    for a, b in d1.items():
        for c, d in d2.items():
            # print(a,b,c,d)
            for i in b.keys():
                for j in d.keys():
                    print(i, j)
                    if set(str(i)).intersection(set(str(j))) == set(str(i)):
                        Result[i] += set_inner_dict_1_val[i] * \
                            set_inner_dict_2_val[j]
                    elif set(str(i)).intersection(set(str(j))) == set(str(j)):
                        Result[j] += set_inner_dict_1_val[i] * \
                            set_inner_dict_2_val[j]

            # if isinstance(v, dict) and isinstance(v2, dict):
            #     sets = set(v.keys()).union(set(v2.keys()))
            #     # print(k)
            #     # if k == k2:
            #     # print("Set:", sets)
            #     print("params:", v, v2)
            #     #print("keys:", set_dict)
            #     c
            #     Result_dict = dict.fromkeys(set_dict, 0)
            #     # print(Result, Result_dict[k])  #
            #     for i in v.keys():
            #         for j in v2.keys():
            #             if set(str(i)).intersection(set(str(j))) == set(str(i)):
            #                 Result[i] += v[i]*v2[j]
            #             elif set(str(i)).intersection(set(str(j))) == set(str(j)):
            #                 Result[j] += v[i]*v2[j]
            #         print(Result)
            #         print(Result, Result_dict)
            #     # normalize the results
            #     f = sum(list(Result.values()))
            #     # print(f)
            #     for i in Result.keys():
            #         for k in Result_dict.keys():
            #             Result[i] /= f
            #             Result_dict[k] = Result
            #     return Result_dict
            # # else:
            # #     print("{0} : {1}".format(k, v))


# %%
# Dic1.items()
# Dict_DSCombination(Dic1a, Dic1b)
sample_DS_combi = [Dict_DSCombination(a, b)
                   for a, b in zip(combined_dict, combined_dict_k_2)]
# sample_DS_combi = [Dict_DSCombination(a, b)
#                    for a in combined_dict for b in combined_dict_k_2]
print((sample_DS_combi))
# %%


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
