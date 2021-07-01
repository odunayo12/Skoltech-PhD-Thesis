# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import random
from algorithms import *
import pickle
from tqdm import tqdm
import numpy as np
import networkx as nx
import sys
sys.path.append(
    r'c:\\Users\\rotim\\OneDrive\\Documents\\Reading\\graph-code\\Skoltech-PhD-Thesis\\_codes\\SIR\\Influential-nodes-identification-in-complex-networks-via-information-entropy')

# %%

# import matplotlib.pyplot as plt

# %%
data_file = 'CEnew'  # 'HepPh'
G = nx.read_adjlist(data_file)
G.remove_edges_from(nx.selfloop_edges(G))
nodes = list(nx.nodes(G))
for node in nodes:
    if G.degree(node) == 0:
        G.remove_node(node)


# %%
print(nx.number_of_nodes(G), nx.number_of_edges(G))


# %%
a, b = {'a': 0.5, 'ab': 0.5}, {'b': 0.3, 'abc': 0.6, 'c': 0.1}


# %%
combi_res = DSCombination(a, b)
combi_res


# %%
node = list(G.degree())[:5]
node_k_2 = [(i, len(n_neighbor(G, i, 2))) for (i, j) in node]
print(node, node_k_2)


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
print(w_d_t, w_d_2_l)


# %%
combined_dict, combined_dict_k_2 = covert_to_dict(
    w_d_h, w_d_l, w_d_t), covert_to_dict(w_d_2_h, w_d_2_l, w_d_2_t)
# print((combined_dict), "\n ...... \n", combined_dict_k_2)


# %%
sample_DS_combi = [DSCombination(a, b)
                   for a, b in zip(combined_dict, combined_dict_k_2)]
print((sample_DS_combi))
