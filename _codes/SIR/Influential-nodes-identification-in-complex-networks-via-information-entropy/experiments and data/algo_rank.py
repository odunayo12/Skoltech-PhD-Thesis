# %%
# %%
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
import networkx as nx
import sys
sys.path.append(r'c:\\Users\\rotim\\OneDrive\\Documents\\Reading\\graph-code\\Skoltech-PhD-Thesis\\_codes\\SIR\\Influential-nodes-identification-in-complex-networks-via-information-entropy')
# %%

# from algorithms import *  # isort:skip
# %%
data_file = 'CEnew'  # 'HepPh'
G = nx.read_adjlist(data_file)
G.remove_edges_from(nx.selfloop_edges(G))
nodes = list(nx.nodes(G))
for node in nodes:
    if G.degree(node) == 0:
        G.remove_node(node)
print(nx.number_of_nodes(G), nx.number_of_edges(G))
# %%
max_ = 0.016  # %age of nodes to be selected
max_topk = round(max_ * nx.number_of_nodes(G))  # number of nodes selected
print(max_topk)
# %%

# %%


def EnRenewRank(G, topk, order):
    # N - 1
    all_degree = nx.number_of_nodes(G) - 1
    # avg degree
    k_ = nx.number_of_edges(G) * 2 / nx.number_of_nodes(G)
    # E<k>
    k_entropy = - k_ * ((k_ / all_degree) * math.log((k_ / all_degree)))

    # node's information pi
    node_information = {}
    for node in nx.nodes(G):
        information = (G.degree(node) / all_degree)
        node_information[node] = - information * math.log(information)
    print(list(node_information)[:5])
    # node's entropy Ei
    node_entropy = {}
    for node in nx.nodes(G):
        node_entropy[node] = 0
        for nbr in nx.neighbors(G, node):
            node_entropy[node] += node_information[nbr]
    print(list(node_entropy)[:5])
    rank = []
    for i in range(topk):
        # choose the max entropy node
        max_entropy_node, entropy = max(
            node_entropy.items(), key=lambda x: x[1])
        rank.append((max_entropy_node, entropy))
        cur_nbrs = nx.neighbors(G, max_entropy_node)
    print(rank, list(cur_nbrs))

    #     for o in range(order):
    #         for nbr in cur_nbrs:
    #             if nbr in node_entropy:
    #                 node_entropy[nbr] -= (node_information[max_entropy_node] /
    #                                       k_entropy) / (2**o) #removes the max entropy node
    #         next_nbrs = []
    #         for node in cur_nbrs:
    #             nbrs = nx.neighbors(G, node)
    #             next_nbrs.extend(nbrs)
    #         cur_nbrs = next_nbrs

    #     #set the information quantity of selected nodes to 0
    #     node_information[max_entropy_node] = 0
    #     # set entropy to 0
    #     node_entropy.pop(max_entropy_node)
    # return list(node_entropy)[:5] # rank


# %%
newmethod_rank = EnRenewRank(G, max_topk, 1)
print(newmethod_rank)

# %%
# https://stackoverflow.com/questions/41498576/networkx-weights-on-edges-with-cumulative-sum
ints = [1] * 5
a = ['A', 'B', 'C', 'A', 'A']
b = ['D', 'A', 'E', 'D', 'B']
df['a'] = a
df['b'] = b
df
df["b'"] = pd.DataFrame([df["a"], df["b"]]).max()
df = df.groupby(by=["a'", "b'"]).sum().reset_index()
G = nx.from_pandas_edgelist(df, "a'", "b'", ['weight'])
[G[u][v]['weight'] for u, v in G.edges()]
