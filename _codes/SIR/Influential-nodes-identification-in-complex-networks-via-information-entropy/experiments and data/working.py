# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%


from algorithms import *  # isort:skip
import random
from typing import Dict
import pickle
from tqdm import tqdm
import numpy as np
import networkx as nx
import pandas as pd
import sys  # isort:skip
import os  # isort:skip
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
df = pd.read_csv("Internet2LatLong.csv")
df = df.reset_index()
df = df.rename({"index": "id", "Latitude ": "lat",
                "Longitude ": "long"}, axis=1)
df = df.drop('Unnamed: 0', 1)
df['id'] = df.id + 1
# df.head()
geo_loc_data = df.set_index("id").to_dict(orient="index")
# %%
node_attribute = {str(k): v for k, v in geo_loc_data.items()}

edge_geo_data_from = {k: {k[0]: v2} for k in G.edges(
) for k2, v2 in node_attribute.items() if k[0] == k2}  # or k[1] == k2

edge_geo_data_to = {k: {k[1]: v2} for k in G.edges(
) for k2, v2 in node_attribute.items() if k[1] == k2}  # or
edge_geo_data_combined = {
    k: (edge_geo_data_from[k], edge_geo_data_to[k]) for k in edge_geo_data_from}

# %%

attr = {k: {'weight': distance(f['lat'], f['long'], t['lat'], t['long']) for f in v[0].values() for t in v[1].values()}
        for k, v in edge_geo_data_combined.items()}
# set edge attributes
nx.set_edge_attributes(G, attr)
# %%
# # draw graph
nx.draw(G, with_labels=True)
# pos = nx.spring_layout(G, k=2)
# nx.draw_networkx(G, pos)
# labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# plt.show()
# %%
# #  sort nodes
# node = [(i, j) for i, j in sorted(list(G.degree()),
#                                   key=lambda item: int(item[0]))]  # [:5]
# node_strength_k_1 = [(i, len(n_neighbor(G, i, 2))) for (i, j) in node]
# node_info_k_1 = hub_information(G, 1)  # stop
# node_info_k_2 = hub_information(G, 2)
# node_k_ = [(i, len(n_neighbor(G, i, 2))) for (i, j) in node]
# node_k_2 = [(i[0],i[1]+j[1]) for i,j in zip(node_k_,node_info_k_1.items()) if i[0] ==j[0]] #[(i, j) for i, j in node_info_k_.items()]  # [:5]
# # print(node, node_k_2)
# k_max, k_min, k_2_max, k_2_min = max([j for i, j in node]), min(
#     [j for i, j in node]), max([j for i, j in node_k_2]), min([j for i, j in node_k_2])
# mu, epsilon = 0.15, 0.15
# sigma = k_max-k_min+(2*mu)
# delta = k_2_max-k_2_min+(2*epsilon)

# w_d_h, w_d_2_h = [(i, abs(k-k_min)/sigma) for (i, k)
#                   in node], [(i, abs(k-k_2_min)/delta) for (i, k) in node_k_2]
# # print(w_d_h)
# w_d_l, w_d_2_l = [(i, abs(k-k_max)/sigma) for (i, k)
#                   in node], [(i, abs(k-k_2_max)/delta) for (i, k) in node_k_2]
# # print(w_d_i)
# # w_d_t, w_d_2_t = [(i, (1-(y + z))) for i, y in w_d_h for k, z in w_d_l if i[0] ==
# #                   k[0]], [(i, (1-(y + z))) for i, y in w_d_2_h for k, z in w_d_2_l if i[0] == k[0]]
# w_d_t, w_d_2_t = [(i, 1-(abs(k-k_min)/sigma + abs(k-k_max)/sigma))
#                   for i, k in node], [(i, 1-(abs(k-k_2_min)/delta + abs(k-k_2_max)/delta))
#                                       for i, k in node_k_2]
# print(w_d_t, "\n ...... \n", w_d_2_t)
# # [(i, (1-(y + z))) for i, y in w_d_h for k, z in w_d_l if i[0] ==
# #  k[0]]
# Experiment
# %%
d = [(i, j) for i, j in sorted(list(G.degree()),
                               key=lambda item: int(item[0]))]
two_SN = [(i, len(n_neighbor(G, i, 2))) for (i, j) in d]
# n_s = hub_information(G, 1)
# node__ = [(i, len(n_neighbor(G, i, 1))) for (i, j) in d]
d_plus_two_SN = [(i[0], i[1]+j[1]) for i, j in zip(two_SN, d) if i[0] == j[0]]

# %%
k_max, k_min, k_2_max, k_2_min = max([j for i, j in d]), min(
    [j for i, j in d]), max([j for i, j in d_plus_two_SN]), min([j for i, j in d_plus_two_SN])  # two_SN
mu, epsilon = 0.15, 0.15
sigma = k_max-k_min+(2*mu)
delta = k_2_max-k_2_min+(2*epsilon)


# %%
w_d_h, w_d_2_h = [(i, abs(k-k_min)/sigma) for (i, k)
                  in d], [(i, abs(k-k_2_min)/delta) for (i, k) in two_SN]
# print(w_d_h)
w_d_l, w_d_2_l = [(i, abs(k-k_max)/sigma) for (i, k)
                  in d], [(i, abs(k-k_2_max)/delta) for (i, k) in two_SN]
# print(w_d_i)
# w_d_t, w_d_2_t = [(i, (1-(y + z))) for i, y in w_d_h for k, z in w_d_l if i[0] ==
#                   k[0]], [(i, (1-(y + z))) for i, y in w_d_2_h for k, z in w_d_2_l if i[0] == k[0]]
w_d_t, w_d_2_t = [(i, 1-(abs(k-k_min)/sigma + abs(k-k_max)/sigma))
                  for i, k in d], [(i, 1-(abs(k-k_2_min)/delta + abs(k-k_2_max)/delta))
                                   for i, k in two_SN]
print(w_d_t, "\n ...... \n", w_d_2_t)
# [(i, (1-(y + z))) for i, y in w_d_h for k, z in w_d_l if i[0] ==
#  k[0]]
# %%
[(i[0], 1-(i[1]+j[1])) for i, j in zip(w_d_h, w_d_l)]
# [(i,j) for i,j in  zip(w_d_h,w_d_l)]
# %%
combined_dict, combined_dict_k_2 = covert_to_dict(
    w_d_h, w_d_l, w_d_t), covert_to_dict(w_d_2_h, w_d_2_l, w_d_2_t)
print(combined_dict, "\n ...... \n", combined_dict_k_2)


# %%
# evidence_t_prob = [{k: DSCombination(a, b) for k, a in x.items() for b in y.values()}
#                    for x in combined_dict for y in combined_dict_k_2]

# evidence_t_prob

# %%
# D_2SN = [{k:{v['h'], v['l']} for k,v in x.items()}for x in evidence_t_prob]
# # %%
# [{k: {'l': v['l'], 'h': v['h'], 'D_2SN': v['h']-v['l']} for k, v in x.items() if (v['h']-v['l']) >= 0}  # if (v['h']-v['l'])>=0
#  for x in evidence_t_prob]

# %%


def evidence(w_d_h, w_d_l, w_d_t, w_d_2_h, w_d_2_l, w_d_2_t):
    k = (w_d_h*w_d_2_l) + (w_d_l*w_d_2_h)
    h = ((w_d_h*w_d_2_h)+(w_d_h*w_d_2_t)+(w_d_2_h*w_d_t))/(1-k)
    l = ((w_d_l*w_d_2_l)+(w_d_l*w_d_2_t)+(w_d_2_l*w_d_t))/(1-k)
    t = (w_d_t*w_d_2_t)/(1-k)
    evi_result = dict(zip(("h", "l", "t"), (h, l, t)))
    return evi_result


# %%
evidence(0.0, 0.9090909090909092, 0.09090909090909083,
         0.18867924528301888, 0.7547169811320755, 0.0566037735849056)
evidence(0.9090909090909092, 0.0, 0.09090909090909083,
         0.37735849056603776, 0.5660377358490566, 0.05660377358490565)
# %%
evidence_result_D_2SN = [{k: evidence(v['h'], v['l'], v['t'], v2['h'], v2['l'], v2['t']) for k, v in x.items() for k2, v2 in y.items() if k2 == k}
                         for x in combined_dict for y in combined_dict_k_2]

# %%

# %%
ranked_nodes = [{k: {'l': v['l'], 'h': v['h'], 'D_2SN': v['h']-v['l']} for k, v in x.items()if (v['h']-v['l']) >= 0}  # if (v['h']-v['l'])>=0
                for x in evidence_result_D_2SN]
# %%
tmp_t = list(range(1, 5))
tmp_t_SN = [{k: [(i, len(n_neighbor(G, i, k))) for (i, j) in d]}
            for k in tmp_t]
# [(i[0], i[1]+j[1]) for i, j in zip(two_SN, d) if i[0] == j[0]]
# [[(v[0], v[1]+j) for i,j in d for k,v in x.items() ]for x in tmp_t_SN]
[{k: [(v_i, v_j+j) for v_i, v_j in v] for i, j in d for k, v in x.items()}
    for x in tmp_t_SN]  # if i == v[k][0] (i, v[k][1]+j) for i, j in d
# %%
