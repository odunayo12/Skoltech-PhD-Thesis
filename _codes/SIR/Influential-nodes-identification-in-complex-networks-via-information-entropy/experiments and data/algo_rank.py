# %%
# %%
from collections import Counter
import pandas as pd
from math import cos, asin, sqrt, pi
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
data_file = 'topo'  # 'CEnew'  # 'HepPh'
G = nx.read_adjlist(data_file)
G.remove_edges_from(nx.selfloop_edges(G))
nodes = list(nx.nodes(G))
for node in nodes:
    if G.degree(node) == 0:
        G.remove_node(node)
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
df2 = pd.read_csv("topo", sep="\t", encoding='unicode_escape',
                  header=None, names=["from", "to"])
df2[["to"]] = df2[["to"]].apply(pd.to_numeric)
df2.tail()

# %%
# draw graph
nx.draw(G, with_labels=True)

# draw degree dsitribution


def n_neighbor(g, id, n_hop):
    node = [id]
    node_visited = set()
    neighbors = []

    while n_hop != 0:
        neighbors = []
        for node_id in node:
            node_visited.add(node_id)
            neighbors += [id for id in g.neighbors(node_id)
                          if id not in node_visited]
        node = neighbors
        n_hop -= 1

        if len(node) == 0:
            return neighbors

    return neighbors


def plot_degree_dist(G, n):
    node = list(G.degree())
    degrees = [len(n_neighbor(G, i, n)) for (i, j) in node]
    plt.hist(degrees, bins=50)
    plt.show()
# %%


# draw n-degree dsitribution
# node = list(G.degree())#[:5]
# node_k_2 = [(i, len(n_neighbor(G, i, 2))) for (i, j) in node]
plot_degree_dist(G, 1)
plot_degree_dist(G, 2)
plot_degree_dist(G, 3)

# %%
max_ = 0.016  # %age of nodes to be selected
max_topk = round(max_ * nx.number_of_nodes(G))  # number of nodes selected
print(max_topk)
# %%

# %%


# def EnRenewRank(G, topk, order):
#     # N - 1
#     all_degree = nx.number_of_nodes(G) - 1
#     # avg degree
#     k_ = nx.number_of_edges(G) * 2 / nx.number_of_nodes(G)
#     # E<k>
#     k_entropy = - k_ * ((k_ / all_degree) * math.log((k_ / all_degree)))

#     # node's information pi
#     node_information = {}
#     for node in nx.nodes(G):
#         information = (G.degree(node) / all_degree)
#         node_information[node] = - information * math.log(information)
#     print(list(node_information)[:5])
#     # node's entropy Ei
#     node_entropy = {}
#     for node in nx.nodes(G):
#         node_entropy[node] = 0
#         for nbr in nx.neighbors(G, node):
#             node_entropy[node] += node_information[nbr]
#     print(list(node_entropy)[:5])
#     rank = []
#     for i in range(topk):
#         # choose the max entropy node
#         max_entropy_node, entropy = max(
#             node_entropy.items(), key=lambda x: x[1])
#         rank.append((max_entropy_node, entropy))
#         cur_nbrs = nx.neighbors(G, max_entropy_node)
#     print(rank, list(cur_nbrs))

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
# newmethod_rank = EnRenewRank(G, max_topk, 1)
# print(newmethod_rank)

# %%
# https://stackoverflow.com/questions/41498576/networkx-weights-on-edges-with-cumulative-sum

# %%
# %%
n_neighbor(G, '1', 2)
# %%
n_neighbor(G, '1', 1)

# %%
n_neighbor(G, '2', 1)

# get node and neighbors
# node_n_nbr = [(i, n_neighbor(G, i, 2)) for (i, j) in node]
# %%
# shows the nodes that connecst 2 parts so far a path exists between them.
#  [1:-1] removes the start and end node e.g. removes "1" and "3"
# Useful for connecting nodes selcted by hubs.
[x[1:-1] for x in nx.edge_disjoint_paths(G, "1", "3")]

# displays the edge data in as (node, destination, {weight})
G.edges(data=True)

# initiate weights by setting all to 1
nx.set_edge_attributes(G, values=1, name='weight')

# assign weights dynamically
# for u, v, d in G.edges(data=True):
#     d['weight'] = int(v)

# %%
# https: // stackoverflow.com/questions/39084661/using-a-list-of-lists-as-a-lookup-table-and-updating-a-value-in-new-list-of-list
# https: // stackoverflow.com/questions/32750257/merge-two-dictionaries-by-key
edge_range = list(range(1, nx.number_of_edges(G)+1))
# [(u,v,d) for u, v, d in G.edges(data=True)]
edge_w_data = [{k: (u, v, d)}
               for k, (u, v, d) in zip(edge_range, G.edges(data=True))]


# %%
for i in edge_w_data:
    for u, v, d in i.values():
        d['weight'] = int(v)*int(u)

# %%
# edgde_w_data
# combimed = {key: [dictionary1[key], dictionary2[key]] for key in dictionary1}
# %%
# combined_w_weights =

# [{key: edgde_w_data[key]} for key in edgde_w_data]
# %%
# https://stackoverflow.com/questions/42272710/networkx-how-to-assign-the-node-coordinates-as-attribute
new_weight_dict = [{k: {v[0]:"", v[1]:"", k:v[2]}
                    for k, v in x.items()} for x in edge_w_data]

# [{k: v2['lat'] for k, v in x.items() for k2, v2 in geo_loc_data.items() if k2 == int()}
#  for x in new_weight_dict]
# [{k: {v[0]:"", v[1]:k2['lat'], k:v[2]} for k, v in x.items() for k1, k2 in y.items() if k1 == int(k)} for x in edgde_w_data for y in geo_loc_data ]

# %%
h = nx.path_graph(3)

nx.nodes(h)
h.edges(data=True)
attrs = {0: {"attr1": 20, "attr2": "nothing"}, 1: {"attr2": 3}}
nx.set_node_attributes(h, attrs)
h.nodes(data=True)
# nx.nodes(h)
# h.nodes[0]["attr2"]
# %%
node_attribute = {str(k): v for k, v in geo_loc_data.items()}
# %%
nx.set_node_attributes(G, node_attribute, 'coord')

# %%
G.nodes(data=True)
# %%

G.edges(data=True)


# %%
for (k) in G.edges():
    for k2, v2 in node_attribute.items():
        if k[0] == k2:
            print()


# %%

# edge_geo_data = {k: {k[0]: v2, k[1]: v3} for k in G.edges(
# ) for k2, v2 in node_attribute.items() for k3, v3 in v2.items() if k[0] == k2 or k[1] == k2}
edge_geo_data_from = {k: {k[0]: v2} for k in G.edges(
) for k2, v2 in node_attribute.items() if k[0] == k2}  # or k[1] == k2

edge_geo_data_to = {k: {k[1]: v2} for k in G.edges(
) for k2, v2 in node_attribute.items() if k[1] == k2}  # or
edge_geo_data_combined = {
    k: (edge_geo_data_from[k], edge_geo_data_to[k]) for k in edge_geo_data_from}
# %%


def distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * \
        cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    dist_ = round(12742 * asin(sqrt(a))/0.62137/200, 3)
    return dist_


# %%
lat_1, long_1, lat_2, long_2 = edge_geo_data_combined[('1', '2')][0]['1']['lat'], edge_geo_data_combined[(
    '1', '2')][0]['1']['long'], edge_geo_data_combined[('1', '2')][1]['2']['lat'], edge_geo_data_combined[('1', '2')][1]['2']['long']
distance(lat_1, long_1, lat_2, long_2)

# %%
# new_weight_dict = [{k: {v[0]:"", v[1]:"", k:v[2]}
#                     for k, v in x.items()} for x in edge_w_data]
# [[v for k,v in x.items()] for d,x in edge_geo_data_combined.items()]
# [v1 for v1 in v for k,v in  edge_geo_data_combined.items()]
[tuple((k1, v['lat'], v['long']) for k, v in v1[1].items())
 for k1, v1 in edge_geo_data_combined.items()]


# %%
