# %%
from algorithms import *  # isort:skip
import networkx as nx
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from math import cos, asin, sqrt, pi, log
import pandas as pd
from collections import Counter
import sys
sys.path.append(r'c:\\Users\\rotim\\OneDrive\\Documents\\Reading\\graph-code\\Skoltech-PhD-Thesis\\_codes\\SIR\\Influential-nodes-identification-in-complex-networks-via-information-entropy')
# %%
# %%

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

# %%


# draw n-degree dsitribution
# node = list(G.degree())#[:5]
# node_k_2 = [(i, len(n_neighbor(G, i, 2))) for (i, j) in node]
plot_degree_dist(G, 1)
plot_degree_dist(G, 2)
plot_degree_dist(G, 3)

# %%
max_ = 1  # 0.016  # %age of nodes to be selected
max_topk = round(max_ * nx.number_of_nodes(G))  # number of nodes selected
print(max_topk)
# %%

# %%
newmethod_rank = hub_information(G, 2)  # , max_topk, 1
print(newmethod_rank)

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
edge_w_data = [{k: (u, v, {'weight': int(v)*int(u)})}
               for k, (u, v, d) in zip(edge_range, G.edges(data=True))]

# %%
# https://stackoverflow.com/questions/42272710/networkx-how-to-assign-the-node-coordinates-as-attribute
new_weight_dict = [{k: {v[0]:"", v[1]:"", k:v[2]}
                    for k, v in x.items()} for x in edge_w_data]


# %%
node_attribute = {str(k): v for k, v in geo_loc_data.items()}

edge_geo_data_from = {k: {k[0]: v2} for k in G.edges(
) for k2, v2 in node_attribute.items() if k[0] == k2}  # or k[1] == k2

edge_geo_data_to = {k: {k[1]: v2} for k in G.edges(
) for k2, v2 in node_attribute.items() if k[1] == k2}  # or
edge_geo_data_combined = {
    k: (edge_geo_data_from[k], edge_geo_data_to[k]) for k in edge_geo_data_from}
# %%


# %%
lat_1, long_1, lat_2, long_2 = edge_geo_data_combined[('1', '2')][0]['1']['lat'], edge_geo_data_combined[(
    '1', '2')][0]['1']['long'], edge_geo_data_combined[('1', '2')][1]['2']['lat'], edge_geo_data_combined[('1', '2')][1]['2']['long']
distance(lat_1, long_1, lat_2, long_2)
# %%

attr = {k: {'weight': distance(f['lat'], f['long'], t['lat'], t['long']) for f in v[0].values() for t in v[1].values()}
        for k, v in edge_geo_data_combined.items()}

# %%
nx.set_edge_attributes(G, attr)
# %%
# draw graph
# nx.draw(G, with_labels=True)
pos = nx.spring_layout(G, k=2)
nx.draw_networkx(G, pos)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

# %%


def path_length(G, nodes):
    w = 0
    for ind, nd in enumerate(nodes[1:]):
        prev = nodes[ind]
        w += G[prev][nd]['weight']
    return w


# %%
path_length(G, [1, 2])
# %%


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


# %%

# %%
sum([nx.shortest_path_length(G, "1", j, weight='weight')
     for j in n_neighbor(G, "1", 2)])

# %%
