# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from algorithms import *  # isort:skip
from networkx.algorithms.assortativity import neighbor_degree
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
data_file = 'topo'  # 'CEnew' # pd.read_csv("topo.txt", sep=" ") #   # 'HepPh'
G = clean_data(data_file)
print(nx.number_of_nodes(G), nx.number_of_edges(G))

# %%
geo_loc_data = get_geo_data("Internet2LatLong.csv")
edge_geo_data_combined = assign_location(G, geo_loc_data)
# set edge attributes
set_edge_attr(G, edge_geo_data_combined)
# %%
# # draw graph
nx.draw(G, with_labels=True)
# pos = nx.spring_layout(G, k=2)
# nx.draw_networkx(G, pos)
# labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# plt.show()
# %% [markdown]
# Experiment
# %%
tmp_t = list(range(1, 5))
tmp_t_SN = [{k: [(i, len(n_neighbor(G, i, k))) for (i, j) in sorted(list(G.degree()),
                                                                    key=lambda item: int(item[0]))]}
            for k in tmp_t]
tmp_t_hub = [{k: [(i, sum(hub_information(G, i, k))) for (i, j) in sorted(list(G.degree()),
                                                                          key=lambda item: int(item[0]))]}
             for k in tmp_t]
tmp_t_SN_1, tmp_t_SN_2, tmp_t_SN_3, tmp_t_SN_4 = tmp_t_SN
tmp_t_hub_1, tmp_t_hub_2, tmp_t_hub_3, tmp_t_hub_4 = tmp_t_hub
# %%
k_max, k_min, k_2_max, k_2_min, sigma, delta = maxi_mini(
    tmp_t_SN_1[1], tmp_t_hub_2[2])


# %%
w_d_h, w_d_2_h, w_d_l, w_d_2_l, w_d_t, w_d_2_t = probability_weights(
    tmp_t_SN_1[1], tmp_t_hub_2[2], k_max, k_min, k_2_max, k_2_min, sigma, delta)
# %%
combined_dict, combined_dict_k_2 = covert_to_dict(
    w_d_h, w_d_l, w_d_t), covert_to_dict(w_d_2_h, w_d_2_l, w_d_2_t)
print(combined_dict, "\n ...... \n", combined_dict_k_2)

# %%
evidence_result_D_2SN = [{k: evidence(v['h'], v['l'], v['t'], v2['h'], v2['l'], v2['t']) for k, v in x.items() for k2, v2 in y.items() if k2 == k}
                         for x in combined_dict for y in combined_dict_k_2]
# %%
ranked_nodes = [{k: {'l': v['l'], 'h': v['h'], 'D_2SN': v['h']-v['l']} for k, v in x.items()if (v['h']-v['l']) >= 0}  # if (v['h']-v['l'])>=0
                for x in evidence_result_D_2SN]
ranked_nodes
