# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import networkx as nx
from method import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle


# %%
data_file = 'HepPh'
G = nx.read_adjlist(data_file)
G.remove_edges_from(G.selfloop_edges())
for node in nx.nodes(G):
    if G.degree(node) == 0:
        G.remove_node(node)


# %%
print(nx.number_of_nodes(G), nx.number_of_edges(G))


# %%
r = [0.0015, 0.003, 0.0045, 0.006, 0.0075, 0.009]
topk_list = []
for k in r:
    topk = round(nx.number_of_nodes(G) * k)
    print(k, topk)
    topk_list.append(topk)


# %%
max_topk = round(r[-1] * nx.number_of_nodes(G))
newmethod_1_rank = EnRenewRank(G, max_topk, 1)
print('done')
newmethod_2_rank = EnRenewRank(G, max_topk, 2)
print('done')
newmethod_3_rank = EnRenewRank(G, max_topk, 3)
print('done')
newmethod_4_rank = EnRenewRank(G, max_topk, 4)
print('done')


# %%
infect_prob = compute_probability(G) * 1.5
atio = 1.5
cover_prob = infect_prob / atio
avg = 100
max_iter = 200000

# %% [markdown]
# ## Different order

# %%
get_ipython().run_cell_magic('time', '', 'newmethod_1_rankresult = []\nnewmethod_2_rankresult = []\nnewmethod_3_rankresult = []\nnewmethod_4_rankresult = []\n\n\nfor k in tqdm(topk_list):\n    topk = k\n    newmethod_1_rankresult.append(get_sir_result(G, newmethod_1_rank, topk, avg, infect_prob, cover_prob, max_iter)[-1] / nx.number_of_nodes(G))\n    newmethod_2_rankresult.append(get_sir_result(G, newmethod_2_rank, topk, avg, infect_prob, cover_prob, max_iter)[-1] / nx.number_of_nodes(G))\n    newmethod_3_rankresult.append(get_sir_result(G, newmethod_3_rank, topk, avg, infect_prob, cover_prob, max_iter)[-1] / nx.number_of_nodes(G))\n    newmethod_4_rankresult.append(get_sir_result(G, newmethod_4_rank, topk, avg, infect_prob, cover_prob, max_iter)[-1] / nx.number_of_nodes(G))')


# %%
get_ipython().run_line_magic('matplotlib', 'notebook')
plt.title('network:{}   avg:{}  i/r:{}   infect_prob:{}'.format(data_file,
                                                                avg, atio, infect_prob))
plt.plot(np.array(topk_list) / nx.number_of_nodes(G),
         newmethod_1_rankresult, 'r-o', label='order=1')
plt.plot(np.array(topk_list) / nx.number_of_nodes(G),
         newmethod_2_rankresult, 'b-o', label='order=2')
plt.plot(np.array(topk_list) / nx.number_of_nodes(G),
         newmethod_3_rankresult, 'y-o', label='order=3')
plt.plot(np.array(topk_list) / nx.number_of_nodes(G),
         newmethod_4_rankresult, 'g-o', label='order=4')
plt.legend()
plt.show()


# %%
