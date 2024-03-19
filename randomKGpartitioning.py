import dgl
import numpy as np


entities = np.genfromtxt('/content/sample_data/entity2id.txt', dtype=np.dtype('U'), delimiter='\t')


relations = np.genfromtxt('/content/sample_data/relation2id.txt', dtype=np.dtype('U'), delimiter='\t')


train_triples = np.genfromtxt('/content/sample_data/train.txt', dtype=np.int32, delimiter='\t')


valid_triples = np.genfromtxt('/content/sample_data/valid.txt', dtype=np.int32, delimiter='\t')


test_triples = np.genfromtxt('/content/sample_data/test.txt', dtype=np.int32, delimiter='\t')

g = dgl.graph((train_triples[:, 0], train_triples[:, 2]), num_nodes=len(entities))


print(f"Nombre d'entités : {len(entities)}")
print(f"Nombre de relations : {len(relations)}")
print(f"Nombre de triplets d'entraînement : {len(train_triples)}")
print(f"Nombre de triplets de validation : {len(valid_triples)}")
print(f"Nombre de triplets de test : {len(test_triples)}")
import dgl
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


entities = np.genfromtxt('/content/sample_data/entity2id.txt', dtype=np.dtype('U'), delimiter='\t')


relations = np.genfromtxt('/content/sample_data/relation2id.txt', dtype=np.dtype('U'), delimiter='\t')


train_triples = np.genfromtxt('/content/sample_data/train.txt', dtype=np.int32, delimiter='\t')


g = dgl.graph((train_triples[:, 0], train_triples[:, 2]), num_nodes=len(relations))


g_nx = g.to_networkx()
communities = list(greedy_modularity_communities(g_nx))


print(f"Nombre de communautés : {len(communities)}")
for i, c in enumerate(communities):
    print(f"Communauté {i} : {c}")
  nx.draw(g_nx, with_labels=True)


colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
for i, c in enumerate(communities):
    nx.draw_networkx_nodes(g_nx, pos=nx.spring_layout(g_nx), nodelist=list(c), node_color=colors[i % len(colors)], alpha=0.7)

plt.show()

df = pd.read_csv('/content/sample_data/train.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])


k = 3


indices = np.random.permutation(df.index)


partitions = np.array_split(indices, k)


for i, part in enumerate(partitions):
    print(f"Partition {i}: {part}")
df = pd.read_csv('/content/sample_data/train.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])


k = 3


indices = np.random.permutation(df.index)

partitions = np.array_split(indices, k)


for i, part in enumerate(partitions):

    partition_df = df.iloc[part]
    
    num_triplets = len(partition_df)
   
    print(f"Partition {i}: {num_triplets} triplets")

import pandas as pd 
import networkx as nx 
from networkx.algorithms import community
import community as community_louvain
import matplotlib.pyplot as plt 
import igraph as ig
%matplotlib inline
for x in lst_b:
  print(x)
  colors = ["#00C98D", "#5030C0", "#50F0F0"]
pos = nx.spring_layout(G)
lst_b = community.girvan_newman(G)
color_map_b = {}
keys = G.nodes()
values = "black"
for i in keys:
        color_map_b[i] = values
counter = 0
for x in lst_b:
  print(1)
  for c in x:
    for n in c:
      #print(n,counter)
      color_map_b[n] = colors[counter]
    counter = counter + 1
  break
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_nodes(G, pos, node_color=dict(color_map_b).values())
nx.draw_networkx_labels(G, pos)
plt.axis("off")
plt.show()
modularity = []
for x in lst_b:
  modularity.append(community.modularity(G, x))
modularity
plt.plot(modularity, 'o')
plt.xlabel('# of clusters')
plt.ylabel('modularity')
plt.show()
