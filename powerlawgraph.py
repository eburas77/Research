#create power law graph

from networkx import *
import sys
import numpy as np

n=10 # Number of nodes
m=5 # Number of edges to attach from a new node to existing nodes

G = barabasi_albert_graph(n,m,.5)

# some properties
print("node degree clustering")
for v in nodes(G):
    print('%s %d %f' % (v,degree(G,v),clustering(G,v)))

# print the adjacency list to terminal
try:
    write_adjlist(G, sys.stdout)
except TypeError: # Python 3.x
    write_adjlist(G,sys.stdout.buffer)

A = adjacency_matrix(G)

#print("adjacency matrix of whole graph")
#print(A)

print("")

k = 3
l=3
H = kl_connected_subgraph(G, k, l, low_memory=False, same_as_graph=False)

A_H = adjacency_matrix(H)

try:
    write_adjlist(H, sys.stdout)
except TypeError: # Python 3.x
    write_adjlist(H,sys.stdout.buffer)

L = laplacian_matrix(G)

#print(L)

eigs_L = laplacian_spectrum(G)
