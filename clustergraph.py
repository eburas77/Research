from networkx import *
import sys
import numpy as np
import scipy
G=Graph()

G.add_nodes_from([1,2,3,4,5,6,7,8,9])

G.add_edges_from([(1,2),(1,3),(1,5),(1,9),(2,3),(2,6),(2,8),(3,4),(4,5),(4,7),(4,9),(5,8),(6,7),(6,9)])
#n=10 # Number of nodes
#m=5 # Number of edges to attach from a new node to existing nodes

#G = barabasi_albert_graph(n,m,.5)

for v in nodes(G):
    print('%s %d %f' % (v,degree(G,v),clustering(G,v)))


L = laplacian_matrix(G)
print("this is the laplacian matrix of G")
print(L)

#eigs_L = laplacian_spectrum(G)

#print(eigs_L)

print("")

k = 3
l=3
#this is fan chungs greedy algorithm
H = kl_connected_subgraph(G, k, l, low_memory=False, same_as_graph=False)

H_L = laplacian_matrix(H)
print("this is the laplacian matrix of the maximum k,l connected subgraph")
print(H_L)

T = L-H_L
print("")
print("this is the teleportation matrix")
print(T)
H_L_inv = scipy.sparse.linalg.inv(H_L)

T_hat = T.todense()
U,s,V = scipy.linalg.svd(T_hat)

solve_L = H_L_inv + H_L_inv*LU.L.A*scipy.sparse.linalg.inv((scipy.sparse.identity(n)-(LU.U.A)*(H_L_inv)*(LU.L.A)))*LU.U.A*H_L_inv

