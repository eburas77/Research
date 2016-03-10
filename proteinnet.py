# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 19:42:39 2016

@author: ericburas
"""

from petsc4py import PETSc as Pet
import networkx as nx
import numpy as np
import scipy
import timeit
import matplotlib.pylab as plt
import kl_connected_subgraph as kl

#F = nx.read_gml('celegansneural.gml')
#F = nx.read_gml('as-22july06.gml')
#G = nx.Graph()
#for i in range(0,nx.number_of_nodes(F)):
#    print i
#    for j in range(0,nx.number_of_nodes(F)):
#        if F.has_edge(i,j):
#            if not G.has_edge(i,j):
#                G.add_edge(i,j)
                
fh=open('newpheno.txt', 'rb')
G=nx.read_edgelist(fh)

A = nx.adjacency_matrix(G)
A = A.todense()



print "read in graph"
L = nx.laplacian_matrix(G)
L = L.todense()


print "created laplacian matrix"
k = 3 #A higher number means a looser connectivity requirement.
l = 3 #A higher number means a stricter connectivity requirement.

#this is fan chungs greedy algorithm
start_time = timeit.default_timer()
P = kl.kl_connected_subgraph(G, k, l, low_memory=True, same_as_graph=False)
elapsed = timeit.default_timer() - start_time
print "Fan Chung's algorithm ran in %f seconds" %elapsed
print "split graph"
P_A = nx.adjacency_matrix(P)
P_A = P_A.todense()
T_A = A - P_A
nx.write_edgelist(P, "proteinlocal.edgelist")
T_graph = nx.from_numpy_matrix(T_A)
nx.write_edgelist(T_graph, "proteinglobal.edgelist")
P_L = nx.laplacian_matrix(P)
P_L = P_L.todense()

T = L-P_L
P_L = P_L+np.eye(len(L))*np.diagonal(T)
T = T-np.eye(len(L))*np.diagonal(T)
print T.shape
print "rank of teleportation matrix: %i" %np.linalg.matrix_rank(T)

print "number of edges in entire graph: %i" %nx.number_of_edges(G)
print "number of edges in k,l connected subgraph: %i" %nx.number_of_edges(P)