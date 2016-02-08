import networkx as nx
import numpy as np
from random import choice
import timeit
import kl_connected_subgraph as kl

#nodes = 100
#k=5
#G = nx.barabasi_albert_graph(nodes,k)
#G = nx.read_gml('as-22july06.gml')
G = nx.read_weighted_edgelist('celegans_metabolic.net')

L = nx.laplacian_matrix(G)
L.todense()





#for i in range(0,500):
#    node_1 = choice(G.nodes())
#    print node_1
#    node_2 = choice(G.nodes())
#    print node_2
#    if not G.has_edge(node_1,node_2):
#        G.add_edge(node_1,node_2)
#    print i
    
#print "added nodes"

#print ""
#print L_P
k=3
l=3
start_time = timeit.default_timer()
P = kl.kl_connected_subgraph(G, k, l, low_memory=True, same_as_graph=False)
elapsed = timeit.default_timer() - start_time
print "Fan Chung's algorithm ran in %f seconds" %elapsed
print "split graph"
L = nx.laplacian_matrix(G)
L.todense()
P_L = nx.laplacian_matrix(P)
P_L = P_L.todense()

T = L-P_L
P_L = P_L+np.eye(L.shape[0])*np.diagonal(T)
T = T-np.eye(L.shape[0])*np.diagonal(T)
print T.shape
print "rank of teleportation matrix: %i" %np.linalg.matrix_rank(T)

print "number of edges in entire graph: %i" %nx.number_of_edges(G)
print "number of edges in k,l connected subgraph: %i" %nx.number_of_edges(P)





#print np.linalg.norm(L-L_P)
#pos = nx.spring_layout(P)
#nx.draw_networkx(G, pos)