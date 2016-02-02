from networkx import *
import sys
import numpy as np
from scipy import *
import matplotlib.pyplot as plt


#using Barabasi Albert Model or to create simple graph
G=Graph()
G.add_nodes_from([1:24])
G.add_edges_from([(1,2),(2,3),(3,4),(1,5),(2,6),(3,7),(4,8),(5,6),(6,7),(7,8),
                    (5,9),(6,10),(7,11),(8,12),(9,10),(10,11),(11,12),
                    (9,13),(10,14),(11,15),(12,16),(13,14),(14,15),(15,16)
                    (1,21),(4,20),(4,24),(14,23),(15,22),(16,17),(3,19),(11,18)])
#n=200# Number of nodes
#m=5 # Number of edges to attach from a new node to existing nodes
#G = barabasi_albert_graph(n,m,.5)

#to create power law degree graph
#nodes = 1000
#beta = 2.5 #power law exponent
#z=[]
#while len(z)<nodes:
#    nextval = int(nx.utils.powerlaw_sequence(1, beta)[0])
#    if nextval!=0:
#        z.append(nextval)
#G = nx.configuration_model(z)
#G=nx.Graph(G) # remove parallel edges
#G.remove_edges_from(G.selfloop_edges())
#this sometimes results in an error because the sum of node degrees is odd
#run again until it is even


#for v in nodes(G):
#    print('%s %d %f' % (v,degree(G,v),clustering(G,v)))


L = laplacian_matrix(G)
print "this is the laplacian matrix of G"
L= L.todense()
print L

# L = L + I to avoid singularity
L = L+eye(len(L))

#eigs_L = laplacian_spectrum(G)
#print(eigs_L)

print("")

k = 3 #A higher number means a looser connectivity requirement.
l = 3 #A higher number means a stricter connectivity requirement.

#this is fan chungs greedy algorithm
P = kl_connected_subgraph(G, k, l, low_memory=False, same_as_graph=False)

P_L = laplacian_matrix(P)
P_L = P_L.todense()

#small changes to keep the degree sequence in P
T = L-P_L
P_L = P_L+eye(len(L))*diagonal(T)
T = T-eye(len(L))*diagonal(T)
print "this is the laplacian matrix of the maximum k,l connected subgraph" 
print P_L
print ""
print "this is the teleportation matrix"
print T
print ""

#we would do multigrid here
P_L_inv = np.linalg.inv(P_L)
P_L_inv = np.array(P_L_inv)

#SVD
U,s,V = np.linalg.svd(T)
size = sum(s>.00000001)

#remove rows and columns for low rank matrix
U = np.array(U[:,0:size-1])
s = s[0:size-1]
s = np.diag(s)
#s = np.reshape(len(s),1)
V = np.array(V[0:size-1,:])

#SHERMAN WOODBERRY MORRISON
a = np.dot(P_L_inv,U)
b = np.dot(V,P_L_inv)
c = np.dot(V,P_L_inv)
d = np.dot(c,U)
solve_L = P_L_inv + np.dot(np.dot(a,(np.linalg.inv(np.linalg.inv(s)-d))),b)


L_inv = np.linalg.inv(L)

dif = solve_L - L_inv
print "test to see if algorithm solved the inverse of the laplacian"
norm_dif = np.linalg.norm(dif)
print "norm of the difference between L_inv and solve_L = ", norm_dif

#pos is a dictionary of node point locations in the x,y plane
#can change this to show that the graph is planar
#pos = nx.spring_layout(G)
#nx.draw_networkx(G, pos)
#plt.savefig('entiregraph.pdf')
#pos2 = nx.spring_layout(H)
#nx.draw_networkx(H, pos2)
#plt.savefig('planargraph.pdf')