from networkx import *
import sys
import numpy as np
from scipy import *
import matplotlib.pyplot as plt

#G=Graph()

#G.add_nodes_from([1,2,3,4,5,6,7,8,9])

#G.add_edges_from([(1,2),(1,3),(1,5),(1,9),(2,3),(2,6),(2,8),(3,4),(4,5),(4,7),(4,9),(5,8),(6,7),(6,9)])
#n=200# Number of nodes
#m=5 # Number of edges to attach from a new node to existing nodes

#G = barabasi_albert_graph(n,m,.5)
nodes = 1000
beta = 2.5 #power law exponent
z=[]
while len(z)<nodes:
    nextval = int(nx.utils.powerlaw_sequence(1, beta)[0])
    if nextval!=0:
        z.append(nextval)
G = nx.configuration_model(z)
G=nx.Graph(G) # remove parallel edges
G.remove_edges_from(G.selfloop_edges())

#for v in nodes(G):
#    print('%s %d %f' % (v,degree(G,v),clustering(G,v)))


L = laplacian_matrix(G)
print "this is the laplacian matrix of G"
L= L.todense()
print L

L = L+eye(len(L))

#eigs_L = laplacian_spectrum(G)
#print(eigs_L)

print("")

k = 3 #A higher number means a looser connectivity requirement.
l = 3 #A higher number means a stricter connectivity requirement.

#this is fan chungs greedy algorithm
H = kl_connected_subgraph(G, k, l, low_memory=False, same_as_graph=False)

H_L = laplacian_matrix(H)
H_L = H_L.todense()


T = L-H_L
H_L = H_L+eye(len(L))*diagonal(T)
T = T-eye(len(L))*diagonal(T)
print "this is the laplacian matrix of the maximum k,l connected subgraph" 
print H_L
print ""
print "this is the teleportation matrix"
print T
print ""


H_L_inv = np.linalg.inv(H_L)
H_L_inv = np.array(H_L_inv)


U,s,V = np.linalg.svd(T)
size = sum(s>.00000001)

U = np.array(U[:,0:size-1])
s = s[0:size-1]
s = np.diag(s)
#s = np.reshape(len(s),1)
V = np.array(V[0:size-1,:])

a = np.dot(H_L_inv,U)
b = np.dot(V,H_L_inv)
c = np.dot(V,H_L_inv)
d = np.dot(c,U)



solve_L = H_L_inv + np.dot(np.dot(a,(np.linalg.inv(np.linalg.inv(s)-d))),b)

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