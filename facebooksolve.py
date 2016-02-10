from petsc4py import PETSc as Pet
import networkx as nx
import numpy as np
import scipy
import timeit
import matplotlib.pylab as plt
import kl_connected_subgraph as kl

fh=open('facebook_combined.txt', 'rb')
G=nx.read_edgelist(fh,nodetype=int)
                
A = nx.adjacency_matrix(G)
A = A.todense()



print "read in graph"
L = nx.laplacian_matrix(G)
L = L.todense()

P = nx.read_edgelist('facebooklocal.edgelist',nodetype=int)

H = nx.Graph()
for node in G.nodes():
    H.add_node(node)
for edge in P.edges():
    H.add_edge(edge[0],edge[1])

P = H

P_L = nx.laplacian_matrix(P)
P_L = P_L.todense()

T = L-P_L
P_L = P_L+np.eye(len(L))*np.diagonal(T)
T = T-np.eye(len(L))*np.diagonal(T)
print T.shape
print "rank of teleportation matrix: %i" %np.linalg.matrix_rank(T)

print "number of edges in entire graph: %i" %nx.number_of_edges(G)
print "number of edges in k,l connected subgraph: %i" %nx.number_of_edges(P)


#plt.spy(A,precision=0.01, markersize=1)
#plt.savefig('celeganspy.png')
#print P_L
#print ""
#print T
P_L_csr = scipy.sparse.csr_matrix(P_L)
T_csr = scipy.sparse.csr_matrix(T)


P_L_petsc = Pet.Mat().createAIJ(size=P_L_csr.shape,
                            csr = (P_L_csr.indptr, P_L_csr.indices, P_L_csr.data))
                                
T_petsc = Pet.Mat().createAIJ(size=T_csr.shape,
                            csr = (T_csr.indptr, T_csr.indices, T_csr.data))
 
x,b = P_L_petsc.getVecs()
r = x.duplicate()
b.set(1)
x.set(0)
ksp = Pet.KSP()
ksp.create(Pet.COMM_WORLD)
ksp.setFromOptions()
pc = ksp.getPC()
pc.setType(pc.Type.GAMG)
ksp.setOperators(P_L_petsc)
print "now solve"

ksp.solve(b,x)   

P_L_petsc.mult(x, r)
r.axpy(-1.0, b)
r.view()
#y,f = T_petsc.getVecs()
#f.set(1)
#y.set(0)
#set to solve LU instead of GAMG
#ksp.setOperators(T_petsc)
#ksp.solve(f,y)