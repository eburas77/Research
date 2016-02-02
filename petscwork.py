from petsc4py import PETSc as Pet
import networkx as nx
import numpy as np
import scipy
import timeit

#G = nx.read_gml('power.gml')
G=nx.Graph()
G.add_nodes_from([1,2,3,4,5,6,7,8,9])
G.add_edges_from([(1,2),(1,3),(1,5),(1,9),
                    (2,3),(2,6),(2,8),(3,4),(4,5),
                    (4,7),(4,9),(5,8),(6,7),(6,9)])
print "read in power grid graph"
L = nx.laplacian_matrix(G)
L = L.todense()

print "created laplacian matrix"
k = 3 #A higher number means a looser connectivity requirement.
l = 3 #A higher number means a stricter connectivity requirement.

#this is fan chungs greedy algorithm
start_time = timeit.default_timer()
P = nx.kl_connected_subgraph(G, k, l, low_memory=False, same_as_graph=False)
elapsed = timeit.default_timer() - start_time
print "Fan Chung's algorithm ran in %f seconds" %elapsed
print "split graph"
P_L = nx.laplacian_matrix(P)
P_L = P_L.todense()

T = L-P_L
P_L = P_L+np.eye(len(L))*np.diagonal(T)
T = T-np.eye(len(L))*np.diagonal(T)

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
b.set(1)
x.set(0)
ksp = Pet.KSP()
ksp.create(Pet.COMM_WORLD)
ksp.setType('cg')
pc = ksp.getPC()
pc.setType(pc.Type.GAMG)

ksp.setOperators(P_L_petsc)
ksp.solve(b,x)   

y,f = T_petsc.getVecs()
f.set(1)
y.set(0)
#set to solve LU instead of GAMG
ksp.setOperators(T_petsc)
ksp.solve(f,y)
