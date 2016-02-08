from petsc4py import PETSc as Pet
import networkx as nx
import numpy as np
import scipy
import timeit
import matplotlib.pylab as plt

G = nx.read_gml('celegansneural.gml')
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

print nx.number_of_edges(G)
G = G.to_undirected()
A = nx.adjacency_matrix(G)
print "read in graph"
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
print T.shape
print "rank of teleportation matrix: %i" %np.linalg.matrix_rank(T)

print "number of edges in entire graph: %i" %nx.number_of_edges(G)
print "number of edges in k,l connected subgraph: %i" %nx.number_of_edges(P)


plt.spy(A,precision=0.01, markersize=1)
plt.savefig('celeganspy.png')
#print P_L
#print ""
#print T
P_L_csr = scipy.sparse.csr_matrix(P_L)
T_csr = scipy.sparse.csr_matrix(T)


P_L_petsc = Pet.Mat().createAIJ(size=P_L_csr.shape,
                                csr = (P_L_csr.indptr, P_L_csr.indices, P_L_csr.data))
                                
T_petsc = Pet.Mat().createAIJ(size=T_csr.shape,
                                csr = (T_csr.indptr, T_csr.indices, T_csr.data))
 
#x,b = P_L_petsc.getVecs()
#b.set(1)
#x.set(0)
#ksp = Pet.KSP()
#ksp.create(Pet.COMM_WORLD)
#ksp.setFromOptions()


#ksp.setOperators(P_L_petsc)
#ksp.solve(b,x)   

#y,f = T_petsc.getVecs()
#f.set(1)
#y.set(0)
#set to solve LU instead of GAMG
#ksp.setOperators(T_petsc)
#ksp.solve(f,y)
