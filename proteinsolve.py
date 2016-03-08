from petsc4py import PETSc as Pet
import networkx as nx
import numpy as np
import scipy
import timeit
#import matplotlib.pylab as plt
import kl_connected_subgraph as kl

fh=open('phenotype.txt', 'rb')
G=nx.read_edgelist(fh,data=(('phenotype',str),))
                
A = nx.adjacency_matrix(G)
A = A.todense()



print "read in graph"
L = nx.laplacian_matrix(G)
L = L.todense()

P = nx.read_edgelist('proteinlocal.edgelist')

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

U,s,V = np.linalg.svd(T)
size = sum(s>.00000001)
m,n = U.shape

#remove rows and columns for low rank matrix
#U = np.array(U[:,0:size-1])
#s = s[0:size-1]
s = np.diag(s)
s_inv = np.linalg.inv(s)
#s = np.reshape(len(s),1)
#V = np.array(V[0:size-1,:])     need to reshape V to keep low rank

P_L_csr = scipy.sparse.csr_matrix(P_L)



P_L_petsc = Pet.Mat().createAIJ(size=P_L_csr.shape,
                            csr = (P_L_csr.indptr, P_L_csr.indices, P_L_csr.data))


U_petsc = Pet.Mat().createDense(size=U.shape,array =U)
s_inv_petsc = Pet.Mat().createDense(size = s_inv.shape,array = s_inv)
V_petsc = Pet.Mat().createDense(size = V.shape, array =V)                        

 
y,b = P_L_petsc.getVecs() #initialize vectors
x = y.duplicate
b.set(1)
y.set(0)
y_1 = y.duplicate()
y_2 = y.duplicate()
y_3 = y.duplicate()
y_4 = y.duplicate()
Qvec = y.duplicate()


ksp = Pet.KSP() #linear solver
ksp.create(Pet.COMM_WORLD)
ksp.setFromOptions()
pc = ksp.getPC()
pc.setType(pc.Type.GAMG) #multigrid preconditioner
ksp.setOperators(P_L_petsc)
print "now solve"

ksp.solve(b,y)                 #y = P^{-1}b

V_petsc.mult(y,y_1)            #y_1 = V*y


Q = Pet.Mat().createDense(size = P_L_csr.shape) 
Q_1 = Pet.Mat().createDense(size = P_L_csr.shape)
Q_2 = Pet.Mat().createDense(size = P_L_csr.shape)  #initialize Q dense matrices
#Q_1 = Q.duplicate()
#Q_2 = Q.duplicate()
Q.setUp()
Q_1.setUp()
Q_2.setUp()
for i in range(0,n):             #Q = P^{-1}*U
    ksp.solve(U_petsc.getColumnVector(i),Qvec)
    Q.getColumnVector(i,Qvec)


Q_1 = V_petsc.matMult(Q)             #Q_1 = V*Q
Q_2 = s_inv_petsc+Q_1    

ksp2 = Pet.KSP()                #second linear solver
ksp2.create(Pet.COMM_WORLD)
ksp2.setOperators(Q_2)          #do i need a preconditioner? 
ksp2.solve(y_1,y_2)             #y_2 = Q_2^{-1}*y_1

U_petsc.mult(y_2,y_3)           #y_3 = U*y_2
ksp.solve(y_3,y_4)              #y_4 = P^{-1}*y_3
x = y-y_4

x.view()


#P_L_petsc.mult(y, r)
#r.axpy(-1.0, b)
#r.view()
#y,f = T_petsc.getVecs()
#f.set(1)
#y.set(0)
#set to solve LU instead of GAMG
#ksp.setOperators(T_petsc)
#ksp.solve(f,y)