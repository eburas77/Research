from petsc4py import PETSc as Pet
import networkx as nx
import numpy as np
import scipy
import timeit
#import matplotlib.pylab as plt
import kl_connected_subgraph as kl
start_time = timeit.default_timer()
F = nx.read_gml('celegansneural.gml')

G = nx.Graph()
for i in range(0,nx.number_of_nodes(F)):
    #print i
    for j in range(0,nx.number_of_nodes(F)):
        if F.has_edge(i,j):
            if not G.has_edge(i,j):
                G.add_edge(i,j)
                
A = nx.adjacency_matrix(G)
A = A.todense()



print "read in graph"
L = nx.laplacian_matrix(G)
L = L.todense()
L = L +np.eye(len(L))


P = nx.read_edgelist('neurallocal.edgelist',nodetype=int)

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
#L = np.array([[11,13,15],[17,19,21],[23,25,27]])
#T = np.array([[1,2,3],[4,5,6],[7,8,9]])
#P_L = np.array([[10,11,12],[13,14,15],[16,17,18]])

#plt.spy(A,precision=0.01, markersize=1)
#plt.savefig('celeganspy.png')
#print P_L
#print ""
#print T

U,s,V = np.linalg.svd(T)
size = sum(s>.00000001)


#remove rows and columns for low rank matrix
U = np.array(U[:,0:size])
s = s[0:size]
s = np.diag(s)
s_inv = np.linalg.inv(s)
V = np.array(V[0:size,:])     #need to reshape V to keep low rank
sizeU1,sizeU2 = U.shape
P_L_csr = scipy.sparse.csr_matrix(P_L)



P_L_petsc = Pet.Mat().createAIJ(size=P_L_csr.shape,
                            csr = (P_L_csr.indptr, P_L_csr.indices, P_L_csr.data))
                                
                      
U_petsc = Pet.Mat().createDense(size=U.shape,array =U)
s_inv_petsc = Pet.Mat().createDense(size = s_inv.shape,array = s_inv)
V_petsc = Pet.Mat().createDense(size = V.shape, array =V) 
                
m,n = P_L_petsc.getSize()
#print "P is: ", (m,n)

b = Pet.Vec().createSeq(m)
b.setRandom()     #set b
#b.view()
y = b.duplicate()
y_1 = Pet.Vec().createSeq(size)
y_2 = y_1.duplicate()
y_3 = y.duplicate()
y_4 = y.duplicate()
x = y.duplicate()
Qvec = b.duplicate()



ksp = Pet.KSP() #linear solver
ksp.create(Pet.COMM_WORLD)
ksp.setFromOptions()
pc = ksp.getPC()
pc.setType(pc.Type.GAMG) #multigrid preconditioner
#pc.setType(pc.Type.LU)
ksp.setOperators(P_L_petsc)
print "now solve"

ksp.solve(b,y)         #y = P^{-1}b
              

V_petsc.mult(y,y_1)            #y_1 = V*y


Q = Pet.Mat().createDense(size = U.shape) 
Q_1 = Pet.Mat().createDense(size = s_inv.shape)
Q_2 = Pet.Mat().createDense(size = s_inv.shape)  #initialize Q matrices

Q.setUp()
Q_1.setUp()
Q_2.setUp()
rows=range(sizeU1)
for i in range(sizeU2):
    col = i
    ksp.solve(U_petsc.getColumnVector(i),Qvec)
    Q.setValues(rows, col, Qvec.getArray())

Q.assemblyBegin()
Q.assemblyEnd() 

U_petsc_2 = Pet.Mat().createDense(size=U.shape,array =U)
V_petsc.matMult(Q,Q_1)              #Q_1 = V*Q
Q_2 = s_inv_petsc+Q_1    

ksp2 = Pet.KSP()                #second linear solver
ksp2.create(Pet.COMM_WORLD)
pc2 = ksp2.getPC()
pc2.setType(pc2.Type.LU)
ksp2.setOperators(Q_2)          #do i need a preconditioner? 
ksp2.solve(y_1,y_2)             #y_2 = Q_2^{-1}*y_1

U_petsc_2.mult(y_2,y_3)           #y_3 = U*y_2

P_L_petsc_2 = Pet.Mat().createAIJ(size=P_L_csr.shape,
                            csr = (P_L_csr.indptr, P_L_csr.indices, P_L_csr.data))
ksp3 = Pet.KSP()                #second linear solver
ksp3.create(Pet.COMM_WORLD)
pc3 = ksp3.getPC()
pc3.setType(pc3.Type.GAMG)                           
ksp3.setOperators(P_L_petsc_2)
ksp3.solve(y_3,y_4)              #y_4 = P^{-1}*y_3
x = y-y_4
x1 = x.getArray()
elapsed = timeit.default_timer() - start_time
print "Neural Solve ran in %f seconds" %elapsed

print "now test vs numpy solve"

b1 = b.getArray()

x2 = np.linalg.solve(L,b1)


print "norm of difference between nplinalg and my way: ", np.linalg.norm(x1-x2)
L_csr = scipy.sparse.csr_matrix(L)



L_petsc = Pet.Mat().createAIJ(size=L_csr.shape,
                            csr = (L_csr.indptr, L_csr.indices, L_csr.data))

ksp4 = Pet.KSP()
ksp4.create(Pet.COMM_WORLD)
pc4 = ksp.getPC()
pc4.setType(pc4.Type.LU)
ksp4.setOperators(L_petsc)
b2 = b.duplicate()
ksp4.solve(b,b2)
barray = b2.getArray()


print "norm of difference between petsc straight solve and my way: ", np.linalg.norm(x1-barray)
print "norm of difference between petsc straight solve and np.linalg: ", np.linalg.norm(x2-barray)
