# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 16:55:25 2016

@author: ericburas
"""

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
#s = np.reshape(len(s),1)
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
b.set(1)     #set b
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
ksp.setOperators(P_L_petsc)
print "now solve"

ksp.solve(b,y)         #y = P^{-1}b
ytest = y.getArray()
y_np = np.linalg.solve(P_L,np.transpose(np.ones(m)))
print np.linalg.norm(ytest-y_np)
               
#m,n = P_L_petsc.getSize()
#print "P is: ", (m,n)
#sizeb = b.getSize()
#print "b is: ", sizeb
#sizey = y.getSize()
#print "y is: ", size
#sizeV = V_petsc.getSize()
#print "V is: ",sizeV
#sizey_1 = y_1.getSize()
#print "y_1 is: ",sizey_1

V_petsc.mult(y,y_1)            #y_1 = V*y
y1test = y_1.getArray()
y_1np = np.dot(V,y_np)
print np.linalg.norm(y1test-y_1np)


Q = Pet.Mat().createDense(size = U.shape) 
Q_1 = Pet.Mat().createDense(size = s_inv.shape)
Q_2 = Pet.Mat().createDense(size = s_inv.shape)  #initialize Q matrices

Q.setUp()
Q_1.setUp()
Q_2.setUp()
rows=range(sizeU2)
Q_np = U
print "now do multiple rhs"
for i in range(sizeU2):
    col = i
    ksp.solve(U_petsc.getColumnVector(i),Qvec)
    Qvec_np = np.linalg.solve(P_L,U[:,i])
    Q_np[:,i] = Qvec_np
    for row in rows: 
        Q.setValues(row, col, Qvec.getValue(row))
    print np.linalg.norm(Q_np[:,i]-Qvec)
Q.assemblyBegin()
Q.assemblyEnd() 



V_petsc.matMult(Q,Q_1)              #Q_1 = V*Q
Q_1np = np.dot(V,Q_np)
Q_2 = s_inv_petsc+Q_1
Q_2np = s_inv+Q_1np    

ksp2 = Pet.KSP()                #second linear solver
ksp2.create(Pet.COMM_WORLD)
ksp2.setOperators(Q_2)          #do i need a preconditioner? 
ksp2.solve(y_1,y_2)             #y_2 = Q_2^{-1}*y_1
y_2np = np.linalg.solve(Q_2np,y_1np)
ytest2 = y_2.getArray()
print np.linalg.norm(ytest2-y_2np)

U_petsc.mult(y_2,y_3)           #y_3 = U*y_2
y_3np = np.dot(U,y_2np)
ytest3 = y_3.getArray()
print np.linalg.norm(ytest3-y_3np)

ksp.solve(y_3,y_4)              #y_4 = P^{-1}*y_3
y_4np = np.linalg.solve(P_L,y_3np)
ytest4 = y_4.getArray()
print np.linalg.norm(ytest4-y_4np)

x = y-y_4
x1 = x.getArray()
x_np = y_np-y_4np
print np.linalg.norm(x1-x_np)


elapsed = timeit.default_timer() - start_time
print "Neural Solve ran in %f seconds" %elapsed

print "now test vs numpy solve"

b1 = np.transpose(np.ones(m))

x2 = np.linalg.solve(L,b1)


print "norm of difference between nplinalg and my way: ", np.linalg.norm(x1-x2)
L_csr = scipy.sparse.csr_matrix(L)



L_petsc = Pet.Mat().createAIJ(size=L_csr.shape,
                            csr = (L_csr.indptr, L_csr.indices, L_csr.data))

ksp3 = Pet.KSP()
ksp3.create(Pet.COMM_WORLD)
ksp3.setOperators(L_petsc)
b2 = b.duplicate()
ksp3.solve(b,b2)
barray = b2.getArray()

print "norm of difference between petsc straight solve and my way: ", np.linalg.norm(x1-barray)

#P_L_petsc.mult(y, r)
#r.axpy(-1.0, b)
#r.view()
#y,f = T_petsc.getVecs()
#f.set(1)
#y.set(0)
#set to solve LU instead of GAMG
#ksp.setOperators(T_petsc)
#ksp.solve(f,y)