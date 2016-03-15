from petsc4py import PETSc as Pet
import networkx as nx
import numpy as np
import scipy
import timeit
#import matplotlib.pylab as plt
import kl_connected_subgraph as kl
G = nx.read_edgelist('newmetabolic.txt',nodetype=int)
                
A = nx.adjacency_matrix(G)
A = A.todense()

time = timeit.default_timer()
P = G.copy()               
counter = 0
for edge in P.edges():
    #counter+=1
    #print counter
    (u,v) = edge    #get edge
    cnt = 1         #accounts for direct path
    uneighbors = list(nx.all_neighbors(P,u))     #get list of neighbors of u    
    uneighbors.remove(v)              # remove v because we already accounted for it

    for neighbor1 in uneighbors:           # loop through neighbors of u
        u1neighbors = list(nx.all_neighbors(P,neighbor1))     #find list of neighbors of each neighbor of u
        #print u1neighbors
        #print ""
        if v in u1neighbors:
            cnt += 1     #if v in this list then there is a path of length 2 from u to v
            #print "path length 2 with ", neighbor1
            
        if cnt >=3:
            break
        u1neighbors = [x for x in u1neighbors if x not in uneighbors]     #remove all items from this second neighbor list that were in the first neighbor list
        if u in u1neighbors:        
            u1neighbors.remove(u)        
        for neighbor2 in u1neighbors:
            u2neighbors = list(nx.all_neighbors(P,neighbor2))      #these are third neighbors
            if v in u2neighbors:
                cnt += 1    #add 1 to count if v is in this set
                #print "path length 3 with ", (neighbor1,neighbor2)
            if cnt >=3:
                break
    
    if cnt <=2:     #cnt must be 3 or greater to remain in the graph
        P.remove_edge(u,v)
        #print "removed edge: ", (u,v)
elapsed = timeit.default_timer()-time
print "Partition ran in %f seconds" %elapsed

L = nx.laplacian_matrix(G)
L = L.todense()
L = L+np.eye(len(L))

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
print "now solve"
time2 = timeit.default_timer()
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
elapsed = timeit.default_timer() - time2
print "Metabolic Solve ran in %f seconds" %elapsed

print "now test vs numpy solve"

b1 = b.getArray()

x2 = np.linalg.solve(L,b1)


print "norm of difference between nplinalg and my way: ", np.linalg.norm(x1-x2)
L_csr = scipy.sparse.csr_matrix(L)



L_petsc = Pet.Mat().createAIJ(size=L_csr.shape,
                            csr = (L_csr.indptr, L_csr.indices, L_csr.data))

ksp4 = Pet.KSP()
ksp4.create(Pet.COMM_WORLD)
pc4 = ksp4.getPC()
pc4.setType(pc4.Type.LU)
ksp4.setOperators(L_petsc)
b2 = b.duplicate()
ksp4.solve(b,b2)
barray = b2.getArray()


print "norm of difference between petsc straight solve and my way: ", np.linalg.norm(x1-barray)
print "norm of difference between petsc straight solve and np.linalg: ", np.linalg.norm(x2-barray)
