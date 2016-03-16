# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from petsc4py import PETSc as Pet
import networkx as nx
import numpy as np
import scipy
import timeit
#import matplotlib.pylab as plt
import kl_connected_subgraph as kl

G = nx.read_edgelist('newmetabolic.txt',nodetype=int)



start_time = timeit.default_timer() 
P = G.copy()               
counter = 0
for edge in P.edges():
    #counter+=1
    #print counter
    
    (u,v) = edge    #get edge
    paths = [[u,v]]    
    cnt = 1         #accounts for direct path
    uneighbors = list(nx.all_neighbors(P,u))     #get list of neighbors of u    
    uneighbors.remove(v)              # remove v because we already accounted for it

    for neighbor1 in uneighbors:           # loop through neighbors of u
        u1neighbors = list(nx.all_neighbors(P,neighbor1))     #find list of neighbors of each neighbor of u
        #print u1neighbors
        #print ""
        if v in u1neighbors:
            #cnt += 1     #if v in this list then there is a path of length 2 from u to v
            temppath = [u,neighbor1,v]
            found = False
            for j in range(1,len(temppath)):
                    for path in paths:
                        for i in range(1,len(path)):
                            if path[i-1] == temppath[j-1] and path[i] == temppath[j]:
                                found = True
                        
            if found == False:
                cnt +=1
                paths.append([u,neighbor1,v])
            #print "path length 2 with ", neighbor1
            
        if cnt >=3:
            break
        
        #u1neighbors = [x for x in u1neighbors if x not in uneighbors]     #remove all items from this second neighbor list that were in the first neighbor list
        if u in u1neighbors:        
            u1neighbors.remove(u)        
        for neighbor2 in u1neighbors:
            u2neighbors = list(nx.all_neighbors(P,neighbor2))      #these are third neighbors
        
            if v in u2neighbors:
                #cnt += 1    #add 1 to count if v is in this set
                temppath = [u,neighbor1,neighbor2,v]
                found = False    
                for j in range(1,len(temppath)):
                    for path in paths:
                        for i in range(1,len(path)):
                            if path[i-1] == temppath[j-1] and path[i] == temppath[j]:
                                found = True
                            
                            
                if found == False:
                    cnt +=1
                    paths.append([u,neighbor1,neighbor2,v])
                            
                            
            
            if cnt >=3:
                break
    
    if cnt <=2:     #cnt must be 3 or greater to remain in the graph
        P.remove_edge(u,v)
        #print "removed edge: ", (u,v)
elapsed = timeit.default_timer() - time
print "Protein partition ran in %f seconds" %elapsed      

P1 = kl.kl_connected_subgraph(G, 3, 3, low_memory=True, same_as_graph=False)


H = nx.Graph()
for node in G.nodes():
    H.add_node(node)
for edge in P1.edges():
    H.add_edge(edge[0],edge[1])

P1 = H 

A_1 = nx.adjacency_matrix(P1)
A_1 = A_1.todense()

A = nx.adjacency_matrix(P)
A = A.todense()

print np.nonzero(A_1-A) 
        
        
        
        