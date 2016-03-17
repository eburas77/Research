from networkx import *
import sys
import numpy as np
from scipy import *
import matplotlib.pyplot as plt


#using Barabasi Albert Model or to create simple graph
G=Graph()
G.add_nodes_from(range(1,25))
G.add_edges_from([(1,2),(2,3),(3,4),(1,5),(2,6),(3,7),(4,8),(5,6),(6,7),(7,8),
                    (5,9),(6,10),(7,11),(8,12),(9,10),(10,11),(11,12),
                    (9,13),(10,14),(11,15),(12,16),(13,14),(14,15),(15,16), 					
                    (1,21),(4,20),(4,24),(14,23),(15,22),(16,17),(3,19),(11,18),
                    (1,13),(2,14),(3,15),(4,16),(1,4),(5,8),(9,12),(13,16),(11,19),(6,24)])
                    
k = 3
l = 3				
P = G.copy()
               
deleted_some = True
while deleted_some == True:
    print "loop for deletions"
    deleted_some = False
    for edge in P.edges():
        
        (u,v) = edge    #get edge
        paths = [[u,v]]    
        cnt = 1         #accounts for direct path
        uneighbors = list(nx.all_neighbors(P,u))     #get list of neighbors of u    
        uneighbors.remove(v)              # remove v because we already accounted for it
    
        for neighbor1 in uneighbors:           # loop through neighbors of u
            u1neighbors = list(nx.all_neighbors(P,neighbor1))     #find list of neighbors of each neighbor of u
            
            
            
            
            if u in u1neighbors:        
                u1neighbors.remove(u)        
            for neighbor2 in u1neighbors:
                u2neighbors = list(nx.all_neighbors(P,neighbor2))      #these are third neighbors
            
                if v in u2neighbors:
                    
                    temppath = [u,neighbor1,neighbor2,v]
                    found = False    
                    for j in range(1,len(temppath)):
                        for path in paths:
                            for i in range(1,len(path)):
                                if path[i-1] == temppath[j-1] and path[i] == temppath[j]:
                                    found = True    #found edge in previous path
                                
                                
                    if found == False:
                        cnt +=1
                        paths.append([u,neighbor1,neighbor2,v])
                        # add 1 to count only if an edge in this path is not in a previous path       
                                
                
                if cnt >=3:
                    break
            
            if v in u1neighbors:
                
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
        
        if cnt <=2:     #cnt must be 3 or greater to remain in the graph
            P.remove_edge(u,v)
            deleted_some = True

#pos is a dictionary of node point locations in the x,y plane
#can change this to show that the graph is planar

pos = nx.spring_layout(P)

pos[1] = [0,1]
pos[2] = [.33,1]
pos[3] = [.66,1]
pos[4] = [1,1]
pos[5] = [0,.66]
pos[6] = [.33,.66]
pos[7] = [.66,.66]
pos[8] = [1,.66]
pos[9] = [0,.33]
pos[10] = [.33,.33]
pos[11] = [.66,.33]
pos[12] = [1,.33]
pos[13] = [0,0]
pos[14] = [.33,0]
pos[15] = [.66,0]
pos[16] = [1,0]
pos[17] = [.5,-.1]
pos[18] = [.5,.5]
pos[19] = [.2,.2]
pos[20] = [1.1,1.1]
pos[21] = [.17,1.1]
pos[22] = [.17,.8]
pos[23] = [.8,.8]
pos[24] = [.8,.2]


nx.draw_networkx(G, pos,with_labels=False)
plt.axis('off')
plt.savefig('entiregraph.png')
plt.clf()

nx.draw_networkx(P, pos,with_labels=False)
plt.axis('off')
plt.savefig('planargraph.png')