# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:54:31 2016

@author: ericburas
"""
import networkx as nx
import kl_connected_subgraph as kl
import numpy as np



#proof that package in network x is not correct
F = nx.read_gml('celegansneural.gml')

G = nx.Graph()
for i in range(0,nx.number_of_nodes(F)):
    #print i
    for j in range(0,nx.number_of_nodes(F)):
        if F.has_edge(i,j):
            if not G.has_edge(i,j):
                G.add_edge(i,j)


#G = nx.read_edgelist('newmetabolic.txt',nodetype=int)
#
#selflist = G.selfloop_edges()
#for edge in selflist:
#    (u,v) = edge
#    G.remove_edge(u,v)

P = G.copy()

    #get all paths of length 2 and 3 
paths = []
for node in P.nodes():
    firstneighbors = P.neighbors(node)
    
    for neighbor1 in firstneighbors:
        secondneighbors = P.neighbors(neighbor1)
        
        for neighbor2 in secondneighbors:
            thirdneighbors = P.neighbors(neighbor2)
            paths.append([node,neighbor1,neighbor2])
            for neighbor3 in thirdneighbors:
                paths.append([node,neighbor1,neighbor2,neighbor3])
                    
        
        
                    
for path in paths:
    if path[0]==63 and path[-1]==81:
        print path
        
#only 2 edge-disjoint paths: 11-54   11-3-14-54    neural
        
#P1 = kl.kl_connected_subgraph(G,3,3,low_memory=True,same_as_graph=False)

#38-44 for metabolic
                    


        
                
    
                
            
        