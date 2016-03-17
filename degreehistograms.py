# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:11:18 2016

@author: ericburas
"""

import networkx as nx
import matplotlib.pyplot as plt
#fh=open('facebook_combined.txt', 'rb')
#G=nx.read_edgelist(fh,nodetype=int)

#G = nx.read_edgelist('newmetabolic.txt',nodetype=int)

#selflist = G.selfloop_edges()
#for edge in selflist:
#    (u,v) = edge
#    G.remove_edge(u,v)
    
#F = nx.read_gml('celegansneural.gml')

#G = nx.Graph()
#for i in range(0,nx.number_of_nodes(F)):
#    #print i
#    for j in range(0,nx.number_of_nodes(F)):
#        if F.has_edge(i,j):
#            if not G.has_edge(i,j):
#                G.add_edge(i,j)
                
fh=open('newpheno.txt', 'rb')
G=nx.read_edgelist(fh)

#G = nx.read_gml('power.gml')


degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
expected_degree = 0
for i in range(len(degree_sequence)):
    expected_degree += degree_sequence[i]/float(len(degree_sequence))
#print "Degree sequence", degree_sequence
dmax=max(degree_sequence)

plt.plot(degree_sequence,'b-',marker='o')
plt.title("Degree rank plot")
plt.ylabel("Degree")
plt.xlabel("Nodes (Ranked by Degree)")

# draw graph in inset
plt.axes([0.45,0.45,0.45,0.45])
Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]
pos=nx.spring_layout(Gcc)
plt.axis('off')
nx.draw_networkx_nodes(Gcc,pos,node_size=20)
nx.draw_networkx_edges(Gcc,pos,alpha=0.4)

plt.savefig("degree_histogram.png")
plt.show()