# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:11:18 2016

@author: ericburas
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def mzeta(x,q,N=1000):
  s=0
  for j in np.arange(1,N):
    s+= 1./(1.*j+1.*q)**x
  return s


fh=open('facebook_combined.txt', 'rb')
G=nx.read_edgelist(fh,nodetype=int)

#G = nx.read_edgelist('newmetabolic.txt',nodetype=int)

#selflist = G.selfloop_edges()
#for edge in selflist:
#    (u,v) = edge
#    G.remove_edge(u,v)
    
F = nx.read_gml('celegansneural.gml')

G = nx.Graph()
for i in range(0,nx.number_of_nodes(F)):
#    print i
    for j in range(0,nx.number_of_nodes(F)):
        if F.has_edge(i,j):
            if not G.has_edge(i,j):
                G.add_edge(i,j)
                
#fh=open('newpheno.txt', 'rb')
#G=nx.read_edgelist(fh)

#G = nx.read_gml('power.gml')

L = nx.laplacian_matrix(G)
plt.spy(L,precision=0.01, markersize=1)
plt.savefig('fbspy.png')

degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
expected_degree = 0
for i in range(len(degree_sequence)):
    expected_degree += degree_sequence[i]/float(len(degree_sequence))
#print "Degree sequence", degree_sequence
dmax=max(degree_sequence)
gamma = 2.1

#powerlaw = np.arange(1,101)
#powerlaw = np.power(powerlaw, -2.1)
#powerlaw = 100*powerlaw
#powerlaw = powerlaw/(mzeta(gamma,0))
#
#plt.semilogy(powerlaw,'bo')
#plt.ylabel("Degree")
#plt.xlabel("Nodes (Ranked by Degree)")
#plt.savefig('powerlawdeg.png')

powerlaw = np.arange(1,G.number_of_nodes()+1)
powerlaw = np.power(powerlaw, -gamma/(3.5))
powerlaw = G.number_of_nodes()*powerlaw
powerlaw = powerlaw/(mzeta(gamma,0))

powerlaw2 = np.arange(1,G.number_of_nodes()+1)
powerlaw2 = np.power(powerlaw2, -gamma)
powerlaw2 = G.number_of_nodes()*powerlaw2
powerlaw2 = powerlaw2/(mzeta(gamma,0))

plt.semilogy(np.arange(1,G.number_of_nodes()+1),degree_sequence,'-',label = 'neural degree histogram')
plt.semilogy(np.arange(1,G.number_of_nodes()+1),powerlaw2,'--',label = 'regular power law')
plt.ylabel("Degree")
plt.xlabel("Nodes (Ranked by Degree)")
plt.legend(prop={'size':14})
plt.savefig('neuralsequenceplot2.png')
# draw graph in inset
plt.axes([0.45,0.45,0.45,0.45])
Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]
pos=nx.spring_layout(Gcc)
plt.axis('off')
nx.draw_networkx_nodes(Gcc,pos,node_size=20)
nx.draw_networkx_edges(Gcc,pos,alpha=0.4)

plt.savefig("power_degree_histogram.png")
plt.show()