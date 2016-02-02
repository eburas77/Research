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
                    (1,13),(2,14),(3,15),(4,16),(1,4),(5,8),(9,12),(13,16)])
                    
k = 3
l = 3				
P = kl_connected_subgraph(G, k, l, low_memory=False, same_as_graph=False)

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


nx.draw_networkx(G, pos)
plt.savefig('entiregraph.pdf')
plt.clf()

nx.draw_networkx(P, pos)
plt.savefig('planargraph.pdf')