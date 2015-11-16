from networkx import *

G=Graph()

G.add_nodes_from([1,2,3,4,5,6,7,8,9])

G.add_edges_from([(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(4,6),(5,8),(8,9),(9,7),(7,8),(7,2)])


for v in nodes(G):
    print('%s %d %f' % (v,degree(G,v),clustering(G,v)))


L = laplacian_matrix(G)

print(L)

eigs_L = laplacian_spectrum(G)

print(eigs_L)
