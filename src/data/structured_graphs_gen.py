import torch
import networkx as nx
from torch_geometric.utils.convert import from_networkx


def create_clique(n):
	clique = nx.complete_graph(n)
	clique = from_networkx(clique)
	clique.x = torch.ones((n, 4))

	return clique

def create_grid(n, m):
	grid = nx.grid_2d_graph(n, m)
	grid = from_networkx(grid)
	grid.x = torch.ones((n*m, 4))

	return grid

def create_general_petersen_graph(n, k):
	general_petersen_graph = nx.Graph()
	general_petersen_graph.add_nodes_from(range(2*n))

	for i in range(n):
		general_petersen_graph.add_edge(i, (i+1)%n)
		general_petersen_graph.add_edge(i, n+i)
		general_petersen_graph.add_edge(n+i, (i+k)%n+n)

	general_petersen_graph = from_networkx(general_petersen_graph)
	general_petersen_graph.x = torch.ones((2*n, 4))

	return general_petersen_graph


for i in range(3, 11):
	torch.save(create_clique(i), 'data/clique/clique_'+str(i)+'.pt')
for n in range(2, 9):
	for m in range(n, 9):
		torch.save(create_grid(n, m), 'data/grid/grid_'+str(n)+'_'+str(m)+'.pt')
for n in range(5,11):
	for k in range(1,n//2+1):
		torch.save(create_general_petersen_graph(n, k), 'data/gp/general_petersen_'+str(n)+'_'+str(k)+'.pt')