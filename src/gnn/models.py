import torch
from torch import nn, functional as F
from torch.nn import Linear

from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class GNN_Block(nn.Module):

	def __init__(self, arch_type, hidden_dim, input_dim=4, output_dim=2, dropout=0.0, batch_norm=False, **kwargs) -> None:
		super().__init__()

		self.arch_type = arch_type
		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.dropout = dropout
		self.batch_norm = batch_norm
		self.LayerNorm = LayerNorm(output_dim)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()
		self.dropout_layer = nn.Dropout(p=self.dropout)

		self.gnn_layers = nn.ModuleList()

		if self.arch_type == 'GCN':
			self.gnn_layers.append(GCNConv(self.input_dim, self.hidden_dim))
			self.gnn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
			self.gnn_layers.append(GCNConv(self.hidden_dim, self.output_dim))
		
		elif self.arch_type == 'GAT':
			self.gnn_layers.append(GATConv(self.input_dim, self.hidden_dim))
			self.gnn_layers.append(GATConv(self.hidden_dim, self.hidden_dim))
			self.gnn_layers.append(GATConv(self.hidden_dim, self.output_dim))
		
		elif self.arch_type == 'GIN':
			self.gnn_layers.append(GINConv(nn.Sequential(Linear(self.input_dim, self.hidden_dim), self.relu, Linear(self.hidden_dim, self.hidden_dim))))
			self.gnn_layers.append(GINConv(nn.Sequential(Linear(self.hidden_dim, self.hidden_dim), self.relu, Linear(self.hidden_dim, self.hidden_dim))))
			self.gnn_layers.append(GINConv(nn.Sequential(Linear(self.hidden_dim, self.hidden_dim), self.relu, Linear(self.hidden_dim, self.output_dim))))
		
		else:
			raise NotImplementedError

	def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
		
		z = x

		for i in range(len(self.gnn_layers)-1):
			x = self.gnn_layers[i](x, edge_index)
			if self.batch_norm:
				x = F.batch_norm(x, batch)
			x = self.relu(x)
			x = self.dropout_layer(x)

		x = self.gnn_layers[-1](x, edge_index)
		
		if z.shape[1] == x.shape[1]:
			x = x + z
		return x



class GNN_Model(nn.Module):

	def __init__(self, arch_type, n_blocks, hidden_dim, RNI=False, input_dim=4, output_dim=2, dropout=0.0, batch_norm=False, **kwargs) -> None:
		super().__init__()

		self.arch_type = arch_type
		self.n_layers = n_blocks
		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.dropout = dropout
		self.batch_norm = batch_norm
		self.LayerNorm_hid = LayerNorm(hidden_dim)
		self.LayerNorm = LayerNorm(output_dim)
		self.RNI = RNI

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()
		self.dropout_layer = nn.Dropout(p=self.dropout)

		self.gnn_layers = nn.ModuleList()

		self.gnn_layers.append(GNN_Block(self.arch_type, self.hidden_dim, self.input_dim, self.hidden_dim, self.dropout, self.batch_norm))
		for i in range(self.n_layers-2):
			self.gnn_layers.append(GNN_Block(self.arch_type, self.hidden_dim, self.hidden_dim, self.hidden_dim, self.dropout, self.batch_norm))
		self.gnn_layers.append(GNN_Block(self.arch_type, self.hidden_dim, self.hidden_dim, self.output_dim, self.dropout, self.batch_norm))


	def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:

		# If self.RNI, randomly select the features of x
		if self.RNI:
			x = 2*torch.rand_like(x)-1
		else:
			x = torch.ones_like(x)

		for i in range(self.n_layers-1):
			x = self.gnn_layers[i](x, edge_index)
			if self.batch_norm:
				x = F.batch_norm(x, batch)
			x = self.relu(x)
			x = self.LayerNorm_hid(x)

		x = self.gnn_layers[-1](x, edge_index)

		x = self.LayerNorm(x)

		return x
		