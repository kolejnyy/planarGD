import torch
import os
import numpy as np
from os import path, listdir
import pandas as pd
from tqdm import tqdm
import networkx as nx

from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import Data, InMemoryDataset, DataLoader, Dataset

import matplotlib.pyplot as plt


from src.data.rome import RomeDataset
from src.eid.model import EdgeIntersectionDetector
from src.utils.intersect import intersect, intersect_tensor, evaluate_intersections
from src.gnn.models import GNN_Model

# ======================================================
# ================  TRAINING PARAMETERS  ================
# ======================================================


model_type = 'GIN'
rni = False
hidden_dim = 16
num_layers = 4
batch_size = 32
lr = 0.0001

# ======================================================
# ====================  TRAINING  ======================
# ======================================================

exp_name = model_type+('_rni' if rni else '')+'_hd'+str(hidden_dim)+'_nl'+str(num_layers)+'_bs'+str(batch_size)+'_lr'+str(lr)
gnn_model = GNN_Model(model_type, num_layers, hidden_dim, RNI=rni)
gnn_model.cuda()

train_data = RomeDataset('data/rome/train/')
val_data = RomeDataset('data/rome/valid/')

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr)

eid_model = EdgeIntersectionDetector()
eid_model.load_state_dict(torch.load('weights/eid/model_eid.pt'))
eid_model.cuda()
eid_model.eval()


def validate(model):

	model.eval()
	total_cr = 0
	num_planar = 0

	for k in (range(len(val_data))):

		batch = val_data[k]
		batch.to('cuda')

		with torch.no_grad():
			positions = model(batch.x, batch.edge_index.long(), batch.batch)
			cr = evaluate_intersections(batch, positions)
			total_cr += cr
			num_planar += 1 if cr==0 else 0

	model.train()
	return total_cr/len(val_data), num_planar/len(val_data)


best_cr = 1000
best_planar = 0

for epoch_id in range(100):

	total_loss = 0

	for batch in train_loader:
		
		batch.to('cuda')

		edge_batch = batch.batch[batch.edge_index[0].long()]
		batch_edges = [batch.edge_index.long().T[edge_batch==batch_id] for batch_id in range(32)]
		cart = [torch.cartesian_prod(torch.arange(batch_edges[batch_id].shape[0]),torch.arange(batch_edges[batch_id].shape[0])) for batch_id in range(32)]
		unique = [cart[batch_id][cart[batch_id][:,0]<cart[batch_id][:,1]] for batch_id in range(32)]

		optimizer.zero_grad()
		out = gnn_model(batch.x, batch.edge_index.long(), batch.batch)

		loss = 0

		intersections = [eid_model(out[batch_edges[batch_id][unique[batch_id]].view(-1,4)].view(-1,8)).sum() for batch_id in range(batch_size)]

		for batch_id in range(batch_size):
			loss += intersections[batch_id]
		
		total_loss += loss.item()
		loss.backward()
		optimizer.step()

	avg_cr, avg_planar = validate(gnn_model)

	if avg_cr<best_cr:
		torch.save(gnn_model.state_dict(), 'weights/gnn/model_'+exp_name+'_best_cr.pt')
		best_cr = avg_cr

	if avg_planar>best_planar:
		torch.save(gnn_model.state_dict(), 'weights/gnn/model_'+exp_name+'_best_planar.pt')
		best_planar = avg_planar
	

	print("Epoch "+str(epoch_id)+":  avg loss = ", total_loss/len(train_data), '    avg val cr = ', avg_cr, '    avg val planar = ', avg_planar)