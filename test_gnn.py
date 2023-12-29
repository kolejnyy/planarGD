import torch
from tqdm import tqdm

from src.utils.intersect import evaluate_intersections
from src.gnn.models import GNN_Model
from src.data.rome import RomeDataset
from src.data.file_data import FileDataset



def test_model_on_data(gnn_model, test_data, n_rounds):
	
	gnn_model.eval()

	sum_cr = 0
	sum_planar = 0

	for k in (range(len(test_data))):

		min_cr = 10000

		graph = test_data[k]
		graph = graph.to('cuda')

		for i in range(n_rounds):
			with torch.no_grad():
				embedding = gnn_model(graph.x, graph.edge_index.long(), graph.batch)
			cr = evaluate_intersections(graph, embedding)
			min_cr = min(cr, min_cr)
			
		sum_cr += min_cr
		sum_planar += 1 if min_cr==0 else 0

	print('Average CR: '+str(sum_cr/len(test_data)))
	print('Average planar: '+str(sum_planar/len(test_data)))



def test_model(model_type, rni):
	model_path = 'weights/gnn/model_'+model_type+('_rni' if rni else '')+'.pt'
	n_layers = 4 if model_type=='GIN' else 5
	hidden_dim = 16 if model_type=='GIN' else 8

	gnn_model = GNN_Model(model_type, n_layers, hidden_dim, RNI=rni)
	gnn_model.load_state_dict(torch.load(model_path))
	gnn_model.cuda()

	print('Testing model: '+model_path)
	
	print('Rome test set:')
	test_model_on_data(gnn_model, RomeDataset('data/rome/test/'), 25)

	print('Grid test set:')
	test_model_on_data(gnn_model, FileDataset('data/grid/'), 100)

	print('Clique test set:')
	test_model_on_data(gnn_model, FileDataset('data/clique/'), 100)

	print('GP test set:')
	test_model_on_data(gnn_model, FileDataset('data/gp/'), 100)

	print('\n')



for model_type in ['GIN', 'GAT', 'GCN']:
	for rni in [True, False]:
		test_model(model_type, rni)