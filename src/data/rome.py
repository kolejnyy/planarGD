import torch
from os import path, listdir
from torch_geometric.data import Dataset

class RomeDataset(Dataset):

	def __init__(self, root, transform=None, pre_transform=None):
		super(RomeDataset, self).__init__(root, transform, pre_transform)

		self.root = root
		self.unfiltered_data = [torch.load(path.join(self.root, x)) for x in listdir(self.root)]
		self.data = [data for data in self.unfiltered_data if data.x.shape[0]<30]

	def len(self):
		return len(self.data)

	def get(self, idx):
		return self.data[idx]