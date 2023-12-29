import torch
from os import path, listdir
from torch_geometric.data import Dataset

class FileDataset(Dataset):

	def __init__(self, root, transform=None, pre_transform=None):
		super(FileDataset, self).__init__(root, transform, pre_transform)

		self.root = root
		self.data = [torch.load(path.join(self.root, x)) for x in listdir(self.root)]
		
	def len(self):
		return len(self.data)

	def get(self, idx):
		return self.data[idx]