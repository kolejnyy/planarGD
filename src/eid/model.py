import torch
from torch import nn, functional as F
from torch.nn import Linear



class EdgeIntersectionDetector(nn.Module):

	def __init__(self, hidden_dim=64) -> None:
		super().__init__()

		self.fc1 = Linear(8, hidden_dim)
		self.fc2 = Linear(hidden_dim,hidden_dim)
		self.fc3 = Linear(hidden_dim, 1)

		self.hidden_dim = hidden_dim

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.relu(self.fc1(x))
		x = self.relu(self.fc2(x))
		x = self.sigmoid(self.fc3(x))
		return x