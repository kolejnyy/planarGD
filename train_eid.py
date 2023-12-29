import torch
import numpy as np

from src.eid.model import EdgeIntersectionDetector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_x = torch.load('data/eid/x.pt').to(device).view(-1, 8)
train_y = torch.load('data/eid/y.pt').to(device)

val_x = torch.load('data/eid/val_x.pt').to(device).view(-1, 8)
val_y = torch.load('data/eid/val_y.pt').to(device)

model = EdgeIntersectionDetector().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

for epoch in range(10000):
	model.train()
	optimizer.zero_grad()

	y_pred = model(train_x)
	loss = criterion(y_pred, train_y)
	loss.backward()
	optimizer.step()

	model.eval()
	with torch.no_grad():
		val_y_pred = model(val_x)
		val_loss = criterion(val_y_pred, val_y)
		val_acc = ((val_y_pred > 0.5) == val_y).float().mean()

	print(f'Epoch {epoch} | Train Loss: {loss.item():.3f} | Val Loss: {val_loss.item():.3f} | Val Acc: {val_acc.item():.3f}')

	if epoch % 1000 == 0:
		torch.save(model.state_dict(), 'weights/eid/model_epoch'+str(epoch+1)+'.pt')
