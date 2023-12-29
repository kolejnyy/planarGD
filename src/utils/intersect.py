import torch


def intersect(p1, p2, p3 ,p4):
	x1, y1 = p1
	x2, y2 = p2
	x3, y3 = p3
	x4, y4 = p4
	
	d1 = (y2-y1)/(x2-x1)
	d2 = (y4-y3)/(x4-x3)

	c1 = y1-d1*x1
	c2 = y3-d2*x3

	if d1==d2:
		if c1 != c2:
			return False
		else:
			return not (max(x1,x2)<min(x3,x4) or max(x3,x4)<min(x1,x2))

	# t = x3+t'-x1
	# t = (y3-y1+d2*t')/d1

	tp = ((y3-y1)/d1 + x1 - x3)/(1-d2/d1)

	xp = x3 + tp
	yp = y3 + tp*d2

	return (min(x1,x2)<=xp<=max(x1,x2) and min(x3,x4)<=xp<=max(x3,x4) and min(y1,y2)<=yp<=max(y1,y2) and min(y3,y4)<=yp<=max(y3,y4))



def intersect_tensor(p1, p2, p3, p4):
	x1, y1 = p1[:,0], p1[:,1]
	x2, y2 = p2[:,0], p2[:,1]
	x3, y3 = p3[:,0], p3[:,1]
	x4, y4 = p4[:,0], p4[:,1]

	d1 = (y2-y1)/(x2-x1)
	d2 = (y4-y3)/(x4-x3)


	c1 = y1-d1*x1
	c2 = y3-d2*x3

	tp = ((y3-y1)/d1 + x1 - x3)/(1-d2/d1)

	xp = x3 + tp
	yp = y3 + tp*d2

	ans = torch.zeros((p1.shape[0],))
	
	# if the points are not unique, check by hand
	ans[(x1==x3) & (y1==y3) & (x2==x4) & (y2==y4)] = 1
	ans[(x1==x4) & (y1==y4) & (x2==x3) & (y2==y3)] = 1

	ans[(d1==d2) & (c1==c2) & (torch.max(x1,x2)>=torch.min(x3,x4)) & (torch.max(x3,x4)>=torch.min(x1,x2))] = 1
	ans[(d1!=d2) & (torch.min(x1,x2)<=xp) & (torch.max(x1,x2)>=xp) & (torch.min(x3,x4)<=xp) & (torch.max(x3,x4)>=xp) & (torch.min(y1,y2)<=yp) & (torch.max(y1,y2)>=yp) & (torch.min(y3,y4)<=yp) & (torch.max(y3,y4)>=yp)] = 1

	# if only one point is the same, zeor it out
	ans[(x1==x3) & (y1==y3) & (x2==x4) & (y2!=y4)] = 0
	ans[(x1==x3) & (y1==y3) & (x2!=x4) & (y2==y4)] = 0
	ans[(x1==x3) & (y1==y3) & (x2!=x4) & (y2!=y4)] = 0
	ans[(x1==x4) & (y1==y4) & (x2==x3) & (y2!=y3)] = 0
	ans[(x1==x4) & (y1==y4) & (x2!=x3) & (y2==y3)] = 0
	ans[(x1==x4) & (y1==y4) & (x2!=x3) & (y2!=y3)] = 0
	ans[(x2==x3) & (y2==y3) & (x1==x4) & (y1!=y4)] = 0
	ans[(x2==x3) & (y2==y3) & (x1!=x4) & (y1==y4)] = 0
	ans[(x2==x3) & (y2==y3) & (x1!=x4) & (y1!=y4)] = 0
	ans[(x2==x4) & (y2==y4) & (x1==x3) & (y1!=y3)] = 0
	ans[(x2==x4) & (y2==y4) & (x1!=x3) & (y1==y3)] = 0
	ans[(x2==x4) & (y2==y4) & (x1!=x3) & (y1!=y3)] = 0

	return ans.bool()


def evaluate_intersections(batch, positions):

	edge_index = batch.edge_index[:, batch.edge_index[0]<batch.edge_index[1]]
	cart = torch.cartesian_prod(torch.arange(edge_index.shape[1]),torch.arange(edge_index.shape[1]))
	unique = cart[cart[:,0]<cart[:,1]]
	segments = positions[edge_index.T[unique].long()].view(-1,8)

	return intersect_tensor(segments[:,[0,1]], segments[:,[2,3]], segments[:,[4,5]], segments[:,[6,7]]).sum().item()
