import torch
from torch import nn, functional as F
from torch.nn import Linear


def create_dataset(n_true, n_false, n_single, n_double, path):
	cnt_true = 0
	cnt_false = 0

	list_true = []
	list_false = []

	while cnt_true<n_true or cnt_false<n_false:
		z = 2*torch.rand((4,2))-1
		
		if intersect(z[0], z[1], z[2], z[3]):
			if cnt_true<n_true:
				list_true.append(z)
				cnt_true += 1
		else:
			if cnt_false<n_false:
				list_false.append(z)
				cnt_false += 1

	one_common_list = []

	for i in range(n_single):
		z = 2*torch.rand((4,2))-1

		r1 = torch.randint(0,2,(1,)).item()
		r2 = torch.randint(0,2,(1,)).item()

		z[2+r1] = z[0+r2]

		one_common_list.append(z)

	two_common_list = []

	for i in range(n_double):
		z = 2*torch.rand((4,2))-1

		r1 = torch.randint(0,2,(1,)).item()
		r2 = torch.randint(0,2,(1,)).item()

		z[2+r1] = z[0+r2]
		z[3-r1] = z[1-r2]

		two_common_list.append(z)


	data_x = torch.cat([torch.stack(list_true), torch.stack(list_false), torch.stack(one_common_list), torch.stack(two_common_list)])
	data_y = torch.cat([torch.ones((n_true,1)), torch.zeros((n_false,1)), torch.zeros((n_single,1)), torch.ones((n_double,1))])

	torch.save(data_x, path + "data_x.pt")
	torch.save(data_y, path + "data_y.pt")

