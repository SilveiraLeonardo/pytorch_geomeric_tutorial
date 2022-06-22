import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import global_mean_pool

class Net(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
		super().__init__()
		self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
		self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations)
		self.task = "graph"

	def forward(self, x, edge_index, edge_type):
		# x => [num_nodes, in_channels]
		# edge_type => The one-dimensional relation type/index for each edge
		x = self.conv1(x, edge_index, edge_type)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_type)

		if self.task == "graph":
			x = global_mean_pool(x)

		return F.log_softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net(data.num_features, args.hidden_channels, data.num_classes).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

def train():
	model.train()
	optimizer.zero_grad()
	out = model(data.x, data.edge_index, data.edge_type)
	# get the loss only for the training nodes
	loss = F.nll_loss(out[data.train_idx], data.train_y) 
	loss.backward()
	optimizer.step()
	return float(loss)

@torch.no_grad()
def test():
	model.eval()
	pred = model(data.x, data.edge_index, data.edge_type).argmax(dim=-1)
	train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
	test_acc = float((pred[data.test_idx] == data.test_y).float().mean())
	return train_acc, test_acc

for epoch in range(1, 51):
	loss = train()
	train_acc, test_acc = test()
	print("[INFO] Epoch {}, Loss {}, Train Acc {}, Test Acc {}".format(epoch, loss, train_acc, test_acc))

