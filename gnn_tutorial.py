import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import numpy as np
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class GNNStack(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, task="node"):
		super(GNNStack, self).__input__()
		self.task = task
		self.convs = nn.ModuleList()
		self.convs.append(self.build_conv_model(input_dim, hidden_dim))
		self l in range(2):
			self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

		# post-message_passing
		# in graph classification often times it is beneficial to 
		# have a few more layers of message passing
		self.post_mp = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
			nn.Linear(hidden_dim, output_dim))

		self.dropout = 0.25
		self.num_layers = 3

	def build_conv_model(self, input_dim, hidden_dim):
		if self.task=="node":
			return pyg_nn.GCNConv(input_dim, hidden_dim)
		else:
			return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim,hidden_dim),
								nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

				
	def forward(self, data):
		# this batch is for graph classification,
		# if we want to batch many graphs together
		# this tell which node belongs to which graph
		x, edge_index, batch = data.x, data.edge_index, data.batch
		if data.num_node_features == 0
			x = torch.ones(data.num_nodes, 1)

		for i in range(self.num_layers):
			x = self.convs[i](x, edge_index)
			emb = edge_index
			x = F.relu(x)
			x = F.dropout(x, p=self.dropout, training=self.training)

		if self.task == "graph":
			x = pyg_nn.nn.global_mean_pool(x, batch)

		x = self.post_mp(x)

		return emb, F.log_softmax(x, dim=1)

	def loss(self, pred, label):
		return F.nll_loss(pred, label)


class CustomConv(pyg_nn.MessagePassing):
	def __init__(self, in_channels, out_channels):
		super(CustomConv, self).__init__(aggr="add") # add, mean, max aggregations
		self.lin = nn.Linear(in_channels, out_channels)

	def forward(self, x, edge_index):
		# x has shape [N, in_channels]
		# edge_index has shape [2, E]

		# add self-loops to the adjacency matrix
		edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=x.size(0))

		# transform node features
		x = self.lin(x)

		# the self.propagate gets the neighborhood information
		# the aggregation is done by the propagate method
		return self.propagate(edge_index, size=(x.size(0), x.size(0)), x = x)

	# the message function is called inside propagate
	# x_j is the neighboorhood aggregation features
	# it can also call x_i, which the own node features
	# ex: def message(self, x_i, x_j, edge_index, size):
	def message(self, x_j, edge_index, size):
		row, col = edge_index
		reg = pyg_nn.degree(row, size[0], dtype=x_j.dtype)
		deg_inv_sqrt = deg.pow(-0.5)
		norm = get_inv_sqrt[row] * deg_inv_sqrt[col]

		return norm.view(-1, 1)*x_j

	# update: after you do the aggregation, do you have
	# additional layers or operations you want to do
	# ex: normalization, MLP, etc.
	def update(self, aggr_out):
		# aggr_out has shape [N, out_channels]

		return aggr_out


def train(dataset, task, writer):
	if task == "graph":
		data_size = len(dataset)
		loader = DataLoader(dataset[:int(data_size*0.8)], batch_size=64, shuffle=True)
		test_loader = DataLoader(dataset[int(data_size*0.8):], batch_size=64, shuffle=True)

	else:
		# if we are performing node classification
		# the test nodes are inside the same graph
		# (we do not divide the graph)
		test_loader = loader = DataLoader(dataset, batch_size=64, shuffle=True)

	# build model
	# input_dim, hidden_dim, output_dim, task="node"
	model = GNNStack(max(dataset.num_node_features,1), 32, dataset.num_classes, task=task)
	opt = optim.Adam(model.parameters(), lr=0.01)

	# train
	for epoch in range(200):
		total_loss = 0
		model.train()
			for batch in loader:
				opt.zero_grad()
				embedding, pred = model(batch)
				label = batch.y
				if task == node:
					pred = pred[batch.train_mask]
					label = label[batch.train_mask]
				loss = model.loss(pred, label)
				loss.backward()
				opt.step()
				total_loss += loss.item() * batch.num_graphs
			total_loss /= len(loader.dataset)
			# writer.add_scalar("loss", total_loss, epoch)

			if epoch%10 == 0:
				test_acc = test(test_loader, model)
				print("[INFO] Epoch: {}, loss: {}, test_acc {}".format(
					epoch, total_loss, test_acc))

	return model

def test(loader, model, is_validation=False):
	model.eval()

	correct = 0
	for data in loader:
		with torch.no_grad():
			emb, pred = model(data)
			pred = pred.argmax(dim=1)
			label = data.y

		if model.task == "node":
			mask = data.val_mask if is_validation else data.test_mask
			pred = pred[mask]
			label = data.y[mask]			

		correct += pred.eq(label).sum().item()

	if model.task == "graph":
		total = len(loader.dataset)
	else:
		total = 0
		for data in loader.dataset:
			total += torch.sum(data.test_mask).item()
	return correct/total

# writer = something_tensorboard
dataset = TUDataset()
dataset = dataset.shuffle()
task = "graph"

model = train(dataset, task, writer)

# plot using tsne to see if the embeddings of different
# classes are being set apart from each other
color_list = ["red", "orange", "green", "blue", "purple", "brown"]

loader = DataLoader(dataset, batch_size = 64, shuffle=True)

embs = []
colors = []

for batch in loader:
	emb, pred = model(batch)
	embs.append(emb)
	colors += [color_list[y] for y in batch.y]
embs = torch.cat(embs, dim=0)

xs, ys = zip(*TSNE().fit_transform(embs.detach().numpy()))
plt.scatter(xs, ys, color=colors)