# @Time    : 2023/7/13 14:02
# @Author  : emo
from copy import deepcopy

import torch.nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.graphgym import optim
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul

import utils


class RobustGCNConv(torch.nn.Module):
	
	def __init__(self, input_dim, output_dim, act0=F.elu, act1=F.relu, initial=False, dropout=0.5):
		super(RobustGCNConv, self).__init__()
		self.mean_conv = Linear(input_dim, output_dim)
		self.var_conv = Linear(input_dim, output_dim)
		self.act0 = act0
		self.act1 = act1
		self.initial = initial
		self.dropout = dropout
	
	def reset_parameters(self):
		self.mean_conv.reset_parameters()
		self.var_conv.reset_parameters()
	
	def forward(self, mean, var=None, adj0=None, adj1=None):
		if self.initial:
			mean = F.dropout(mean, p=self.dropout, training=self.training)
			var = mean
			mean = self.mean_conv(mean)
			var = self.var_conv(var)
			mean = self.act0(mean)
			var = self.act1(var)
		else:
			mean = F.dropout(mean, p=self.dropout, training=self.training)
			var = F.dropout(var, p=self.dropout, training=self.training)
			mean = self.mean_conv(mean)
			var = self.var_conv(var)
			mean = self.act0(mean)
			var = self.act1(var) + 1e-6  # avoid abnormal gradient
			attention = torch.exp(-var)
			mean = mean * attention
			var = var * attention * attention
			mean = adj0 @ mean
			var = adj1 @ var
		return mean, var


class RobustGCN(torch.nn.Module):
	
	def __init__(self, input_dim, hidden_dim, output_dim,
	             dropout=0.5, dropout_training=True, num_layers=2, device=None):
		
		super(RobustGCN, self).__init__()
		
		# config
		self.device = device
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.dropout = dropout
		self.dropout_training = dropout_training
		
		self.act0 = F.elu
		self.act1 = F.relu
		
		# layers
		self.convs = torch.nn.ModuleList()
		self.convs.append(RobustGCNConv(input_dim=input_dim, output_dim=hidden_dim[0], act0=self.act0, act1=self.act1,
		                                initial=True, dropout=dropout))
		for i in range(num_layers - 2):
			self.convs.append(RobustGCNConv(input_dim=input_dim, output_dim=hidden_dim[i + 1], act0=self.act0,
			                                act1=self.act1, initial=False, dropout=dropout))
		
		self.convs.append(RobustGCNConv(input_dim=input_dim, output_dim=output_dim, act0=self.act0,
		                                act1=self.act1, dropout=dropout))
		
		# data
		self.A = None
		self.X = None
		self.Y = None
		self.W = None
		
		self.reset_parameters()
	
	def reset_parameters(self):
		for layer in self.layers:
			layer.reset_parameters()
	
	def gcn_norm(self, adj_t, order=-0.5, add_self_loops=True):
		if add_self_loops:
			adj_t = fill_diag(adj_t, 1.0)
		deg = sparsesum(adj_t, dim=1)
		deg_inv_sqrt = deg.pow_(order)
		deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
		adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
		adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
		return adj_t
	
	def forward(self, X, A, W):
		# todo W
		adj0, adj1 = gcn_norm(A), gcn_norm(A, order=-1.0)
		mean = X
		var = X
		for conv in self.convs:
			mean, var = conv(mean, var=var, adj0=adj0, adj1=adj1)
		sample = torch.randn(var.shape).to(self.device)
		output = mean + sample * torch.pow(var, 0.5)
		return output.log_softmax(dim=-1)
