# @Time    : 2023/7/13 14:02
# @Author  : emo
from copy import deepcopy

import torch.nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.graphgym import optim
from torch_geometric.nn.conv.gcn_conv import gcn_norm

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
			# print("adj0 mean",adj0,mean.shape)
			mean = adj0 @ mean
			var = adj1 @ var
		# print("mean1",mean.shape)
		return mean, var


class RobustGCN(torch.nn.Module):
	
	def __init__(self, input_dim, hidden_dim, output_dim,
				 dropout=0.5, dropout_training=True, num_layers=2, device=None):
		
		super().__init__()
		
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
	
	def reset_parameters(self):
		for layer in self.layers:
			layer.reset_parameters()
	
	def forward(self, X, A, W):
		adj0, adj1 = gcn_norm(A), gcn_norm(A, order=-1.0)
		# adj0, adj1 = normalize_adj(adj), normalize_adj(adj, -1.0)
		mean = X
		var = X
		for conv in self.convs:
			# print(mean.shape,var.shape)
			mean, var = conv(mean, var=var, adj0=adj0, adj1=adj1)
		sample = torch.randn(var.shape).to(self.device)
		output = mean + sample * torch.pow(var, 0.5)
		return output.log_softmax(dim=-1)

	def fit(self, X, A, W, Y, idx_train, idx_val=None, train_iters=200, verbose=True):
		self.A = A.to(self.args.device)
		self.X = X.to(self.args.device)
		self.Y = Y.to(self.args.device)
		self.W = W.to(self.args.device)

		if idx_val is None:
			self.__train_without_val(Y, idx_train, train_iters, verbose)
		else:
			self.__train_with_val(Y, idx_train, idx_val, train_iters, verbose)


	def __train_without_val(self, Y, idx_train, train_iters, verbose):
		self.model.train()
		optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
		for epoch in range(train_iters):
			optimizer.zero_grad()
			output = self.forward(self.model.X, self.model.A, self.model.W)
			acc_train = utils.accuracy(output[idx_train], Y[idx_train])
			loss_train = F.nll_loss(output[idx_train], Y[idx_train])
			loss_train.backward()
			optimizer.step()
			if verbose and epoch % 10 == 0:
				print('[Epoch:{:03d}]-[Loss:{:.4f}]-[TrainAcc:{:.4f}]'.format(epoch, loss_train, acc_train))
	
	
	def __train_with_val(self, Y, idx_train, idx_val, train_iters, verbose):
		if verbose:
			print('=== training gat model ===')
		optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
		best_acc_val = 0
		weights = None
		for epoch in range(train_iters):
			self.model.train()
			optimizer.zero_grad()
			output = self.model.forward(self.model.X, self.model.A, self.model.W)
			acc_train = utils.accuracy(output[idx_train], Y[idx_train])
			loss_train = F.nll_loss(output[idx_train], Y[idx_train])
			loss_train.backward()
			optimizer.step()
			
			self.model.eval()
			output = self.model.forward(self.model.X, self.model.A, self.model.W)
			acc_val = utils.accuracy(output[idx_val], Y[idx_val])
			
			if verbose and epoch % 10 == 0:
				print('[Epoch:{:03d}]-[Loss:{:.4f}]-[TrainAcc:{:.4f}]-[ValidAcc:{:.4f}]'.format(
					epoch, loss_train, acc_train, acc_val))
			if acc_val > best_acc_val:
				best_acc_val = acc_val
				self.output = output
				weights = deepcopy(self.model.state_dict())
		if verbose:
			print('=== picking the best model according to the performance on validation ===')
		self.model.load_state_dict(weights)
	
	
	def test(self, Y, idx_test):
		self.model.eval()
		output = self.model.forward(self.model.X, self.model.A, self.model.W)
		acc_test = utils.accuracy(output[idx_test], Y[idx_test])
		print('[TestAcc:{:.4f}]'.format(acc_test))