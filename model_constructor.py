from copy import deepcopy

from torch_geometric.graphgym import optim

import utils
from models.GAT import GAT
from models.GCN import GCN
from models.GraphSAGE import GraphSAGE
import torch.nn.functional as F

from models.RobustGCN import RobustGCN


class ModelConstructor:
	
	def __init__(self, data, args):
		assert args.model in ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'RobustGCN'], "Please specify 'model' correctly!"
		self.model = None
		self.data = data
		self.args = args
		if args.model is 'GCN':
			self.model = GCN(input_dim=data.x.shape[1],
			                 output_dim=int(data.y.max() + 1),
			                 num_layers=args.num_layers,
			                 hidden_dim=args.hidden_dim,
			                 dropout=args.dropout,
			                 dropout_training=args.dropout_training,
			                 device=args.device).to(args.device)
		
		elif args.model is 'GraphSAGE':
			self.model = GraphSAGE(input_dim=data.x.shape[1],
			                       output_dim=int(data.y.max() + 1),
			                       num_layers=args.num_layers,
			                       hidden_dim=args.hidden_dim,
			                       dropout=args.dropout,
			                       dropout_training=args.dropout_training,
			                       device=args.device).to(args.device)
		
		elif args.model is 'GAT':
			self.model = GAT(input_dim=data.x.shape[1],
			                 output_dim=int(data.y.max() + 1),
			                 num_layers=args.num_layers,
			                 hidden_dim=args.hidden_dim,
			                 heads=8,
			                 dropout=args.dropout,
			                 dropout_training=args.dropout_training,
			                 device=args.device).to(args.device)
		
		elif args.model is 'RobustGCN':
			self.model = RobustGCN(input_dim=data.x.shape[1],
			                       output_dim=int(data.y.max() + 1),
			                       num_layers=args.num_layers,
			                       hidden_dim=args.hidden_dim,
			                       dropout=args.dropout,
			                       dropout_training=args.dropout_training,
			                       device=args.device).to(args.device)
	
	def fit(self, X, A, W, Y, idx_train, idx_val=None, train_iters=200, verbose=True):
		self.model.A = A.to(self.args.device)
		self.model.X = X.to(self.args.device)
		self.model.Y = Y.to(self.args.device)
		self.model.W = W.to(self.args.device)
		
		if idx_val is None:
			self.__train_without_val(idx_train, train_iters, verbose)
		else:
			self.__train_with_val(idx_train, idx_val, train_iters, verbose)
	
	def get_h(self, X, A, W):
		return self.model.get_h(X, A, W)
	
	def __train_without_val(self, idx_train, train_iters, verbose):
		X = self.model.X
		Y = self.model.Y
		W = self.model.W
		A = self.model.A
		
		if verbose:
			print('=== training model ===')
		optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
		for epoch in range(train_iters):
			self.model.train()
			optimizer.zero_grad()
			output = self.model.forward(X, A, W)
			loss_train = F.nll_loss(output[idx_train], Y[idx_train])
			loss_train.backward(retain_graph=True)
			optimizer.step()
			acc_train = utils.accuracy(output[idx_train], Y[idx_train])
			if verbose and epoch % 10 == 0:
				print('[Epoch:{:03d}]-[Loss:{:.4f}]-[TrainAcc:{:.4f}]'.format(epoch, loss_train, acc_train))
	
	def __train_with_val(self, idx_train, idx_val, train_iters, verbose):
		X = self.model.X
		Y = self.model.Y
		W = self.model.W
		A = self.model.A
		
		if verbose:
			print('=== training model ===')
		optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
		for epoch in range(train_iters):
			self.model.train()
			optimizer.zero_grad()
			output = self.model.forward(X, A, W)
			loss_train = F.nll_loss(output[idx_train], Y[idx_train])
			loss_train.backward(retain_graph=True)
			optimizer.step()
			acc_train = utils.accuracy(output[idx_train], Y[idx_train])
			self.model.eval()
			output = self.model.forward(X, A, W)
			acc_val = utils.accuracy(output[idx_val], Y[idx_val])
			if verbose and epoch % 10 == 0:
				print('[Epoch:{:03d}]-[Loss:{:.4f}]-[TrainAcc:{:.4f}]-[ValidAcc:{:.4f}]'.format(
					epoch, loss_train, acc_train, acc_val))
	
	def test(self, idx_test):
		X = self.model.X
		Y = self.model.Y
		W = self.model.W
		A = self.model.A
		
		self.model.eval()
		output = self.model.forward(X, A, W)
		acc_test = utils.accuracy(output[idx_test], Y[idx_test])
		print('[TestAcc:{:.4f}]'.format(acc_test))
