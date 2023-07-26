import torch
from torch import optim
from torch.nn import Dropout, Linear, ReLU
import torch.nn.functional as F

import utils

from attacks.atk_node_selector import AtkNodeSelector


class GTA:
	
	def __init__(self, args, data):
		self.args = args
		
		# 原始数据集
		self.X = data.x
		self.A = data.edge_index
		self.Y = data.y
		self.W = data.edge_weight
		
		# 训练集被注入触发器节点的id
		self.idx_train_atk = AtkNodeSelector(args, data).random_obtain_trigger_ids(data.idx_train)
		self.trigger_size = args.trigger_size
		self.tg_hidden_dim = args.tg_hidden_dim
		self.trojan_epochs = self.args.trojan_epochs
		self.backdoor_model_epochs = self.args.backdoor_model_epochs
		self.tg = TriggerGenerator(self.X.shape[1], self.trigger_size, hidden_dim=self.tg_hidden_dim).to(
			self.args.device)
		
		# self.poisoned_A = None
		# self.poisoned_X = None
		# self.poisoned_W = None
		# self.poisoned_Y = None
	
	def get_trigger_index(self, trigger_size):
		"""
        根据触发器size获取到连通图的边，其中有一条 self-loop 的边
        """
		edge_list = [[0, 0]]
		for j in range(trigger_size):
			for k in range(j):
				edge_list.append([j, k])
		return edge_list
	
	def get_trigger_edge(self, start, idx_attach, trigger_size):
		"""
        输出的边集是无向图边集
        :param start: 触发器节点集的起始 id
        :param idx_attach: 被感染节点 idx
        :param trigger_size: 触发器个数
        :return: 所有连通图触发器的边集
        """
		edge_list = []
		init_edge_list = self.get_trigger_index(trigger_size)
		for idx in idx_attach:
			edges = torch.tensor(init_edge_list, device=self.args.device).long().T
			
			edges[0, 0] = idx
			edges[1, 0] = start
			edges[:, 1:] = edges[:, 1:] + start
			
			edge_list.append(edges)
			start += trigger_size
		edge_index = torch.cat(edge_list, dim=1)
		return edge_index
	
	def poisoned_data(self, idx_atk, trigger_X, trigger_W):
		X = self.X
		Y = self.Y
		W = self.W
		A = self.A
		
		trigger_A = self.get_trigger_edge(len(X), idx_atk, self.args.trigger_size)
		poisoned_A = torch.cat([A, trigger_A], dim=1)  # 拼接邻接矩阵
		poisoned_X = torch.cat([X, trigger_X])
		poisoned_W = torch.cat([W, trigger_W.flatten()])
		Y_temp = Y.clone()
		Y_temp[idx_atk] = self.args.target_class
		poisoned_Y = Y_temp
		
		return poisoned_X, poisoned_A, poisoned_W, poisoned_Y
	
	def train_trigger_generator(self, benign_gnn_model, poisoned_node_ids):
		pass
	
	def fit(self, idx_train, benign_model, idx_valid=None):
		
		X = self.X
		Y = self.Y
		W = self.W
		A = self.A
		
		idx_train_atk = self.idx_train_atk
		
		optimizer_benign_model = optim.Adam(benign_model.parameters(), lr=self.args.lr,
		                                    weight_decay=self.args.weight_decay)
		optimizer_trigger_generator = optim.Adam(self.tg.parameters(), lr=self.args.lr,
		                                         weight_decay=self.args.weight_decay)
		for i in range(self.trojan_epochs):
			self.tg.train()
			# 生成触发器子图的特征以及连边权重
			trigger_X, trigger_W = self.tg.forward(X[idx_train_atk])
			# 将连边的权重补上与攻击节点的连边
			trigger_W = torch.cat(
				[torch.ones([len(trigger_X), 1], dtype=torch.float, device=self.args.device), trigger_W], dim=1)
			trigger_X = trigger_X.view([-1, X.shape[1]])
			
			# 注入触发器到clean数据集
			poisoned_train_X, poisoned_train_A, poisoned_train_W, poisoned_train_Y \
				= self.poisoned_data(idx_train_atk, trigger_X, trigger_W)
			
			for j in range(self.backdoor_model_epochs):
				benign_model.train()
				optimizer_benign_model.zero_grad()
				output = benign_model.forward(poisoned_train_X, poisoned_train_A, poisoned_train_W)
				loss_inner = F.nll_loss(output[idx_train], poisoned_train_Y[idx_train])
				loss_inner.backward(retain_graph=True)
				optimizer_benign_model.step()
				acc_inner_train = utils.accuracy(output[idx_train], poisoned_train_Y[idx_train])
				
				if j % 10 == 0:
					print('backdoor_model_epochs: [Epoch:{:03d}]-[Loss:{:.4f}]-[TrainAcc:{:.4f}]'.format(
						j, loss_inner, acc_inner_train))
				
				if idx_valid is not None and j %10 == 0:
					benign_model.eval()
					output = benign_model.forward(poisoned_train_X, poisoned_train_A, poisoned_train_W)
					acc_val = utils.accuracy(output[idx_valid], Y[idx_valid])
					print('backdoor_model_epochs: [ValidAcc:{:.4f}]'.format(acc_val))
			
			output = benign_model.forward(poisoned_train_X, poisoned_train_A, poisoned_train_W)
			loss_outer = F.nll_loss(output[idx_train_atk], poisoned_train_Y[idx_train_atk])
			loss_outer.backward()
			optimizer_trigger_generator.step()
			asr_train = utils.accuracy(output[idx_train_atk], poisoned_train_Y[idx_train_atk])
	
			if i % 10 == 0:
				print('=== trojan epochs: [Epoch:{:03d}]-[Asr_train:{:.4f}] ==='.format(i, asr_train))
	
	def test(self, idx_atk, backdoor_model):
		X = self.X
		Y = self.Y
		W = self.W
		A = self.A
		tg = self.tg
		
		tg.eval()
		backdoor_model.eval()
		
		trigger_X, trigger_W = tg.forward(X[idx_atk])
		trigger_W = torch.cat(
			[torch.ones([len(trigger_X), 1], dtype=torch.float, device=self.args.device), trigger_W], dim=1)
		trigger_X = trigger_X.view([-1, X.shape[1]])

		poisoned_test_X, poisoned_test_A, poisoned_test_W, poisoned_test_Y \
			= self.poisoned_data(idx_atk, trigger_X, trigger_W)
		
		output = backdoor_model.forward(poisoned_test_X, poisoned_test_A, poisoned_test_W)
		asr_test = utils.accuracy(output[idx_atk], poisoned_test_Y[idx_atk])
		print('[GTA_Asr_test:{:.4f}]'.format(asr_test))


class TriggerGenerator(torch.nn.Module):
	
	def __init__(self, input_dim, tg_size, hidden_dim=None, num_layers=2, dropout=0.00):
		super(TriggerGenerator, self).__init__()
		self.lins = None
		if hidden_dim is None:
			hidden_dim = [32]
		lins = torch.nn.ModuleList()
		lins.append(Linear(input_dim, hidden_dim[0]))
		for i in range(num_layers - 2):
			lins.append(Linear(hidden_dim[i], hidden_dim[i + 1]))
			lins.append(ReLU(inplace=True))
			if dropout > 0:
				lins.append(Dropout(p=dropout))
		
		self.lins = lins
		# 触发器特征生成器: 传入 feat_dim, 输出 3*feat_dim
		self.feat = Linear(hidden_dim[-1], tg_size * input_dim)
		
		# 触发器邻接矩阵生成器: 传入 feat_dim, 输出 n*(n-1)/2 个连边组合
		self.edge = Linear(hidden_dim[-1], int(tg_size * (tg_size - 1) / 2))
	
	def forward(self, input_nodes):
		h = input_nodes
		for lin in self.lins:
			h = lin(input_nodes)
		trigger_feat = self.feat(h)
		trigger_edge = self.edge(h)
		return trigger_feat, trigger_edge
