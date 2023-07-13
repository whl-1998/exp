# @Time    : 2023/7/13 11:33
# @Author  : emo
import torch
import torch.nn.functional as F


class Prune:
	
	def __init__(self, data, args):
		self.args = args
		
		self.X = data.x
		self.A = data.edge_index
		self.W = data.edge_weight
		self.Y = data.y
		
		self.prune_thr = args.prune_thr
		
	def prune_edge(self):
		"""
		只砍掉不满足同质性的连边
		Returns
		-------

		"""
		sims = F.cosine_similarity(self.X[self.A[0]], self.X[self.A[1]])
		
		updated_A = self.A[:, sims > self.prune_thr]
		updated_W = self.W[sims > self.prune_thr]
		
		return updated_A, updated_W
	
	def prune_edge_nodes(self):
		
		sims = F.cosine_similarity(self.X[self.A[0]], self.X[self.A[1]])
		
		updated_A = self.A[:, sims > self.prune_thr]
		updated_W = self.A[:, sims > self.prune_thr]
		nodes_ids = list(set(torch.cat([updated_A[0], updated_A[1]]).tolist()))
		updated_X = self.X[nodes_ids]
		
		return updated_X, updated_A, updated_W
		