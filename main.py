# -------------------------------------- ArgParse -------------------------------------- #
import argparse

import numpy as np
import torch
from torch_geometric.utils import to_undirected

from attacks.atk_node_selector import AtkNodeSelector
from attacks.gta import GTA
from attacks.ugba import UGBA
from data_processor import DataProcessor
from defend.prune import Prune
from model_constructor import ModelConstructor
from arg_processor import args
from models import TriggerGenerator
from models.RobustGCN import RobustGCN

dp = DataProcessor(args=args)
dp.get_split()
data = dp.data
idx_train = data.idx_train
idx_val = data.idx_val
idx_clean_test = data.idx_clean_test
idx_atk_test = data.idx_atk_test

data.edge_index = to_undirected(data.edge_index)  # 仅考虑无向图场景
data.edge_weight = torch.ones(data.num_edges, device=args.device) if data.edge_weight is None else data.edge_weight

# 原始模型
mc = ModelConstructor(data=data, args=args)
mc.fit(X=data.x, A=data.edge_index, W=data.edge_weight, Y=data.y, idx_train=idx_train, idx_val=idx_val)
mc.test(Y=data.y, idx_test=idx_clean_test)

ns = AtkNodeSelector(args, data)
poisoned_idx = ns.cluster_degree(idx_train)

data.edge_index, data.edge_weight = Prune(data, args).prune_edge()

gta = GTA(args=args, data=data)

gta.fit(idx_train=idx_train, idx_atk=poisoned_idx, benign_model=mc.model)
gta.test(idx_atk=data.idx_test, backdoor_model=mc)