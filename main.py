# -------------------------------------- ArgParse -------------------------------------- #
import torch
from torch_geometric.utils import to_undirected

from attacks.atk_node_selector import AtkNodeSelector
from attacks.gta import GTA
from data_processor import DataProcessor
from defend.prune import Prune
from model_constructor import ModelConstructor
from arg_processor import args
# from models.GAN import GAN

# 数据获取
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
mc.test(idx_test=idx_clean_test)

# 后门攻击
gta = GTA(args=args, data=data)
gta.fit(idx_train=idx_train, idx_valid=idx_val, benign_model=mc.model)
gta.test(idx_atk=idx_atk_test, backdoor_model=mc.model)

mc.test(idx_test=idx_clean_test)
data.edge_index, data.edge_weight = Prune(data, args).prune_edge()

trigger_ids = AtkNodeSelector(args, data).random_obtain_trigger_ids(idx_train)
select_nodes = data.x[trigger_ids]

# gan = GAN(select_nodes.shape[1], select_nodes.shape[1], args.device)
# gan.train(100, select_nodes)
# gan.generate(select_nodes[0])