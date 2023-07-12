import argparse

import numpy as np
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=1998, help='Random seed.')
parser.add_argument('--dataset', type=str, default='Cora', help='Dataset',
                    choices=['Cora', 'Citeseer', 'Pubmed', 'Flickr', 'ogbn-arxiv', 'Reddit2'])

# model settings
parser.add_argument('--model', type=str, default='GCN', help='model', choices=['GCN', 'GAT', 'GraphSAGE', 'GIN'])
parser.add_argument('--num_layers', type=int, default='2', help='conv layer num')
parser.add_argument('--hidden_dim', type=list, default=[32], help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout_training', type=bool, default=True)
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train benign and backdoor model.')

# trigger settings
parser.add_argument('--trigger_size', type=int, default=3, help='trigger_size')
parser.add_argument('--num_poisoned_node', type=int, default=80, help="number of poisoning nodes")
parser.add_argument('--target_class', type=int, default=0, help='fake label')
parser.add_argument('--selection_method', type=str, default='cluster_degree',
                    choices=['cluster', 'random', 'cluster_degree'],
                    help='Method to select idx_attach for training trojan model')

# AtkNodeSelector
# trigger generator settings
parser.add_argument('--tg_hidden_dim', type=int, default=32, help='trigger generator hidden dim')

parser.add_argument('--thrd', type=float, default=0.5, help='Threshold')

parser.add_argument('--trojan_epochs', type=int, default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int, default=20, help='Number of inner')
parser.add_argument('--debug', type=bool, default=True)

# backdoor setting
parser.add_argument('--degree_weight', type=float, default=1, help="Weight of cluster distance")

parser.add_argument('--use_vs_number', default=True, help="if use detailed number to decide Vs")

parser.add_argument('--vs_ratio', type=float, default=0, help="ratio of poisoning nodes relative to the full graph")

# defense setting
parser.add_argument('--defense_mode', type=str, default="none", choices=['prune', 'isolate', 'none'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.8, help="Threshold of prunning edges")
parser.add_argument('--target_loss_weight', type=float, default=1, help="Weight of optimize outter trigger generator")
parser.add_argument('--homo_loss_weight', type=float, default=100, help="Weight of optimize similarity loss")
parser.add_argument('--homo_boost_thrd', type=float, default=0.8, help="Threshold of increase similarity")

# attack setting


parser.add_argument('--test_model', type=str, default='GCN', choices=['GCN', 'GAT', 'GraphSage', 'GIN'],
                    help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='1by1', choices=['overall', '1by1'],
                    help='Model used to attack')

# GPU setting
parser.add_argument('--device_id', type=int, default=0)

# args process
args = parser.parse_args()

args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device


