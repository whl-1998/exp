import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit2, Flickr
from ogb.nodeproppred import PygNodePropPredDataset


class DataProcessor:
    def __init__(self, args):
        assert args.dataset is not None, 'choose the dataset!!!'
        self.data = None
        self.dataset = None
        self.args = args
        assert args.dataset is not None, 'choose the dataset!!!'

        transform = T.Compose([T.NormalizeFeatures()])
        if args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed':
            self.dataset = Planetoid(root='./data/', name=args.dataset, transform=transform)
        elif args.dataset == 'Flickr':
            self.dataset = Flickr(root='./data/Flickr/', transform=transform)
        elif args.dataset == 'Reddit2':
            self.dataset = Reddit2(root='./data/Reddit2/', transform=transform)
        elif args.dataset == 'ogbn-arxiv':
            self.dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/', transform=transform)

        self.data = self.dataset[0].to(device=self.args.device.type)

    def get_split(self):
        random_state = np.random.RandomState(self.args.seed)
        perm = random_state.permutation(self.data.num_nodes)  # get random idx of nodes
        train_number = int(0.2 * len(perm))  # train_set: 20%
        idx_train = sorted(perm[:train_number])
        train_mask = np.full_like(a=perm, fill_value=False, dtype=bool)
        train_mask[idx_train] = True

        val_number = int(0.1 * len(perm))  # val_set: 10%
        idx_val = sorted(perm[train_number: train_number + val_number])
        val_mask = np.full_like(a=perm, fill_value=False, dtype=bool)
        val_mask[idx_val] = True

        clean_test_number = int(0.1 * len(perm))  # clean test_set: 10%
        idx_clean_test = sorted(perm[train_number + val_number: train_number + val_number + clean_test_number])
        clean_test_mask = np.full_like(a=perm, fill_value=False, dtype=bool)
        clean_test_mask[idx_clean_test] = True

        atk_test_number = int(0.1 * len(perm))
        idx_atk_test = sorted(perm[train_number + val_number + clean_test_number:
                                   train_number + val_number + clean_test_number + atk_test_number])
        atk_test_mask = np.full_like(a=perm, fill_value=False, dtype=bool)
        atk_test_mask[idx_atk_test] = True

        self.data.idx_train = torch.tensor(idx_train).long()
        self.data.idx_val = torch.tensor(idx_val).long()
        self.data.idx_clean_test = torch.tensor(idx_clean_test).long()
        self.data.idx_atk_test = torch.tensor(idx_atk_test).long()

        self.data.train_mask = torch.tensor(train_mask)
        self.data.val_mask = torch.tensor(val_mask)
        self.data.clean_test_mask = torch.tensor(clean_test_mask)
        self.data.atk_test_mask = torch.tensor(atk_test_mask)
