from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, Linear, BatchNorm1d
from torch_geometric.graphgym import optim
from torch_geometric.nn import GINConv

import utils


class GIN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num=2,
                 learn_eps=False, dropout=0.5, lr=0.01, weight_decay=5e-4, device=None):
        super(GIN, self).__init__()

        # config
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.learn_eps = learn_eps
        self.hidden_dim = hidden_dim

        self.conv1 = GINConv(nn=Sequential(Linear(input_dim, hidden_dim),
                                           ReLU(),
                                           Linear(hidden_dim, hidden_dim),
                                           ReLU(),
                                           BatchNorm1d(hidden_dim)), train_eps=False)

        self.conv2 = GINConv(nn=Sequential(Linear(hidden_dim, hidden_dim),
                                           ReLU(),
                                           Linear(hidden_dim, hidden_dim),
                                           ReLU(),
                                           BatchNorm1d(hidden_dim)), train_eps=False)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)

        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None

    def forward(self, X, A, W=None):
        H = self.conv1(X, A)
        H = self.conv2(H, A)
        H = F.dropout(H, p=self.dropout, training=True)
        Z = self.lin2(H)
        return F.log_softmax(Z, dim=1)

    def fit(self, X, A, Y, idx_train, idx_val=None, train_iters=200, verbose=True):
        self.A = A
        self.X = X.to(self.device)
        self.Y = Y.to(self.device)
        if idx_val is None:
            self._train_without_val(self.Y, idx_train, train_iters, verbose)
        else:
            self._train_with_val(self.Y, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, Y, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.X, self.A, self.W)
            acc_train = utils.accuracy(output[idx_train], Y[idx_train])
            loss_train = F.nll_loss(output[idx_train], Y[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and epoch % 10 == 0:
                print('[Epoch:{:03d}]-[Loss:{:.4f}]-[TrainAcc:{:.4f}]'.format(epoch, loss_train, acc_train))
        self.eval()
        output = self.forward(self.X, self.A, self.W)
        self.output = output

    def _train_with_val(self, Y, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gat model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_acc_val = 0
        weights = None
        for epoch in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.X, self.A, self.W)
            acc_train = utils.accuracy(output[idx_train], Y[idx_train])
            loss_train = F.nll_loss(output[idx_train], Y[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.X, self.A, self.W)
            acc_val = utils.accuracy(output[idx_val], Y[idx_val])

            if verbose and epoch % 10 == 0:
                print('[Epoch:{:03d}]-[Loss:{:.4f}]-[TrainAcc:{:.4f}]-[ValidAcc:{:.4f}]'.format(
                    epoch, loss_train, acc_train, acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def test(self, X, A, Y, idx_test, W=None):
        self.eval()
        output = self.forward(X, A, W)
        acc_test = utils.accuracy(output[idx_test], Y[idx_test])
        print('[TestAcc:{:.4f}]'.format(acc_test))
        torch.cuda.empty_cache()
