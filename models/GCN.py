import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from torch_geometric.nn import GCNConv

import utils


class GCN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, dropout_training=True, num_layers=2, device=None):

        super(GCN, self).__init__()
        assert device is not None, "Please specify 'device'!"

        # config
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.dropout_training = dropout_training

        # layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim[0]))
        for i in range(num_layers-2):
            self.convs.append(GCNConv(hidden_dim[i], hidden_dim[i+1]))
        self.convs.append(GCNConv(hidden_dim[-1], output_dim))

        # data
        self.Y = None
        self.X = None
        self.A = None
        self.W = None

    def forward(self, X, A, W):
        H = X
        for conv in self.convs[:-1]:
            H = F.relu(conv(H, A, W))
            H = F.dropout(H, self.dropout, training=self.dropout_training)
        Z = self.convs[-1](H, A)
        return F.log_softmax(Z, dim=1)

    def get_h(self, X, A, W):
        H = X
        for conv in self.convs[:-1]:
            H = F.relu(conv(H, A, W))
            H = F.dropout(H, self.dropout, training=self.dropout_training)
        return H