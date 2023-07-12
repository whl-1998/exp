import torch
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,
                 dropout=0.5, dropout_training=True, num_layers=2, device=None):

        super().__init__()
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
        self.convs.append(SAGEConv(in_channels=input_dim, out_channels=hidden_dim[0], normalize=True))
        for i in range(num_layers - 2):
            self.convs.append(SAGEConv(in_channels=hidden_dim[i], out_channels=hidden_dim[i+1], normalize=True))
        self.gc2 = SAGEConv(in_channels=hidden_dim[-1], out_channels=output_dim)

        # data
        self.A = None
        self.X = None
        self.Y = None
        self.W = None

    def forward(self, X, A, W):
        H = X
        for conv in self.convs:
            H = F.relu(conv(H, A, W))
            H = F.dropout(H, self.dropout, training=self.dropout_training)
        Z = self.gc2(H, A, W)
        return F.log_softmax(Z, dim=1)
