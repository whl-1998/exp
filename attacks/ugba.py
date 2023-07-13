import torch
from torch import optim
from torch.nn import Dropout, Linear, ReLU
import torch.nn.functional as F

import utils
from attacks.atk_node_selector import AtkNodeSelector

class UGBA:

    def __init__(self, args, data):
        self.args = args

        self.X = data.x
        self.A = data.edge_index
        self.Y = data.y
        self.W = data.edge_weight

        self.idx_atk = AtkNodeSelector(args, data).cluster_degree(data.idx_train)
        self.X_atk = self.X[self.idx_atk]

        self.idx_atk_test = AtkNodeSelector(args, data).cluster_degree(data.idx_test)
        self.X_atk_test = self.X[self.idx_atk_test]

        self.poisoned_A = None
        self.poisoned_X = None
        self.poisoned_W = None
        self.poisoned_Y = None

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

    def poisoned_data(self, trigger_feat, trigger_edge_weight):
        trigger_A = self.get_trigger_edge(len(self.X), self.idx_atk, self.args.trigger_size)
        self.poisoned_A = torch.cat([self.A, trigger_A], dim=1)  # 拼接邻接矩阵

        self.poisoned_X = torch.cat([self.X, trigger_feat])

        self.poisoned_W = torch.cat([self.W, trigger_edge_weight.flatten()])

        Y_temp = self.Y.clone()
        Y_temp[self.idx_atk] = self.args.target_class
        self.poisoned_Y = Y_temp

    def train_trigger_generator(self, benign_gnn_model, poisoned_node_ids):
        pass

    def fit(self, idx_train, idx_atk, benign_model):
        self.tg = TriggerGenerator(self.X.shape[1], self.args.trigger_size, hidden_dim=self.args.hidden_dim).to(self.args.device)
        self.hl = HomoLoss(self.args)

        optimizer_benign_model = optim.Adam(benign_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        optimizer_trigger_generator = optim.Adam(self.tg.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        for i in range(self.args.trojan_epochs):
            self.tg.train()
            trigger_feat, trigger_edge_weight = self.tg.forward(self.X_atk)

            trigger_edge_weight = torch.cat([torch.ones([len(trigger_feat), 1], dtype=torch.float, device=self.args.device), trigger_edge_weight], dim=1)  # 加上链接到中毒节点的权重，累计4条边
            trigger_feat = trigger_feat.view([-1, self.X.shape[1]])

            self.poisoned_data(trigger_feat, trigger_edge_weight)

            benign_model.train()

            for j in range(self.args.inner):
                optimizer_benign_model.zero_grad()
                output = benign_model.forward(self.poisoned_X, self.poisoned_A, self.poisoned_W)
                acc_train = utils.accuracy(output[idx_train], self.Y[idx_train])
                loss_inner = F.nll_loss(output[torch.cat([idx_train, idx_atk])], self.Y[torch.cat([idx_train, idx_atk])])
                if j % 10 == 0:
                    print(
                        'inner epoch: [Epoch:{:03d}]-[Loss:{:.4f}]-[TrainAcc:{:.4f}]'.format(j, loss_inner, acc_train))
                loss_inner.backward(retain_graph=True)
                optimizer_benign_model.step()

            loss_homo = self.hl.forward(self.poisoned_X, self.poisoned_A)
            output = benign_model.forward(self.poisoned_X, self.poisoned_A, self.poisoned_W)
            loss_outer = F.nll_loss(output[torch.cat([idx_train, idx_atk])], self.Y[torch.cat([idx_train, idx_atk])])

            acc_outer_train_clean = utils.accuracy(output[idx_train], self.Y[idx_train])
            acc_outer_train_atk = utils.accuracy(output[torch.cat([idx_train, idx_atk])],
                                                 self.Y[torch.cat([idx_train, idx_atk])])
            asr_outer_atk = utils.accuracy(output[idx_atk], self.Y[idx_atk])

            loss_outer = loss_outer + loss_homo
            loss_outer.backward()
            optimizer_trigger_generator.step()

            if i % 10 == 0:
                print('outer epoch: [Epoch:{:03d}]-[ACC_train_clean:{:.4f}]-[ACC_train_atk:{:.4f}]-'
                      '[ASR_outer_atk:{:.4f}]'.format(i, acc_outer_train_clean, acc_outer_train_atk, asr_outer_atk))

    def test(self, idx_atk, backdoor_model):
        trigger_feat, trigger_edge_weight = self.tg.forward(self.X[idx_atk])

        trigger_edge_weight = torch.cat(
            [torch.ones([len(trigger_feat), 1], dtype=torch.float, device=self.args.device), trigger_edge_weight],
            dim=1)  # 加上链接到中毒节点的权重，累计4条边
        trigger_feat = trigger_feat.view([-1, self.X.shape[1]])
        self.poisoned_data(trigger_feat, trigger_edge_weight)

        output = backdoor_model.forward(self.poisoned_X, self.poisoned_A, self.poisoned_W)

        acc_test = utils.accuracy(output[idx_atk], self.Y[idx_atk])
        print('[acc_test:{:.4f}]'.format(acc_test))


class TriggerGenerator(torch.nn.Module):

    def __init__(self, input_dim, generate_trigger_number, hidden_dim=None, num_layers=2, dropout=0.00):
        super().__init__()
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
        self.feat = Linear(hidden_dim[-1], generate_trigger_number * input_dim)

        # 触发器邻接矩阵生成器: 传入 feat_dim, 输出 n*(n-1)/2 个连边组合
        self.edge = Linear(hidden_dim[-1], int(generate_trigger_number * (generate_trigger_number - 1) / 2))

    def forward(self, input_nodes):
        h = input_nodes
        for lin in self.lins:
            h = lin(input_nodes)
        trigger_feat = self.feat(h)
        trigger_edge = self.edge(h)
        return trigger_feat, trigger_edge


class HomoLoss(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, X, A):
        sims = F.cosine_similarity(X[A[0]], X[A[1]])
        return torch.relu(self.args.thrd - sims).mean()
