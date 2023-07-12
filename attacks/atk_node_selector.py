import numpy as np
from torch_geometric.utils import degree

from arg_processor import args
from model_constructor import ModelConstructor
from sklearn.cluster import KMeans


class AtkNodeSelector:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.poisoned_ids = None

    def random_obtain_trigger_ids(self, node_ids):
        np.random.seed(args.seed)
        num_poisoned_node = self.args.num_poisoned_node
        num_poisoned_node = min(len(node_ids), num_poisoned_node)
        self.poisoned_ids = np.sort(np.random.choice(a=node_ids, size=num_poisoned_node, replace=False))
        return self.poisoned_ids

    def cluster_degree(self, node_ids):
        num_poisoned_node = self.args.num_poisoned_node
        num_poisoned_node = min(len(node_ids), num_poisoned_node)

        print("=== start training encoder of cluster degree method ===")
        gcn_encoder = ModelConstructor(args=args, data=self.data)
        gcn_encoder.fit(X=self.data.x, A=self.data.edge_index, Y=self.data.y, W=self.data.edge_weight,
                        idx_train=self.data.idx_train, idx_val=self.data.idx_val)
        gcn_encoder.test(Y=self.data.y, idx_test=self.data.idx_clean_test)
        print("=== end training encoder of cluster degree method ===")
        h = gcn_encoder.get_h(X=self.data.x, A=self.data.edge_index, W=self.data.edge_weight)

        kmeans = KMeans(n_clusters=int(self.data.y.max() + 1), random_state=args.seed)
        kmeans.fit(X=h.detach().cpu().numpy())
        h_c = kmeans.predict(X=h[node_ids].detach().cpu().numpy())
        cluster_centers = kmeans.cluster_centers_
        dis = np.linalg.norm(x=h[node_ids].detach().cpu().numpy() - cluster_centers[h_c], axis=1)
        degree_score = (degree(self.data.edge_index[0]) + degree(self.data.edge_index[1]))[node_ids].detach().cpu().numpy()
        dis_norm = (dis - np.mean(dis)) / np.std(dis)
        degree_score_norm = (degree_score - np.mean(degree_score)) / np.std(degree_score)
        mi = dis_norm + degree_score_norm
        return node_ids[np.argsort(mi)][:num_poisoned_node]

