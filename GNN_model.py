import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool



network_num = 17
compo_num = 54
subject_feature = 2
n_nodes = 200


class BrainGNN(nn.Module):
    def __init__(self, node_dim, global_dim, hidden, conv_hidden,dropout, use_graph=True, use_global=True,readout="network"):
        super().__init__()
        assert use_graph or use_global, "Need at least one branch"
        assert readout in ("network", "node", "compo"), f"Unknown readout: {readout}"
        self.dropout = dropout
        self.use_graph = use_graph
        self.use_global = use_global
        self.readout = readout

        # Only build the branches we use, so an ablated branch costs nothing.
        head_dim = 0
        if use_graph:
            self.conv1 = GCNConv(node_dim, conv_hidden)
            self.conv2 = GCNConv(conv_hidden, conv_hidden)

            n_group = {"network": network_num, "compo": compo_num, "node": n_nodes}[self.readout]
            self.group_dim = n_group*(conv_hidden+subject_feature)
            head_dim += self.group_dim

        if use_global:
            self.global_mlp = nn.Linear(global_dim, hidden)
            head_dim += hidden

        self.head = nn.Linear(head_dim, 1)

    def graph_branch(self, data):
        h = self.conv1(data.x, data.edge_index, data.edge_weight)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.conv2(h, data.edge_index, data.edge_weight)
        h = F.relu(h)

        h = torch.cat([h,data.x[:,:subject_feature]],dim=1)
        # group = data.batch * network_num + data.network_id
        # net_vec = global_mean_pool(h, group)
        # net_vec = net_vec.view(-1,self.group_dim)

        if self.readout == "network":
            group = data.batch * network_num + data.network_id
            h = global_mean_pool(h, group)   
        elif self.readout == "compo":
            group = data.batch * compo_num + data.compo_id
            h = global_mean_pool(h, group) 
        else:
            assert data.num_nodes == data.num_graphs * n_nodes, "Every graph must have 200 nodes"
        
        h = h.view(-1, self.group_dim)
        return h  

    def global_branch(self, data):
        g = self.global_mlp(data.global_x)
        g = F.relu(g)
        g = F.dropout(g, self.dropout, training=self.training)
        return g

    def forward(self, data):
        parts = []
        if self.use_graph:
            parts.append(self.graph_branch(data))
        if self.use_global:
            parts.append(self.global_branch(data))

        combined = torch.cat(parts, dim=1)             
        combined = F.dropout(combined, self.dropout, training=self.training)
        return self.head(combined).squeeze(-1)

if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    from GNN_data import train_graphs
    hidden = 64
    dropout= 0.5

    loader = DataLoader(train_graphs, batch_size=4)
    iterator = iter(loader)
    batch = next(iterator)

    for use_graph, use_global in [(True, True), (True, False), (False, True)]:
        model = BrainGNN(node_dim=batch.x.shape[1], global_dim=batch.global_x.shape[1],
                            hidden=hidden, conv_hidden=4, dropout=dropout,
                            use_graph=use_graph, use_global=use_global)
        logits = model(batch)
        assert logits.shape == (4,), logits.shape
        assert torch.isfinite(logits).all()
        print(f"graph={use_graph}, global={use_global}: "
                f"params {sum(p.numel() for p in model.parameters()):,}")