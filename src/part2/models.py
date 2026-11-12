import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool
from torch_geometric.utils import dropout_edge


def graph_ptr(data):
    ptr = data.ptr
    if ptr.device != data.x.device:
        ptr = ptr.to(data.x.device)
    return ptr


class RootMLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, data):
        roots = data.x[graph_ptr(data)[:-1]]
        return self.net(roots)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout, layers=2):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1)
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], hidden) for i in range(layers)]
        )
        self.dropout = dropout
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(global_mean_pool(x, data.batch))


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout, layers=2):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1)
        self.convs = nn.ModuleList(
            [SAGEConv(dims[i], hidden, aggr="mean") for i in range(layers)]
        )
        self.dropout = dropout
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(global_mean_pool(x, data.batch))


class GAT(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout, heads=4, layers=2):
        super().__init__()
        self.conv_in = GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout)
        self.conv_out = GATConv(hidden, hidden, heads=1, concat=False, dropout=dropout)
        self.extra = nn.ModuleList(
            [
                GATConv(hidden, hidden, heads=1, concat=False, dropout=dropout)
                for _ in range(max(0, layers - 2))
            ]
        )
        self.dropout = dropout
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, data):
        x = F.relu(self.conv_in(data.x, data.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.extra:
            x = F.relu(conv(x, data.edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv_out(x, data.edge_index))
        return self.head(global_mean_pool(x, data.batch))


def conv_dims(in_dim, hidden, layers):
    dims = [in_dim]
    for layer in range(1, layers):
        dims.append(hidden + (in_dim if layer == 1 else hidden))
    head_dim = hidden + (in_dim if layers == 1 else hidden)
    return dims, head_dim


class BiGCN(nn.Module):
    def __init__(
        self, in_dim, hidden, num_classes, dropout, dropedge=0.2, layers=2
    ):
        super().__init__()
        dims, self.head_dim = conv_dims(in_dim, hidden, layers)
        self.top_down = nn.ModuleList(
            [GCNConv(dims[i], hidden) for i in range(layers)]
        )
        self.bottom_up = nn.ModuleList(
            [GCNConv(dims[i], hidden) for i in range(layers)]
        )
        self.dropout = dropout
        self.dropedge = dropedge
        self.head = nn.Linear(2 * self.head_dim, num_classes)

    def _direction(self, convs, x, edge_index, batch, ptr):
        h = x
        prev_root = x[ptr]
        for layer, conv in enumerate(convs):
            if layer == 0:
                inp = h
            else:
                inp = torch.cat([h, prev_root[batch]], dim=-1)
            h_next = F.relu(conv(inp, edge_index))
            h_next = F.dropout(h_next, p=self.dropout, training=self.training)
            prev_root = h[ptr]
            h = h_next
        out = torch.cat([h, prev_root[batch]], dim=-1)
        return global_mean_pool(out, batch)

    def forward(self, data):
        edge_index = dropout_edge(
            data.edge_index, p=self.dropedge, training=self.training
        )[0]
        ptr = graph_ptr(data)[:-1]
        top_down = self._direction(
            self.top_down, data.x, edge_index, data.batch, ptr
        )
        bottom_up = self._direction(
            self.bottom_up, data.x, edge_index.flip(0), data.batch, ptr
        )
        return self.head(torch.cat([top_down, bottom_up], dim=-1))


def build_model(arch, in_dim, num_classes, cfg):
    hidden = int(cfg["hidden"])
    layers = int(cfg["layers"])
    dropout = float(cfg["dropout"])
    if arch == "mlp":
        return RootMLP(in_dim, hidden, num_classes, dropout)
    if arch == "gcn":
        return GCN(in_dim, hidden, num_classes, dropout, layers)
    if arch == "sage":
        return GraphSAGE(in_dim, hidden, num_classes, dropout, layers)
    if arch == "gat":
        return GAT(in_dim, hidden, num_classes, dropout, int(cfg["heads"]), layers)
    if arch == "bigcn":
        return BiGCN(
            in_dim,
            hidden,
            num_classes,
            dropout,
            float(cfg["dropedge"]),
            layers,
        )
    raise ValueError(f"unknown architecture: {arch}")
