import torch, numpy
from torch import nn
from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.data import Data
from torch_geometric.nn import Sequential, GCNConv, RGCNConv, FastRGCNConv, TransformerConv, GATv2Conv


class SimpleAdapter(nn.Module):
    def __init__(self, dim, hidden_dim=None, post_ln=False, dropout=0.1):
        super().__init__()
        self.post_ln = post_ln
        hidden_dim = hidden_dim if hidden_dim is not None else dim
        self.seq = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(dim)
        pass

    def forward(self, x):
        return self.seq(x) + x


class AdpaterWrapper(nn.Module):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter

    def forward(self, pointer_mask, x):
        return self.adapter(x[pointer_mask])


class GraphConvAdapter(nn.Module):
    def __init__(self, input_dim=1024, graph_hid_dim=1024, sample_nodes=False, pre_transform=True,
                 residual=True, dropout=0.0,
                 use_pretrained_gnn=False,
                 nn_layer='GCNConv',
                 num_gnn_layers=1,
                 nn_layer_args=[]
                 ):
        super().__init__()
        self.sample_nodes = sample_nodes
        self.pre_transform = pre_transform
        self.residual = residual
        self.use_pretrained_gnn = use_pretrained_gnn
        self.nn_layer = nn_layer

        if pre_transform:
            self.x_transform = nn.Sequential(
                nn.Linear(input_dim, input_dim)
            )

        nn_layer = globals()[nn_layer]

        def nn_layer_init(layer_class, args):
            args += nn_layer_args
            if layer_class == FastRGCNConv or layer_class == RGCNConv:
                return layer_class(*args, num_relations=2), 'x, edge_index, edge_type -> x'
            else:
                return layer_class(*args, add_self_loops=False),  'x, edge_index -> x'

        gnn_layers_seq = []
        for i in range(num_gnn_layers):
            gnn_layers_seq.append(nn_layer_init(nn_layer, [input_dim, graph_hid_dim]))
            gnn_layers_seq.append(nn.GELU())
        gnn_layers_seq.append(nn.Linear(graph_hid_dim, input_dim))
        gnn_layers_seq.append(nn.Dropout(dropout))

        if nn_layer == FastRGCNConv or nn_layer == RGCNConv:
            self.graph_net = Sequential('x, edge_index, edge_type', gnn_layers_seq)
        else:
            self.graph_net = Sequential('x, edge_index', gnn_layers_seq)

    def forward(self, x, mask=None, edges=None):
        nodes_emb = x[mask].clone()

        graph = Data(x=nodes_emb, edge_index=edges)
        nodes_emb = self.graph_net(graph.x, graph.edge_index)

        x[mask] = x[mask] + nodes_emb
        return x, None, None


class MultiGraphConvAdapter(nn.Module):
    def __init__(self, num_layers=2, acum_edges=False, **kwargs):
        super().__init__()
        self.acum_edges = acum_edges
        self.layers = nn.ModuleList([GraphConvAdapter(**kwargs) for _ in range(num_layers)])

    def forward(self, x, pad_mask):
        prev_edges = None
        for i, layer in enumerate(self.layers):
            x, info = layer(x, pad_mask, prev_edges)
            if self.acum_edges:
                prev_edges = info['edges']

        return x, info