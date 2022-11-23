import torch
import torch.nn as nn
from torch_geometric.nn import SAGPooling, TransformerConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class Graph_Transformer(nn.Module):
    def __init__(self, input_dim, head_num, hidden_dim):
        super(Graph_Transformer, self).__init__()
        #  multi-head self-attention
        self.graph_conv = TransformerConv(input_dim, input_dim//head_num, head_num)
        self.lin_out = nn.Linear(input_dim, input_dim)

        # feed forward network
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        #  multi-head self-attention
        out1 = self.lin_out(self.graph_conv(x, edge_index))

        # feed forward network
        out2 = self.ln1(out1 + x)
        out3 = self.lin2(self.act(self.lin1(out2)))
        out4 = self.ln2(out3 + out2)

        return out4


class GraphNet(nn.Module):
    def __init__(self, input_dim, head_num=4, hidden_dim=64, ratio=0.8):
        super(GraphNet, self).__init__()
        self.conv1 = Graph_Transformer(input_dim, head_num, hidden_dim)
        self.pool1 = SAGPooling(input_dim, ratio)
        self.conv2 = Graph_Transformer(input_dim, head_num, hidden_dim)
        self.pool2 = SAGPooling(input_dim, ratio)
        self.conv3 = Graph_Transformer(input_dim, head_num, hidden_dim)
        self.pool3 = SAGPooling(input_dim, ratio)
        self.lin1 = nn.Linear(input_dim*2, input_dim)
        self.lin2 = nn.Linear(input_dim, input_dim//2)
        self.lin3 = nn.Linear(input_dim//2, 1)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim//2)
        self.act = nn.ReLU()

    def forward(self, graph_data):
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        x = self.conv1(x, edge_index)

        x, edge_index, _, batch, perm, score = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index)

        x, edge_index, _, batch, perm, score = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv3(x, edge_index)

        x, edge_index, _, batch, perm, score = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # multi-level features from read out layers
        x_feature = x1 + x2 + x3

        x_out = self.act(self.bn1(self.lin1(x_feature)))
        x_out = self.act(self.bn2(self.lin2(x_out)))
        x_out = self.lin3(x_out).squeeze(1)

        return x_out
