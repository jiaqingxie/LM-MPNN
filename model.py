import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential
from torch_geometric.nn import NNConv, Set2Set


class ChemBERTaForPropertyPrediction(nn.Module):
    def __init__(self, chemberta_model):
        super().__init__()
        self.chemberta = chemberta_model
        self.regressor = nn.Linear(chemberta_model.config.hidden_size, 1)  # Adjust the output size

    def forward(self, input_ids, attention_mask=None):
        outputs = self.chemberta(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        return self.regressor(hidden_states[:,0]).view(-1), hidden_states[:,0]


class NNConvModel(nn.Module):
    def __init__(self, num_features, dim, graph_embedding_dim):
        super().__init__()
        self.lin0 = Linear(num_features, dim)

        nn = Sequential(Linear(4, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, graph_embedding_dim)
        self.lin2 = torch.nn.Linear(graph_embedding_dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        graph_embed = self.lin1(out)
        out = F.relu(graph_embed)
        out = self.lin2(out)
        return out.view(-1), graph_embed


class HighwayGateLayer(nn.Module):
    def __init__(self, in_out_size, bias=True):
        super(HighwayGateLayer, self).__init__()
        self.transform = nn.Linear(in_out_size, in_out_size, bias=bias)

    def forward(self, x, y):
        out_transform = torch.sigmoid(self.transform(x))
        return out_transform * x + (1 - out_transform) * y


class FusionModel(nn.Module):
    def __init__(self, chemberta_model, num_features, dim, graph_embedding_dim):
        super().__init__()
        self.bert_model = ChemBERTaForPropertyPrediction(chemberta_model)
        self.gnn_model = NNConvModel(num_features, dim, graph_embedding_dim)
        self.gate = HighwayGateLayer(graph_embedding_dim)
        self.regressor = nn.Linear(chemberta_model.config.hidden_size, 1)

    def forward(self, batch):
        _, bert_embed = self.bert_model(batch.input_ids, batch.attention_mask)
        _, gnn_embed = self.gnn_model(batch)
        fusion_embed = self.gate(bert_embed, gnn_embed)
        return self.regressor(fusion_embed)

