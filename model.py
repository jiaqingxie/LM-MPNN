import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential
from torch_geometric.nn import GCNConv, NNConv, Set2Set, global_max_pool


class ChemBERTaForPropertyPrediction(nn.Module):
    def __init__(self, chemberta_model, out_dim, task):
        super().__init__()
        self.task = task
        self.chemberta = chemberta_model
        self.predictor = nn.Linear(chemberta_model.config.hidden_size, out_dim)

    def forward(self, batch):
        """
        Output: prediction: (batch_size,) or (batch_size, out_dim)
                graph embedding: (batch_size, hidden_size)
                node embedding: (batch_size, seq_len, hidden_size)
        """
        input_ids, attention_mask = batch.input_ids, batch.attention_mask
        outputs = self.chemberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        if self.task == "reg":
            return self.predictor(hidden_states[:,0]).view(-1), hidden_states[:,0], hidden_states
        elif self.task == "clf":
            return F.log_softmax(self.predictor(hidden_states[:,0]), dim=1), hidden_states[:,0], hidden_states


class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_dim, embed_dim, out_dim, task):
        super().__init__()
        self.task = task
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)
        self.predictor = nn.Linear(embed_dim, out_dim)

    def forward(self, data):
        """
        Output: prediction: (batch_size,) or (batch_size, out_dim)
                graph embedding: (batch_size, embed_dim)
                node embedding: (num_nodes, embed_dim)
        """
        h = F.relu(self.conv1(data.x, data.edge_index))
        node_embed = self.conv2(h, data.edge_index)
        graph_embed = global_max_pool(node_embed, data.batch)
        out = self.predictor(graph_embed)

        if self.task == "reg":
            return out.view(-1), graph_embed, node_embed
        elif self.task == "clf":
            return F.log_softmax(out, dim=1), graph_embed, node_embed

class NNConvModel(nn.Module):
    def __init__(self, num_features, hidden_dim, embed_dim, out_dim, task):
        super().__init__()
        self.task = task
        self.lin0 = Linear(num_features, hidden_dim)
        nn = Sequential(Linear(3, 128), ReLU(), Linear(128, hidden_dim * hidden_dim))
        self.conv = NNConv(hidden_dim, hidden_dim, nn, aggr='mean')
        self.gru = GRU(hidden_dim, hidden_dim)
        self.set2set = Set2Set(embed_dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(hidden_dim, embed_dim)
        self.lin2 = torch.nn.Linear(embed_dim * 2, embed_dim)
        self.lin3 = torch.nn.Linear(embed_dim, out_dim)

    def forward(self, data):
        """
        Output: prediction: (batch_size,) or (batch_size, out_dim)
                graph embedding: (batch_size, embed_dim)
                node embedding: (num_nodes, embed_dim)
        """
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        node_embed = self.lin1(out)
        set2set_out = self.set2set(node_embed, data.batch)
        graph_embed = self.lin2(set2set_out)
        out = self.lin3(graph_embed)

        if self.task == "reg":
            return out.view(-1), graph_embed, node_embed
        elif self.task == "clf":
            return F.log_softmax(out, dim=1), graph_embed, node_embed


class HighwayGateLayer(nn.Module):
    def __init__(self, in_out_size, bias=True):
        super(HighwayGateLayer, self).__init__()
        self.transform = nn.Linear(in_out_size, in_out_size, bias=bias)

    def forward(self, x, y):
        out_transform = torch.sigmoid(self.transform(x))
        return out_transform * x + (1 - out_transform) * y


class LateFusion(nn.Module):
    def __init__(self, chemberta_model, num_features, graph_model, hidden_dim, embed_dim, out_dim, task, aggr):
        super().__init__()
        self.task = task
        self.aggr = aggr
        self.bert_model = ChemBERTaForPropertyPrediction(chemberta_model, out_dim, task)
        if graph_model == "mpnn":
            self.mpnn_model = NNConvModel(num_features, hidden_dim, embed_dim, out_dim, task)
        elif graph_model == "gnn":
            self.mpnn_model = GCNModel(num_features, hidden_dim, embed_dim, out_dim, task)
        self.gate = HighwayGateLayer(embed_dim)
        self.concat2embed = nn.Linear(embed_dim * 2, embed_dim)
        self.predictor = nn.Linear(chemberta_model.config.hidden_size, out_dim)

    def forward(self, batch):
        """
        Output: prediction: (batch_size,) or (batch_size, out_dim)
        """
        _, bert_graph_embed, _ = self.bert_model(batch)
        _, mpnn_graph_embed, _ = self.mpnn_model(batch)

        if self.aggr == "sum":
            fusion_graph_embed = bert_graph_embed + mpnn_graph_embed
        elif self.aggr == "max":
            fusion_graph_embed = torch.maximum(bert_graph_embed, mpnn_graph_embed)
        elif self.aggr == "concat":
            fusion_graph_embed = self.concat2embed(torch.cat([bert_graph_embed, mpnn_graph_embed], dim=1))
        elif self.aggr == "gate":
            fusion_graph_embed = self.gate(bert_graph_embed, mpnn_graph_embed)

        if self.task == "reg":
            return self.predictor(fusion_graph_embed).view(-1)
        elif self.task == "clf":
            return F.log_softmax(self.predictor(fusion_graph_embed), dim=1)


class JointFusionMPNN2LM(nn.Module):
    def __init__(self, chemberta_model, num_features, graph_model, hidden_dim, embed_dim, out_dim, task, aggr):
        super().__init__()
        self.graph_model = graph_model
        self.task = task
        self.aggr = aggr
        self.chemberta = chemberta_model
        self.embed_dim = embed_dim
        self.word_embeddings = nn.Embedding(chemberta_model.config.vocab_size,
                                            chemberta_model.config.hidden_size,
                                            padding_idx=chemberta_model.config.pad_token_id)
        if graph_model == "mpnn":
            self.mpnn_model = NNConvModel(num_features, hidden_dim, embed_dim, out_dim, task)
        elif graph_model == "gnn":
            self.mpnn_model = GCNModel(num_features, hidden_dim, embed_dim, out_dim, task)
        self.concat2embed = nn.Linear(embed_dim * 2, embed_dim)
        self.predictor = nn.Linear(chemberta_model.config.hidden_size, out_dim)

    def forward(self, batch):
        """
        Output: prediction: (batch_size,) or (batch_size, out_dim)
        """
        input_ids, attention_mask, mol_mask = batch.input_ids, batch.attention_mask, batch.mol_mask
        bert_embeds = self.word_embeddings(input_ids)
        _, _, mpnn_embeds = self.mpnn_model(batch)

        bert_embeds = bert_embeds.reshape(-1, self.embed_dim)
        if self.aggr == "sum":
            bert_embeds[batch.mol_mask.view(-1)] += mpnn_embeds
        elif self.aggr == "max":
            max_embeds = torch.maximum(bert_embeds[batch.mol_mask.view(-1)], mpnn_embeds)
            bert_embeds[batch.mol_mask.view(-1)] = max_embeds
        elif self.aggr == "concat":
            concat_embeds = torch.cat([bert_embeds[batch.mol_mask.view(-1)], mpnn_embeds], dim=1)
            bert_embeds[batch.mol_mask.view(-1)] = self.concat2embed(concat_embeds)

        # fused_embeds = bert_embeds.reshape(batch.num_graphs, -1, self.node_embed_dim)

        fused_embeds = bert_embeds.reshape(batch.num_graphs, -1, self.embed_dim)

        outputs = self.chemberta(inputs_embeds=fused_embeds, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        if self.task == "reg":
            return self.predictor(hidden_states[:, 0]).view(-1)
        elif self.task == "clf":
            return F.log_softmax(self.predictor(hidden_states[:, 0]), dim=1)


class JointFusionLM2MPNN(nn.Module):
    def __init__(self, chemberta_model, num_features, graph_model, hidden_dim, embed_dim, out_dim, task, aggr):
        super().__init__()
        self.graph_model = graph_model
        self.task = task
        self.aggr = aggr
        self.embed_dim = embed_dim
        self.bert_model = ChemBERTaForPropertyPrediction(chemberta_model, out_dim, task)
        if graph_model == "mpnn":
            self.mpnn_model = NNConvModel(num_features, hidden_dim, embed_dim, out_dim, task)
        elif graph_model == "gnn":
            self.mpnn_model = GCNModel(num_features, hidden_dim, embed_dim, out_dim, task)
        self.embed2fea = nn.Linear(embed_dim, num_features)
        self.concat2fea = nn.Linear(num_features + embed_dim, num_features)
        self.predictor = nn.Linear(chemberta_model.config.hidden_size, out_dim)

    def forward(self, batch):
        """
        Output: prediction: (batch_size,) or (batch_size, out_dim)
        """
        _, _, bert_node_embed = self.bert_model(batch)
        batch.x = F.normalize(batch.x)
        bert_embeds = bert_node_embed.reshape(-1, self.embed_dim)[batch.mol_mask.view(-1)]
        if self.aggr == "sum":
            batch.x += self.embed2fea(bert_embeds)
        elif self.aggr == "max":
            batch.x = torch.maximum(batch.x, self.embed2fea(bert_embeds))
        elif self.aggr == "concat":
            batch.x = self.concat2fea(torch.cat([batch.x, bert_embeds], dim=1))
        pred, _, _ = self.mpnn_model(batch)
        return pred

