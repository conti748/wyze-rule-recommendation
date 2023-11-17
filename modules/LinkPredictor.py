import torch
from torch_geometric.nn import SAGEConv
import torch.nn as nn
import numpy as np

class LinkPredictor(torch.nn.Module):
    def __init__(self, out_channels= 128, node_dim=16,
                 embedding_edge_trigger_dim=4,
                 embedding_edge_action_dim=4, num_edge_categories=552):
        super().__init__()
        self.embedding_edges_trigger = EdgeTypeEmbedding(embedding_edge_trigger_dim, num_edge_types= 45)
        self.embedding_edges_action = EdgeTypeEmbedding(embedding_edge_action_dim, num_edge_types= 47)
        self.conv1 = SAGEConv(node_dim + embedding_edge_trigger_dim + embedding_edge_action_dim,
                              2 * out_channels, add_self_loops=False)
        self.conv2 = SAGEConv(2 * out_channels, out_channels, add_self_loops=False)
        self.category_predictor = nn.Sequential(
                nn.Linear(2*out_channels, 128),  # Concatenate features of both connected nodes
                nn.ReLU(),
                nn.Linear(128, num_edge_categories)
                )

    def forward(self, x, edge_index, edge_index_gt, edge_feats):
        if len(edge_feats.shape) > 1:
          edge_feats_trigger = self.embedding_edges_trigger(edge_feats[:, 0]).squeeze()
          edge_feats_action = self.embedding_edges_action(edge_feats[:, 1]).squeeze()
        else:
          edge_feats_trigger = torch.zeros(len(edge_index[1, :]), 4)
          edge_feats_action = torch.zeros(len(edge_index[1, :]), 4)
        if len(edge_feats_trigger.shape) == 1:
          edge_feats_trigger = edge_feats_trigger.unsqueeze(0)
          edge_feats_action = edge_feats_action.unsqueeze(0)

        aggregated_edge_feats_trigger = aggregate_edge_features(edge_index, edge_feats_trigger, len(x))
        aggregated_edge_feats_action = aggregate_edge_features(edge_index, edge_feats_action, len(x), trigger = False)
        x = torch.cat([x, aggregated_edge_feats_trigger, aggregated_edge_feats_action], dim=1)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        edge_features = torch.cat( [x[edge_index_gt[0]], x[edge_index_gt[1]]], dim=1)
        category_logits = self.category_predictor(edge_features)
        return torch.sigmoid(category_logits)

def aggregate_edge_features(edge_index, edge_feats, num_nodes, trigger=True):
    index = edge_index[0] if trigger else edge_index[1]
    # Initialize aggregated edge features with zeros
    aggregated_edge_feats = torch.zeros(num_nodes, edge_feats.size(1))
    # Use index_add to aggregate edge features
    aggregated_edge_feats.index_add_(0, index, edge_feats)
    return aggregated_edge_feats


class EdgeTypeEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_edge_types = 552):
        super(EdgeTypeEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_edge_types, embedding_dim)
    def forward(self, edge_type_indices):
        return self.embedding(edge_type_indices)

class NodeTypeEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_node_types=16):
        super(NodeTypeEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_node_types, embedding_dim)
    def forward(self, node_type_indices):
        return self.embedding(node_type_indices)

if __name__ == '__main__':
    model = LinkPredictor()
    num_params_per_layer = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = torch.prod(torch.tensor(param.size())).item()
            num_params_per_layer[name] = num_params

    # Print the number of parameters per layer
    for name, num_params in num_params_per_layer.items():
        print(f'Layer: {name}, Number of Parameters: {num_params}')

    print(f"\nTotal number of parameters: {np.sum(np.array(list(num_params_per_layer.values())))}")

