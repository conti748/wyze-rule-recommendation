import torch
from torch_geometric.nn import SAGEConv
import torch.nn as nn

class LinkPredictor(torch.nn.Module):
    def __init__(self, node_dim=16,
                 num_trigger_states=45, num_actions=47,
                 embedding_edge_trigger_dim=4,
                 embedding_edge_action_dim=4,
                 num_edge_categories=552,
                 hidden_dim=128, hidden_linear=128):
        """

        :param hidden_dim: number of neurons for SAGEConv layers
        :param node_dim: number of device types
        :param embedding_edge_trigger_dim: embedding dimension for trigger_states
        :param embedding_edge_action_dim: embedding dimension for actions
        :param num_edge_categories: cardinality of rule-set considered
        """
        super().__init__()
        self.embedding_edges_trigger = EdgeTypeEmbedding(embedding_edge_trigger_dim, num_edge_types=num_trigger_states)
        self.embedding_edges_action = EdgeTypeEmbedding(embedding_edge_action_dim, num_edge_types=num_actions)
        self.conv1 = SAGEConv(node_dim + embedding_edge_trigger_dim + embedding_edge_action_dim,
                              2 * hidden_dim, add_self_loops=False)
        self.conv2 = SAGEConv(2 * hidden_dim, hidden_dim, add_self_loops=False)
        self.category_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_linear),  # Concatenate features of both connected nodes
            nn.ReLU(),
            nn.Linear(hidden_linear, num_edge_categories)
        )

    def forward(self, x, edge_index, edge_index_eval, edge_feats):
        """

        :param x: Node features vector
        :param edge_index: Input graph edges
        :param edge_index_eval: a new graph with same nodes of x, but with edges to be predicted
        :param edge_feats: edge features
        :return: category_logits encoding prediction for each edge (num_edge_categories for each edge, one for rule type)
        """
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
        aggregated_edge_feats_action = aggregate_edge_features(edge_index, edge_feats_action, len(x), trigger=False)
        x = torch.cat([x, aggregated_edge_feats_trigger, aggregated_edge_feats_action], dim=1)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        edge_features = torch.cat([x[edge_index_eval[0]], x[edge_index_eval[1]]], dim=1)
        category_logits = self.category_predictor(edge_features)
        return torch.sigmoid(category_logits)


def aggregate_edge_features(edge_index, edge_feats, num_nodes, trigger=True):
    """
    Aggregate edge features to node
    Trigger state are aggregated by source node,
    Actions are aggregated by destination node
    """
    index = edge_index[0] if trigger else edge_index[1]
    # Initialize aggregated edge features with zeros
    aggregated_edge_feats = torch.zeros(num_nodes, edge_feats.size(1))
    # Use index_add to aggregate edge features
    aggregated_edge_feats.index_add_(0, index, edge_feats)
    return aggregated_edge_feats


class EdgeTypeEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_edge_types=552):
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
