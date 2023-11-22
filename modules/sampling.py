import torch
from torch_geometric.data import Data
import random
import torch_geometric


def positive_sampling(data, batch_vector, rule_encoding):
    """

    :param data: Input batch of graphs
    :param batch_vector: batch mask
    :param rule_encoding: dictionary that maps rule to encoding
    :return: g_input graph with leave-one-out, g_gt graph with the single edge
    """
    unique_batches = torch.unique(batch_vector)
    selected_src = []
    selected_dst = []
    selected_f = []
    leaveout_src = []
    leaveout_dst = []
    leaveout_f = []

    for batch in unique_batches:
        # Filter nodes belonging to the current batch
        batch_mask = batch_vector == batch
        subgraph_pyg_edge, subgraph_pyg_edge_attr = torch_geometric.utils.subgraph(batch_mask, data.edge_index,
                                                                                   data.edge_attr)
        num_edges = subgraph_pyg_edge.shape[1]

        if num_edges == 0:
            raise ValueError("Graph in batch {} has no edges.".format(batch))
        elif num_edges == 1:
            src_nodes, dst_nodes = subgraph_pyg_edge[0], subgraph_pyg_edge[1]
            leaveout_f.append(subgraph_pyg_edge_attr[0])
            leaveout_src.append(src_nodes[0])
            leaveout_dst.append(dst_nodes[0])
        else:
            edge_indices = list(range(num_edges))
            random.shuffle(edge_indices)
            selected_indices = edge_indices[:-1]
            src_nodes, dst_nodes = subgraph_pyg_edge[0], subgraph_pyg_edge[1]
            selected_src.extend(src_nodes[selected_indices])
            selected_dst.extend(dst_nodes[selected_indices])
            selected_f.extend(subgraph_pyg_edge_attr[selected_indices])
            leaveout_src.append(src_nodes[edge_indices[-1]])
            leaveout_dst.append(dst_nodes[edge_indices[-1]])
            leaveout_f.append(subgraph_pyg_edge_attr[edge_indices[-1]])
    leaveout_f = [rule_encoding[(gt1.item(), gt2.item())] for gt1, gt2 in leaveout_f]
    edge_index = torch.tensor([selected_src,
                               selected_dst], dtype=torch.long)
    edge_index_gt = torch.tensor([leaveout_src,
                                  leaveout_dst], dtype=torch.long)
    if selected_f:
        selected_f = torch.stack(selected_f, dim=1)
    else:
        selected_f = torch.tensor([])
    leaveout_f = torch.tensor(leaveout_f)
    g_input = Data(x=data.x, edge_index=edge_index, edge_attr=selected_f.T)

    g_gt = Data(x=data.x, edge_index=edge_index_gt, edge_attr=leaveout_f.unsqueeze(1))

    return g_input, g_gt


def construct_negative_graph(graph, batch_vector, rule_encoding):
    """

    :param graph: input batch of graphs
    :param batch_vector: batch mapping
    :param rule_encoding: rule_encoding: dictionary that maps rule to encoding
    :return: neg_graph graph with negative edges
    """
    num_nodes_per_graph = torch.bincount(batch_vector)
    src, dst = graph.edge_index[0, :], graph.edge_index[1, :]
    edge_attr = graph.edge_attr
    # Initialize lists to store negative samples
    neg_src_list = []
    neg_dst_list = []
    neg_edge_attr = []

    list_edge_attr = [rule_encoding[(edge_attr_item[0], edge_attr_item[1])] for edge_attr_item in edge_attr.tolist()]
    # Create a set of edges that are present in the original graph
    original_edges_set = set(zip(src.tolist(), dst.tolist(), list_edge_attr))
    # Iterate over each graph in the batch
    for graph_idx in range(len(num_nodes_per_graph)):
        # Extract nodes for the current graph
        start_idx = torch.sum(num_nodes_per_graph[:graph_idx])
        end_idx = start_idx + num_nodes_per_graph[graph_idx]

        neg_src = torch.randint(start_idx, end_idx, (1,))
        neg_dst = torch.randint(start_idx, end_idx, (1,))
        neg_attr = torch.randint(0, len(rule_encoding), (1,))

        # Ensure that the sampled edge is not already in the original graph
        while (neg_src.item(), neg_dst.item(), neg_attr.item()) in original_edges_set:
            neg_src = torch.randint(start_idx, end_idx, (1,))
            neg_dst = torch.randint(start_idx, end_idx, (1,))
            neg_attr = torch.randint(0, len(rule_encoding), (1,))

        # Append negative samples to the lists
        neg_src_list.append(neg_src)
        neg_dst_list.append(neg_dst)
        neg_edge_attr.append(neg_attr)

    edge_index_neg = torch.tensor([neg_src_list,
                                   neg_dst_list], dtype=torch.long)
    edge_attr_neg = torch.tensor(neg_edge_attr, dtype=torch.long).unsqueeze(1)
    neg_graph = Data(x=graph.x, edge_index=edge_index_neg, edge_attr=edge_attr_neg)

    return neg_graph
