import numpy as np
import torch


def decode_output(user_graph,
                  prediction_edges,
                  model_output,
                  rule_encoding,
                  node_mapping):
    """
    Decode the output of the model to top-50 edges
    :param user_graph: input pytorch-geometric graph
    :param prediction_edges: list of links to be predicted
    :param model_output: raw output of the model, torch model (n_nodes * n_nodes, num_edges_categories)
    :param rule_encoding: encoding of possible rules
    :param node_mapping: dict mapping nodes to device_ids
    :return: list of tuple with top50 edges
    """
    idx2edges = {}
    edges2idx = {}
    for i, j in enumerate(prediction_edges):
        idx2edges[i] = j
        edges2idx[j] = i

    edges = user_graph.edge_index
    for ed_idx in range(len(edges[0])):
        model_output[edges2idx[(edges[0][ed_idx].item(), edges[1][ed_idx].item())],
        rule_encoding[(user_graph.edge_attr[ed_idx][0].item(), user_graph.edge_attr[ed_idx][1].item())]] = 0
    topk_test = get_topk_rules(model_output, idx2edges, k=50)
    rule_decoding = {v: k for k, v in rule_encoding.items()}
    return decode_nodes(topk_test, node_mapping, rule_decoding)


def get_topk_rules(model_output, idx2edges, k=50):
    """

    :param model_output: raw output of the model
    :param idx2edges:
    :param k: select top-k
    :return: top_list, list of tuple identifying top edges
    """
    v_topk, idx_topk = torch.topk(model_output.flatten(), k)
    top_list = [(idx2edges[i[0]], i[1], v_topk[idx]) for idx, i in
                enumerate(np.array(np.unravel_index(idx_topk.numpy(), model_output.shape)).T)]
    return top_list


def decode_nodes(top_results, node_mapping, rule_decoding):
    """
    Decode tor results from node index to device_ids
    :param top_results: top_list, list of tuple identifying top edges
    :param node_mapping: maps node ids to device_ids
    :param rule_decoding: maps encoded rule_types to rule_types
    :return: list of tuple (trigger_device, action_device, (trigger_state, action) )
    """
    results = []
    for top_item in top_results:
        nodeidx_dict = {v: k for k, v in node_mapping.items()}
        results.append((nodeidx_dict[top_item[0][0]], nodeidx_dict[top_item[0][1]], rule_decoding[top_item[1]],
                        top_item[2].detach().numpy()))
    return results
