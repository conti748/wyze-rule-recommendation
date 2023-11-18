import itertools
import torch
from modules.LinkPredictor import LinkPredictor
from torch_geometric.data import Data
from modules.data_pipeline.GraphDataset import GraphDataset
from modules.data_pipeline.RulesDataset import RulesDataset
from tqdm import tqdm
import numpy as np
import csv

def decode_output(user_graph,
                  all_possible_edges,
                  model_output,
                  rule_encoding,
                  node_mapping):
    idx2edges = {}
    edges2idx = {}
    for i, j in enumerate(all_possible_edges):
        idx2edges[i] = j
        edges2idx[j] = i

    edges = user_graph.edge_index
    for ed_idx in range(len(edges[0])):
        model_output[edges2idx[(edges[0][ed_idx].item(), edges[1][ed_idx].item())],
        rule_encoding[(user_graph.edge_attr[ed_idx][0].item(), user_graph.edge_attr[ed_idx][1].item())]] = 0
    top50_test = get_topk_rules(model_output, idx2edges, k=50)
    rule_onehot_decode = {v: k for k, v in rule_encoding.items()}
    return decode_nodes(top50_test, node_mapping, rule_onehot_decode)


def get_topk_rules(out, idx2edges, k=50):
    v_topk, idx_topk = torch.topk(out.flatten(), k)
    top_list = [ (idx2edges[i[0]], i[1], v_topk[idx]) for idx, i in enumerate(np.array(np.unravel_index(idx_topk.numpy(), out.shape)).T)]
    return top_list

def decode_nodes(top_results, node_mapping, rule_onehot_decode):
  results = []
  for top_item in top_results:
    nodeidx_dict = { v:k for k, v in node_mapping.items() }
    results.append((nodeidx_dict[top_item[0][0]], nodeidx_dict[top_item[0][1]], rule_onehot_decode[top_item[1]], top_item[2].detach().numpy()) )
  return results

def dump_results(results_dict, file_name = "./results/results_private.csv"):
    user_ids = []
    rules = []
    ranks = []
    for user_id, res in tqdm(results_dict.items()):
        for rank, r in enumerate(res):
            user_ids.append(user_id)
            rules.append(f"{r[0]}_{r[2][0]}_{r[2][1]}_{r[1]}")
            ranks.append(rank + 1)

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user_id', 'rule', 'rank'])
        for user_id, score, rank in zip(user_ids, rules, ranks):
            writer.writerow([user_id, score, rank])


if __name__ == '__main__':
    # Load the best model
    print('Loading model')
    model = LinkPredictor()
    checkpoint = torch.load("./models/best_model.ckpt")
    model.load_state_dict(checkpoint)
    print('Model loaded')

    # Load test-set
    rule_dataset = RulesDataset(mode='test', test_set_private=True)
    graph_dict_test = rule_dataset.create_graph_dataset(rule_dataset.dataset)
    graph_generator_test = GraphDataset(rule_dataset.dataset, graph_dict_test)
    print(f"Number of users: {len(graph_dict_test)}")

    results_dict = {}

    for idx in tqdm(range(len(graph_generator_test))):

        data_test = graph_generator_test.__getitem__(idx)
        num_nodes = len(data_test.x)

        # Evaluate the graph with all the possibile edges
        all_possible_edges = list(itertools.product(range(num_nodes), repeat=2))
        src = [i[0] for i in all_possible_edges]
        dst = [i[1] for i in all_possible_edges]
        edge_index_comb = torch.tensor([src,
                                        dst], dtype=torch.long)
        g_combination = Data(x=data_test.x, edge_index=edge_index_comb)

        # apply the model
        out = model(data_test.x, data_test.edge_index, g_combination.edge_index,
                    data_test.edge_attr)

        # decode the output
        results_dict[graph_generator_test.userid2idx[idx]] = decode_output(data_test, all_possible_edges,
                      out, rule_dataset.rule_encoding,
                      rule_dataset.node_mapping[graph_generator_test.userid2idx[idx]])
    dump_results(results_dict)