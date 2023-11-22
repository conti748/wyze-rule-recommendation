import argparse
import itertools
import torch
from tqdm import tqdm
import csv
from torch_geometric.data import Data
from modules.LinkPredictor import LinkPredictor
from modules.data_pipeline.GraphDataset import GraphDataset
from modules.data_pipeline.RulesDataset import RulesDataset
from modules.utils.decode_results import decode_output


def dump_results(results, file_name):
    """
    Dumps results in an output csv
    :param results: results for each user (key) containing list of tuple (trigger_device, action_device, (trigger_state, action) ) (value)
    :param file_name: name of the output csv
    """
    user_ids = []
    rules = []
    ranks = []
    for user_id, res in tqdm(results.items()):
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

    parser = argparse.ArgumentParser(description='Script to test the model on public/private')
    parser.add_argument('--mode', choices=['public', 'private'], default='public', help='Specify the mode (public or '
                                                                                     'private)')
    parser.add_argument('--model_path', type=str, default="./models/best_model.ckpt",
                        help='Specify the model name as a string')
    # Parse the command-line arguments
    args = parser.parse_args()
    mode = args.mode
    model_path = args.model_path
    model_name = model_path.split('/')[-1].split('.')[0]

    # Your script logic based on the selected mode
    if mode == 'public':
        print('Public mode selected')
        test_set_private = False
        output_filename = f"./results/results_public_{model_name}.csv"
    elif mode == 'private':
        print('Private mode selected')
        test_set_private = True
        output_filename = f"./results/results_private_{model_name}.csv"

    # Load the best model
    print(f'Loading model {model_path}')
    model = LinkPredictor()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    print('Model loaded')
    # Load test-set
    rule_dataset = RulesDataset(mode='test', test_set_private=test_set_private)
    graph_dict_test = rule_dataset.create_graph_dataset(rule_dataset.dataset)
    graph_generator_test = GraphDataset(rule_dataset.dataset, graph_dict_test)
    print(f"Number of users: {len(graph_dict_test)}")

    results_dict = {}

    for idx in tqdm(range(len(graph_generator_test))):
        data_test = graph_generator_test.__getitem__(idx)
        num_nodes = len(data_test.x)

        # Evaluate the graph with all the possible edges
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
                                                                           rule_dataset.node_mapping[
                                                                               graph_generator_test.userid2idx[idx]])
    dump_results(results_dict, output_filename)
