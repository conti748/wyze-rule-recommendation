import torch
from modules.data_pipeline.RulesDataset import RuleDataset
import yaml


def get_config(file_path: str) -> dict:
    """"
    Read the config file
    """
    with open(file_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, data, graph_dict):
        self.data = data
        self.graph_dict = graph_dict
        self.userid2idx = { j:k for j, k in enumerate(data.keys())}
        self.idx2userid = { k:j for j, k in enumerate(data.keys())}
        self.data = {self.idx2userid[k]:v for k, v in data.items() }
        self.graph_dict = { self.idx2userid[k]:v for k, v in graph_dict.items() }
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.graph_dict[idx]



if __name__ == '__main__':
    config = get_config("./cfg/training.yaml")
    rule_dataset = RuleDataset()
    rule_train, rule_val, rule_test = rule_dataset.split_dataset(rule_dataset.dataset,
                               config['training']['split']['train'],
                               config['training']['split']['val'])
    graph_dict_train = rule_dataset.create_graph_dataset(rule_train)
    graph_dict_val= rule_dataset.create_graph_dataset(rule_val)
    graph_dict_test = rule_dataset.create_graph_dataset(rule_test)
    graph_generator = GraphDataset(rule_train, graph_dict_train)
    graph_generator_val = GraphDataset(rule_val, graph_dict_val)
    graph_generator_test = GraphDataset(rule_test, graph_dict_test)