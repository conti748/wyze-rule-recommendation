import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data
import json


class RulesDataset:
    def __init__(self, cfg=None, mode='train',
                 test_set_private=False):
        if cfg is None:
            cfg = {}
        self.mode = mode
        self.test_set_private = test_set_private
        self.cfg = cfg
        # load the dataset splits
        self._get_dataset()
        self.prepare_graph_dataset()

    def _get_dataset(self):
        """Get the device and rule pd.DataFrame"""
        print("Start loading data...")
        if self.mode == 'train':
            self.device_dataset = load_dataset("wyzelabs/RuleRecommendation",
                                               data_files=f"{self.mode}_device.csv",
                                               cache_dir=r"./data/")
            self.rule_dataset = load_dataset("wyzelabs/RuleRecommendation",
                                             data_files=f"{self.mode}_rule.csv",
                                             cache_dir=r"./data/")
        elif self.mode == 'test':
            if not self.test_set_private:
                self.device_dataset = load_dataset("wyzelabs/RuleRecommendation",
                                                   data_files=f"{self.mode}_device.csv",
                                                   cache_dir=r"./data/")
                self.rule_dataset = load_dataset("wyzelabs/RuleRecommendation",
                                                 data_files=f"{self.mode}_rule.csv",
                                                 cache_dir=r"./data/")
            else:
                self.device_dataset = load_dataset("wyzelabs/RuleRecommendation",
                                                   data_files=f"{self.mode}_device_private.csv",
                                                   cache_dir=r"./data/")
                self.rule_dataset = load_dataset("wyzelabs/RuleRecommendation",
                                                 data_files=f"{self.mode}_rule_private.csv",
                                                 cache_dir=r"./data/")

        print("Data loaded")
        self.df_device = self.device_dataset['train'].to_pandas()
        self.df_rule = self.rule_dataset['train'].to_pandas()
        print(f"Number of rules: {len(self.df_rule)}")
        print(f"Number of devices: {len(self.df_device)}")

    def _prepare_ruletype(self):
        """
        Insert a column with the rule type encoded as a string with triggerstate_action
        """
        self.df_rule['rule_type'] = self.df_rule['rule'].apply(lambda x: x.split('_')[1:3])

    def _get_rule_encoding(self):
        """
        Get the encoding for rules.
        This function filters out the rule_type with less than self.cfg['dataset']['rule_count_frequency'] occurrences.
        """
        if self.mode == 'train':
            value_counts_rule = self.df_rule['rule_type'].value_counts()
            print(f"Filtering rules with fewer than {self.cfg['dataset']['rule_count_frequency']} occurrences")
            value_counts_rule = value_counts_rule[value_counts_rule >= self.cfg['dataset']['rule_count_frequency']]
            self.rule_encoding = {}
            counter = 0
            for k, v in value_counts_rule.items():
                self.rule_encoding[(int(k[0]), int(k[1]))] = counter
                counter += 1
            with open('./cfg/rule_encoding.json', 'w') as fp:
                rule_encoding_dump = {str(k):v for k, v in self.rule_encoding.items()}
                json.dump(rule_encoding_dump, fp, indent=2)
        else:
            with open('./cfg/rule_encoding.json', 'r') as fp:
                rule_encoding_dump = json.load(fp)
                self.rule_encoding = {eval(k): v for k, v in rule_encoding_dump.items()}

    def _get_device_mapping(self):
        """
        Get a dictionary to map device to a numerical encoding
        """
        if self.mode == 'train':
            self.device_model2id = {dev: i for i, dev in enumerate(self.df_device.device_model.unique())}
            self.device_model2id['Cloud'] = 15
            with open('./cfg/device_encoding.json', 'w') as fp:
                json.dump(self.device_model2id, fp, indent=2)
        else:
            with open('./cfg/device_encoding.json', 'r') as fp:
                self.device_model2id = json.load(fp)

    def _get_onehot_device_mapping(self):
        """
        Define a dictionary mapping from device to onehot encoding
        """
        num_values = len(self.device_model2id)
        self.device_onehot_mapping = {}
        for k, v in self.device_model2id.items():
            one_hot_encoded = np.zeros(num_values)
            one_hot_encoded[v] = 1
            self.device_onehot_mapping[v] = one_hot_encoded
        self.onehot_device_mapping = {tuple(v): k for k, v in self.device_onehot_mapping.items()}

    def _create_device_dict(self, group):
        """
        Define a function to aggregate the devices for each user
        """
        device_dict = dict(zip(group['device_id'],
                               group['device_model'].apply(lambda x: self.device_model2id[x])))
        device_dict[1797174] = self.device_model2id['Cloud']
        return {'devices_dict': dict(sorted(device_dict.items(), key=lambda item: item[1]))}

    def _get_user_dicts(self):
        """
        Aggregate rules per user and collect info to build graphs
        :return: user_device_dicts, a dict containing variables to each user used to build the graph
        """
        print("Aggregating devices by user")
        user_device_dicts = self.df_device.groupby('user_id').apply(self._create_device_dict).to_dict()
        # Aggregate rules for each user in user_device_dicts
        print("Aggregating rules by user")
        for index, rule_item in tqdm(self.df_rule.iterrows()):
            rule_item = rule_item.to_dict()
            if (rule_item['trigger_state_id'], rule_item['action_id']) in self.rule_encoding.keys():
                if not user_device_dicts[rule_item['user_id']].get('rules', None):
                        user_device_dicts[rule_item['user_id']]['rules'] = {}
                user_device_dicts[rule_item['user_id']]['rules'][index] = {
                        'trigger_device_id': rule_item['trigger_device_id'],
                        'trigger_state_id': rule_item['trigger_state_id'],
                        'action_id': rule_item['action_id'],
                        'action_device_id': rule_item['action_device_id']}
        # create a mapping device to node-id
        self.node_mapping = {}
        for user_id, v in user_device_dicts.items():
            self.node_mapping[user_id] = {id: i for i, id in enumerate(list(v['devices_dict']))}
        return user_device_dicts
    def _create_dataset_dict(self, user_dicts):
        """
        Starting from info about users, evaluate a dictionary with nodes and edges
        :param user_dicts: dictionary with info about each user
        """
        # Create the dataset identifying nodes and edges for each user
        self.dataset = {}
        error_list = []
        for user_id in tqdm(user_dicts.keys()):
            edges = []
            edges_gt = []
            senders = []
            receivers = []
            try:
                for k, v in user_dicts[user_id]['rules'].items():
                    nodes = [[self.device_onehot_mapping[n]] for n in user_dicts[user_id]['devices_dict'].values()]
                    nodes_tuple = tuple([n for n in user_dicts[user_id]['devices_dict'].values()])
                    senders.append(self.node_mapping[user_id][v['trigger_device_id']])
                    receivers.append(self.node_mapping[user_id][v['action_device_id']])
                    edges_gt.append([self.rule_encoding[(v['trigger_state_id'], v['action_id'])]])
                    edges.append((v['trigger_state_id'], v['action_id']))
                data_dict = {'globals': 0,
                             'n_node': len(nodes),
                             'n_edge': len(edges),
                             'nodes': nodes,
                             'nodes_tuple': nodes_tuple,
                             'edges': edges,
                             'edges_gt': edges_gt,
                             'senders': senders,
                             'receivers': receivers}
                self.dataset[user_id] = data_dict
            except Exception as e:
                # Filter out some users because user/device has inconsistencies
                error_list.append(user_id)
        print(f"Error for {len(error_list)} users, skipping.")

    def create_graph_dataset(self, dataset: dict) -> dict:
        """
        builds the graph dataset with torch_geometric.data.Data
        :param dataset: input dataset with user graphs
        :return: a dict containing graphs as pytorch-geometric Data
        """
        graph_dict_train = {}
        for idx in tqdm(dataset.keys()):
            senders = [dataset[idx]['senders'][s] for s in range(0, len(dataset[idx]['senders']))]
            receivers = [dataset[idx]['receivers'][s] for s in range(0, len(dataset[idx]['receivers']))]
            edges = [dataset[idx]['edges'][s] for s in range(0, len(dataset[idx]['edges']))]
            edge_index = torch.tensor([senders,
                                       receivers], dtype=torch.long)
            x = torch.tensor(dataset[idx]['nodes'], dtype=torch.long).squeeze()
            edge_attr = torch.tensor(edges, dtype=torch.long)
            graph_dict_train[idx] = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return graph_dict_train
    def _filter_single_rule(self):
        """
        Filter users with less than 2 edges.
        """
        # Filter out users with just one rule
        print("Total users before filtering:", len(self.dataset))
        self.dataset = {key: value for key, value in self.dataset.items() if len(self.dataset[key]['senders']) >= 2}
        print("Total users after filtering:", len(self.dataset))

    @staticmethod
    def split_dataset(graph_dataset, train_ratio, val_ratio):
        """
        split the data in train/val/test
        :param graph_dataset: graph dataset
        :param train_ratio: split ratio for training
        :param val_ratio: split ratio for validation
        :return: train_dict, val_dict, test_dict
        """
        keys = list(graph_dataset.keys())
        train_split = int(len(keys) * train_ratio)
        val_split = int(len(keys) * val_ratio)
        train_keys = keys[:train_split]
        val_keys = keys[train_split:(train_split + val_split)]
        test_keys = keys[(train_split + val_split):]
        train_dict = {key: graph_dataset[key] for key in train_keys}
        val_dict = {key: graph_dataset[key] for key in val_keys}
        test_dict = {key: graph_dataset[key] for key in test_keys}
        return train_dict, val_dict, test_dict
    def prepare_graph_dataset(self):
        """
        Staring from the device/rule data, build the graph dataset
        """
        self._prepare_ruletype()
        self._get_rule_encoding()
        self._get_device_mapping()
        self._get_onehot_device_mapping()
        user_dicts = self._get_user_dicts()
        self._create_dataset_dict(user_dicts)
        if self.mode == 'train':
            self._filter_single_rule()
