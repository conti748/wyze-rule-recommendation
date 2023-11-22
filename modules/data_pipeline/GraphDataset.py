import torch


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, data, graph_dict):
        self.data = data
        self.graph_dict = graph_dict
        self.userid2idx = {j: k for j, k in enumerate(data.keys())}
        self.idx2userid = {k: j for j, k in enumerate(data.keys())}
        self.data = {self.idx2userid[k]: v for k, v in data.items()}
        self.graph_dict = {self.idx2userid[k]: v for k, v in graph_dict.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.graph_dict[idx]
