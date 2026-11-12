from collections import defaultdict

import torch
from torch_geometric.data import Batch
from torch_geometric.datasets import UPFD


def load_split(root, name, feature, split):
    return UPFD(root=root, name=name, feature=feature, split=split)


@torch.no_grad()
def feature_stats(dataset):
    total = None
    total_sq = None
    count = 0
    for data in dataset:
        x = data.x
        if total is None:
            total = torch.zeros(x.size(1), dtype=torch.float64)
            total_sq = torch.zeros(x.size(1), dtype=torch.float64)
        total += x.sum(0).double()
        total_sq += (x * x).sum(0).double()
        count += x.size(0)
    mean = total / max(count, 1)
    variance = total_sq / max(count, 1) - mean * mean
    std = variance.clamp_min(0.0).sqrt()
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean.float(), std.float()


def edge_profile(dataset, sample=64):
    reverse = 0
    total = 0
    incoming_root = 0
    for i in range(min(sample, len(dataset))):
        edge_index = dataset[i].edge_index
        if edge_index.numel() == 0:
            continue
        pairs = set(map(tuple, edge_index.t().tolist()))
        reverse += sum(1 for u, v in pairs if (v, u) in pairs)
        total += len(pairs)
        incoming_root += int((edge_index[1] == 0).sum())
    return {
        "edges_sampled": total,
        "reverse_present_fraction": reverse / max(total, 1),
        "incoming_to_root": incoming_root,
    }


def infer_orientation(dataset, sample=256):
    incoming = 0
    for i in range(min(sample, len(dataset))):
        edge_index = dataset[i].edge_index
        if edge_index.numel():
            incoming += int((edge_index[1] == 0).sum())
    return "child_to_parent" if incoming > 0 else "parent_to_child"


def num_classes(dataset):
    highest = 0
    for data in dataset:
        highest = max(highest, int(data.y.max()))
    return highest + 1


def make_collate(mean, std, flip):
    def collate(graphs):
        batch = Batch.from_data_list(graphs)
        if mean is not None:
            batch.x = (batch.x - mean) / std
        if flip:
            batch.edge_index = batch.edge_index.flip(0)
        return batch

    return collate


class FeatureStore:
    def __init__(self, root, name="gossipcop", normalize=True):
        self.root = root
        self.name = name
        self.normalize = normalize
        self.cache = {}

    def get(self, feature):
        if feature not in self.cache:
            splits = {
                split: load_split(self.root, self.name, feature, split)
                for split in ("train", "val", "test")
            }
            if self.normalize:
                stats = feature_stats(splits["train"])
            else:
                stats = (None, None)
            orientation = infer_orientation(splits["train"])
            self.cache[feature] = {
                "splits": splits,
                "stats": stats,
                "orientation": orientation,
                "flip": orientation == "child_to_parent",
                "classes": num_classes(splits["train"]),
            }
        return self.cache[feature]

    def release(self, feature):
        self.cache.pop(feature, None)
