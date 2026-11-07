import json

import torch
from torch.utils.data import Dataset


def build_offsets(path, limit=None):
    offsets = []
    with open(path, "rb") as f:
        position = f.tell()
        line = f.readline()
        while line:
            offsets.append(position)
            position = f.tell()
            line = f.readline()
            if limit is not None and len(offsets) >= limit:
                break
    return offsets


def read_at(path, offsets, index):
    with open(path, "rb") as f:
        f.seek(offsets[index])
        return json.loads(f.readline().decode("utf-8"))


class CipherPairs(Dataset):
    def __init__(self, root, split, max_source_len, max_target_len, limit=None):
        self.source_path = f"{root}/{split}/cipher.jsonl"
        self.target_path = f"{root}/{split}/cleaned.jsonl"
        self.source_offsets = build_offsets(self.source_path, limit)
        self.target_offsets = build_offsets(self.target_path, limit)
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return min(len(self.source_offsets), len(self.target_offsets))

    def __getitem__(self, index):
        source = read_at(self.source_path, self.source_offsets, index)["ciphertext"]
        target = read_at(self.target_path, self.target_offsets, index)["text"]
        return source, target


def make_collate(tokenizer, max_source_len, max_target_len):
    def collate(batch):
        inputs = []
        labels = []
        for source, target in batch:
            s = tokenizer.encode_source(source)[:max_source_len]
            t = tokenizer.encode_target(target)[:max_target_len]
            ids = s + [tokenizer.sep_id] + t + [tokenizer.eos_id]
            lab = [-100] * (len(s) + 1) + t + [tokenizer.eos_id]
            inputs.append(ids)
            labels.append(lab)
        width = max(len(x) for x in inputs)
        input_ids = torch.full((len(inputs), width), tokenizer.pad_id, dtype=torch.long)
        label_ids = torch.full((len(inputs), width), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(inputs), width), dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(inputs, labels)):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            label_ids[i, :len(lab)] = torch.tensor(lab)
            attention_mask[i, :len(ids)] = 1
        return input_ids, label_ids, attention_mask
    return collate
