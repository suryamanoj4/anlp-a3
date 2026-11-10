import json
import math
from bisect import bisect_right

import torch
from torch.utils.data import Dataset, Sampler

from .tokenizer import strip_source


def build_offsets(path, limit=None):
    offsets = []
    lengths = []
    with open(path, "rb") as f:
        position = f.tell()
        line = f.readline()
        while line and (limit is None or len(offsets) < limit):
            offsets.append(position)
            lengths.append(len(line))
            position = f.tell()
            line = f.readline()
    return offsets, lengths


def limit_for_tokens(path, max_tokens):
    if max_tokens is None:
        return None
    total = 0
    count = 0
    with open(path, "rb") as f:
        for line in f:
            if total >= max_tokens:
                break
            total += len(line)
            count += 1
    return count


def read_at(path, offsets, index):
    with open(path, "rb") as f:
        f.seek(offsets[index])
        return json.loads(f.readline().decode("utf-8"))


class CipherPairs(Dataset):
    def __init__(self, root, split, max_tokens=None, max_docs=None):
        self.source_path = f"{root}/{split}/cipher.jsonl"
        self.target_path = f"{root}/{split}/cleaned.jsonl"
        limit = limit_for_tokens(self.target_path, max_tokens)
        if max_docs is not None:
            limit = max_docs if limit is None else min(limit, max_docs)
        self.source_offsets, self.source_lengths = build_offsets(self.source_path, limit)
        self.target_offsets, _ = build_offsets(self.target_path, limit)

    def __len__(self):
        return min(len(self.source_offsets), len(self.target_offsets))

    def __getitem__(self, index):
        source = read_at(self.source_path, self.source_offsets, index)["ciphertext"]
        target = read_at(self.target_path, self.target_offsets, index)["text"]
        return source, target


class PatchedPairs(Dataset):
    def __init__(self, root, split, patch_size, max_docs=None):
        self.source_path = f"{root}/{split}/cipher.jsonl"
        self.target_path = f"{root}/{split}/cleaned.jsonl"
        self.patch_size = patch_size
        self.source_offsets, _ = build_offsets(self.source_path, max_docs)
        self.target_offsets, _ = build_offsets(self.target_path, max_docs)
        self.cumulative = []
        total = 0
        with open(self.target_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_docs is not None and i >= max_docs:
                    break
                length = len(json.loads(line)["text"])
                total += (length + patch_size - 1) // patch_size
                self.cumulative.append(total)

    def __len__(self):
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, index):
        doc = bisect_right(self.cumulative, index)
        start = (index - (self.cumulative[doc - 1] if doc else 0)) * self.patch_size
        source = strip_source(
            read_at(self.source_path, self.source_offsets, doc)["ciphertext"]
        )
        target = read_at(self.target_path, self.target_offsets, doc)["text"]
        width = min(len(source), len(target))
        return (
            source[start:start + self.patch_size],
            target[start:start + self.patch_size],
        )


class LengthBucketSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True, seed=0):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches = [
            order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)
        ]
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in permutation]
        for batch in batches:
            yield batch


class Collate:
    def __init__(self, tokenizer, max_source_len, max_target_len):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __call__(self, batch):
        tokenizer = self.tokenizer
        inputs = []
        labels = []
        for source, target in batch:
            s = tokenizer.encode_source(source)[: self.max_source_len]
            t = tokenizer.encode_target(target)[: self.max_target_len]
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


def make_collate(tokenizer, max_source_len, max_target_len):
    return Collate(tokenizer, max_source_len, max_target_len)
