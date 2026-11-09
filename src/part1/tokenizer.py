import json


class Tokenizer:
    def __init__(self, source_chars, target_chars):
        self.source_chars = list(source_chars)
        self.target_chars = list(target_chars)
        self.pad_id = 0
        self.sep_id = 1
        self.eos_id = 2
        self.src_offset = 3
        self.src_stoi = {c: i + self.src_offset for i, c in enumerate(source_chars)}
        self.tgt_offset = self.src_offset + len(source_chars)
        self.tgt_stoi = {c: i + self.tgt_offset for i, c in enumerate(target_chars)}
        self.src_itos = {i: c for c, i in self.src_stoi.items()}
        self.tgt_itos = {i: c for c, i in self.tgt_stoi.items()}
        self.vocab_size = self.tgt_offset + len(target_chars)

    def encode_source(self, text):
        default = self.src_offset
        return [self.src_stoi.get(c, default) for c in text]

    def encode_target(self, text):
        default = self.tgt_offset
        return [self.tgt_stoi.get(c, default) for c in text]

    def decode_target(self, ids):
        out = []
        for i in ids:
            if i == self.eos_id:
                break
            c = self.tgt_itos.get(i)
            if c is not None:
                out.append(c)
        return "".join(out)

    @staticmethod
    def build(source_path, target_path, limit=2000):
        source_chars = set()
        target_chars = set()
        with open(source_path, encoding="utf-8") as fs, open(target_path, encoding="utf-8") as ft:
            for i, (ls, lt) in enumerate(zip(fs, ft)):
                if i >= limit:
                    break
                source_chars.update(json.loads(ls)["ciphertext"])
                target_chars.update(json.loads(lt)["text"])
        return Tokenizer(sorted(source_chars), sorted(target_chars))
