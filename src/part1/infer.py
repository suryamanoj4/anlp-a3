import argparse
import json
import os

import torch

from . import env
from .config import load_config
from .data import read_at, build_offsets
from .generate import generate_ids
from .models.build import build_model
from .tokenizer import Tokenizer


def load_model(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    tokenizer = Tokenizer(payload["source_chars"], payload["target_chars"])
    model = build_model(cfg, tokenizer.vocab_size).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, tokenizer, cfg


def read_records(path, limit=None):
    if os.path.isdir(path):
        path = os.path.join(path, "cipher.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".jsonl"):
        records = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                records.append(json.loads(line))
        return [(r["id"], r["ciphertext"]) for r in records]
    offsets, _ = build_offsets(path, limit)
    out = []
    for i in range(len(offsets)):
        record = read_at(path, offsets, i)
        out.append((record["id"], record["ciphertext"]))
    return out


def resolve_checkpoint(cfg):
    local = f"models/part1/{cfg['run_name']}.pt"
    if cfg.get("hf_pull"):
        try:
            from .hf import pull_checkpoint

            return pull_checkpoint(
                f"{cfg['run_name']}.pt", env.hf_repo(cfg.get("hf_repo")), "models/part1"
            )
        except Exception as exc:
            if os.path.exists(local):
                print(json.dumps({"hf_pull_failed": str(exc), "fallback": local}))
                return local
            raise
    return local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or resolve_checkpoint(cfg)
    model, tokenizer, model_cfg = load_model(checkpoint, device)
    source = args.input or f"{cfg['data_root']}/{args.split}/cipher.jsonl"
    output = args.output or f"{cfg['out_dir']}/{args.split}_decrypt.jsonl"
    records = read_records(source, args.limit)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for record_id, ciphertext in records:
            ids = tokenizer.encode_source(ciphertext)[: model_cfg["max_source_len"]]
            out = generate_ids(
                model,
                ids,
                model_cfg["max_new_tokens"],
                tokenizer.eos_id,
                tokenizer.sep_id,
                device,
            )
            f.write(json.dumps({"id": record_id, "text": tokenizer.decode_target(out)}) + "\n")
    print(json.dumps({"records": len(records), "output": output, "checkpoint": checkpoint}))


if __name__ == "__main__":
    main()
