import argparse
import json
import os
import time

import torch

from .config import load_config
from .data import CipherPairs
from .flops import count_parameters, estimate_flops
from .generate import decode_document
from .infer import load_model
from .metrics import char_accuracy, mean_edit_distance, macro_prf, word_accuracy


def generation_metrics(model, tokenizer, cfg, split, limit, all_lengths, device):
    model.eval()
    data = CipherPairs(cfg["data_root"], split)
    max_chunks = cfg.get("eval_max_chunks")
    deadline = time.time() + cfg.get("eval_max_minutes", 15) * 60
    preds = []
    refs = []
    stopped_early = False
    for i in range(len(data)):
        if len(preds) >= limit:
            break
        if time.time() > deadline:
            stopped_early = True
            break
        source, target = data[i]
        if not all_lengths and len(target) > cfg["max_target_len"]:
            continue
        preds.append(decode_document(model, tokenizer, cfg, source, device, max_chunks))
        refs.append(target)
    precision, recall, f1 = macro_prf(preds, refs)
    return {
        "split": split,
        "documents": len(preds),
        "short_only": not all_lengths,
        "stopped_early": stopped_early,
        "char_accuracy": char_accuracy(preds, refs),
        "word_accuracy": word_accuracy(preds, refs),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "edit_distance": mean_edit_distance(preds, refs),
        "parameters": count_parameters(model),
        "flops": estimate_flops(
            cfg, cfg["max_source_len"] + cfg["max_target_len"], tokenizer.vocab_size
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--all-lengths", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or f"models/part1/{cfg['run_name']}.pt"
    model, tokenizer, model_cfg = load_model(checkpoint, device)
    result = generation_metrics(
        model, tokenizer, model_cfg, args.split, args.limit, args.all_lengths, device
    )
    result["arch"] = cfg.get("arch")
    output = args.output or f"results/part1/{cfg['run_name']}_{args.split}.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
