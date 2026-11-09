import argparse
import contextlib
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import env
from .config import load_config
from .data import CipherPairs, LengthBucketSampler, make_collate
from .flops import count_parameters, estimate_flops
from .generate import generate_ids
from .metrics import char_accuracy, macro_prf, mean_edit_distance, word_accuracy
from .models.build import build_model
from .tokenizer import Tokenizer


def compute_loss(model, input_ids, label_ids, attention_mask):
    logits = model(input_ids, attention_mask)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = label_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def lr_factor(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model, dataset, tokenizer, cfg, device):
    model.eval()
    preds = []
    refs = []
    for i in range(min(cfg["gen_samples"], len(dataset))):
        source, target = dataset[i]
        ids = tokenizer.encode_source(source)[: cfg["max_source_len"]]
        out = generate_ids(
            model,
            ids,
            cfg["max_new_tokens"],
            tokenizer.eos_id,
            tokenizer.sep_id,
            device,
        )
        preds.append(tokenizer.decode_target(out))
        refs.append(target)
    precision, recall, f1 = macro_prf(preds, refs)
    return {
        "char_accuracy": char_accuracy(preds, refs),
        "word_accuracy": word_accuracy(preds, refs),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "edit_distance": mean_edit_distance(preds, refs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg["max_steps"] = min(int(cfg["max_steps"]), 20)
        cfg["eval_interval"] = min(int(cfg["eval_interval"]), 10)
        cfg["gen_samples"] = min(int(cfg["gen_samples"]), 5)
        cfg["max_minutes"] = min(float(cfg["max_minutes"]), 2.0)
        cfg["wandb"] = False
        cfg["log_file"] = cfg["log_file"].replace(".jsonl", "_smoke.jsonl")
        cfg["run_name"] = f"{cfg['run_name']}_smoke"
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = cfg["data_root"]
    tokenizer = Tokenizer.build(
        f"{root}/{cfg['train_split']}/cipher.jsonl",
        f"{root}/{cfg['train_split']}/cleaned.jsonl",
        cfg["vocab_sample"],
    )
    train_set = CipherPairs(
        root, cfg["train_split"], cfg.get("max_tokens"), cfg.get("max_docs")
    )
    val_set = CipherPairs(
        root, cfg["val_split"], cfg.get("val_max_tokens"), cfg.get("val_max_docs")
    )
    collate = make_collate(tokenizer, cfg["max_source_len"], cfg["max_target_len"])
    sampler = LengthBucketSampler(
        train_set.source_lengths, cfg["batch_size"], shuffle=True, seed=cfg["seed"]
    )
    loader = DataLoader(
        train_set, batch_sampler=sampler, collate_fn=collate, num_workers=cfg.get("workers", 2)
    )

    model = build_model(cfg, tokenizer.vocab_size).to(device)
    params = count_parameters(model)
    flops = estimate_flops(cfg, cfg["max_source_len"] + cfg["max_target_len"], tokenizer.vocab_size)
    print(json.dumps({"arch": cfg.get("arch"), "parameters": params, "flops": flops}))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, cfg["warmup_steps"], cfg["max_steps"])
    )
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}.get(
        cfg.get("precision", "fp16")
    )
    use_amp = dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and dtype == torch.float16))

    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["log_file"]), exist_ok=True)
    os.makedirs("models/part1", exist_ok=True)

    run = None
    if cfg.get("wandb"):
        import wandb

        env.apply_wandb_key()
        run = wandb.init(
            project=env.wandb_project(cfg.get("wandb_project", "anlp-a3")),
            entity=env.wandb_entity(),
            name=cfg["run_name"],
            config=cfg,
        )

    step = 0
    best = -1.0
    tokens_seen = 0
    started = time.time()
    deadline = started + cfg["max_minutes"] * 60
    epoch = 0
    model.train()
    while step < cfg["max_steps"] and time.time() < deadline:
        sampler.set_epoch(epoch)
        for input_ids, label_ids, attention_mask in loader:
            if step >= cfg["max_steps"] or time.time() >= deadline:
                break
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            attention_mask = attention_mask.to(device)
            amp = (
                torch.amp.autocast(device.type, dtype=dtype)
                if use_amp
                else contextlib.nullcontext()
            )
            with amp:
                loss = compute_loss(model, input_ids, label_ids, attention_mask)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            tokens_seen += int(attention_mask.sum().item())
            if step % cfg.get("log_interval", 20) == 0:
                elapsed = time.time() - started
                record = {
                    "step": step,
                    "loss": round(loss.item(), 4),
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_per_sec": round(tokens_seen / max(elapsed, 1e-6), 1),
                    "elapsed_min": round(elapsed / 60, 2),
                }
                print(json.dumps(record))
                if run is not None:
                    run.log(record, step=step)
            if step % cfg["eval_interval"] == 0:
                metrics = evaluate(model, val_set, tokenizer, cfg, device)
                metrics.update({"step": step, "loss": loss.item(), "split": "validation"})
                with open(cfg["log_file"], "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics) + "\n")
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}))
                if run is not None:
                    logged = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                    run.log(logged, step=step)
                if metrics["char_accuracy"] > best:
                    best = metrics["char_accuracy"]
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": cfg,
                            "source_chars": tokenizer.source_chars,
                            "target_chars": tokenizer.target_chars,
                            "step": step,
                            "char_accuracy": best,
                        },
                        f"models/part1/{cfg['run_name']}.pt",
                    )
                model.train()
        epoch += 1

    summary = {
        "arch": cfg.get("arch"),
        "parameters": params,
        "flops": flops,
        "best_char_accuracy": best,
        "steps": step,
        "tokens_seen": tokens_seen,
        "elapsed_min": round((time.time() - started) / 60, 2),
        "checkpoint": f"models/part1/{cfg['run_name']}.pt",
    }
    os.makedirs("results/part1", exist_ok=True)
    with open(f"results/part1/{cfg['run_name']}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))
    if run is not None:
        run.finish()
    if args.push:
        from .hf import push_checkpoint

        pushed = push_checkpoint(
            f"models/part1/{cfg['run_name']}.pt",
            env.hf_repo(cfg.get("hf_repo")),
        )
        print(json.dumps({"pushed": pushed}))


if __name__ == "__main__":
    main()
