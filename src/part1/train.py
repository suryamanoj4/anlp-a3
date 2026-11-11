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
from .data import (
    CipherPairs,
    LengthBucketSampler,
    PatchedPairs,
    TransducerCollate,
    limit_for_tokens,
    make_collate,
)
from .evaluate import generation_metrics
from .flops import count_parameters, estimate_flops
from .models.build import build_model
from .tokenizer import Tokenizer


def compute_loss(model, input_ids, label_ids, attention_mask, label_smoothing=0.0, head=None, shift=True):
    logits = model(input_ids, attention_mask)
    if head is not None:
        offset, classes = head
        logits = logits[:, :, offset:offset + classes]
    if shift:
        logits = logits[:, :-1, :]
        label_ids = label_ids[:, 1:]
    logits = logits.contiguous()
    labels = label_ids.contiguous()
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )


def lr_factor(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def validation_loss(model, loader, device, max_batches=60, head=None, shift=True):
    model.eval()
    total = 0.0
    count = 0
    for input_ids, label_ids, attention_mask in loader:
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)
        attention_mask = attention_mask.to(device)
        total += float(compute_loss(model, input_ids, label_ids, attention_mask, head=head, shift=shift))
        count += 1
        if count >= max_batches:
            break
    return total / max(1, count)


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
    train_set = None
    val_set = None
    sampler = None
    patch_size = cfg.get("patch_size")
    if patch_size:
        train_docs = cfg.get("max_docs")
        if train_docs is None and cfg.get("max_tokens"):
            train_docs = limit_for_tokens(
                f"{root}/{cfg['train_split']}/cleaned.jsonl", cfg["max_tokens"]
            )
        val_docs = cfg.get("val_max_docs")
        if val_docs is None and cfg.get("val_max_tokens"):
            val_docs = limit_for_tokens(
                f"{root}/{cfg['val_split']}/cleaned.jsonl", cfg["val_max_tokens"]
            )
        train_set = PatchedPairs(
            root, cfg["train_split"], patch_size, train_docs, cfg.get("warmup_chars", 0)
        )
        val_set = PatchedPairs(
            root, cfg["val_split"], patch_size, val_docs, cfg.get("warmup_chars", 0)
        )
        transducer = cfg.get("formulation") == "transducer"
        collate = TransducerCollate(tokenizer) if transducer else make_collate(tokenizer, patch_size, patch_size)
        loader = DataLoader(
            train_set,
            batch_size=cfg["batch_size"],
            shuffle=True,
            collate_fn=collate,
            num_workers=cfg.get("workers", 2),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg["batch_size"],
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
    else:
        transducer = cfg.get("formulation") == "transducer"
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
        val_sampler = LengthBucketSampler(
            val_set.source_lengths, cfg["batch_size"], shuffle=False, seed=cfg["seed"]
        )
        val_loader = DataLoader(
            val_set, batch_sampler=val_sampler, collate_fn=collate, num_workers=0
        )

    raw_model = build_model(cfg, tokenizer.vocab_size).to(device)
    model = torch.compile(raw_model, dynamic=True) if cfg.get("compile") else raw_model
    params = count_parameters(raw_model)
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

    head = (tokenizer.tgt_offset, len(tokenizer.target_chars)) if cfg.get("formulation") == "transducer" else None
    step = 0
    best = float("inf")
    saved = False
    tokens_seen = 0
    started = time.time()
    minutes = cfg.get("max_minutes")
    deadline = started + minutes * 60 if minutes else float("inf")
    epoch = 0
    model.train()
    while step < cfg["max_steps"] and time.time() < deadline:
        if sampler is not None:
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
                loss = compute_loss(
                    model, input_ids, label_ids, attention_mask,
                    cfg.get("label_smoothing", 0.0),
                    head=head, shift=not transducer,
                )
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
                    "epoch": epoch,
                    "loss": round(loss.item(), 4),
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_seen": tokens_seen,
                    "tokens_per_sec": round(tokens_seen / max(elapsed, 1e-6), 1),
                    "elapsed_min": round(elapsed / 60, 2),
                }
                print(json.dumps(record))
                if run is not None:
                    run.log(record, step=step)
            if step % cfg["eval_interval"] == 0:
                val = validation_loss(model, val_loader, device, head=head, shift=not transducer)
                metrics = {
                    "step": step,
                    "split": "validation",
                    "val_loss": val,
                    "train_loss": loss.item(),
                }
                with open(cfg["log_file"], "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics))
                if run is not None:
                    run.log(metrics, step=step)
                if val < best:
                    best = val
                    saved = True
                    torch.save(
                        {
                            "model": raw_model.state_dict(),
                            "config": cfg,
                            "source_chars": tokenizer.source_chars,
                            "target_chars": tokenizer.target_chars,
                            "step": step,
                            "val_loss": best,
                        },
                        f"models/part1/{cfg['run_name']}.pt",
                    )
                model.train()
        epoch += 1

    if not saved:
        torch.save(
            {
                "model": raw_model.state_dict(),
                "config": cfg,
                "source_chars": tokenizer.source_chars,
                "target_chars": tokenizer.target_chars,
                "step": step,
                "val_loss": best,
            },
            f"models/part1/{cfg['run_name']}.pt",
        )

    summary = {
        "arch": cfg.get("arch"),
        "parameters": params,
        "flops": flops,
        "best_val_loss": best if saved else None,
        "steps": step,
        "tokens_seen": tokens_seen,
        "elapsed_min": round((time.time() - started) / 60, 2),
        "checkpoint": f"models/part1/{cfg['run_name']}.pt",
    }
    os.makedirs("results/part1", exist_ok=True)
    if saved and cfg.get("final_eval", True):
        payload = torch.load(
            f"models/part1/{cfg['run_name']}.pt", map_location=device, weights_only=False
        )
        raw_model.load_state_dict(payload["model"])
        evaluator_started = time.time()
        test_metrics = generation_metrics(
            raw_model,
            tokenizer,
            cfg,
            cfg.get("eval_split", "test"),
            cfg.get("eval_limit", 300),
            cfg.get("eval_all_lengths", False),
            device,
        )
        test_metrics["eval_minutes"] = round((time.time() - evaluator_started) / 60, 2)
        summary["test"] = test_metrics
        with open(f"results/part1/{cfg['run_name']}_{cfg.get('eval_split', 'test')}.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        print(json.dumps({"test": test_metrics}))
        if run is not None:
            run.log(
                {f"test/{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))}
            )
    with open(f"results/part1/{cfg['run_name']}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))
    if run is not None:
        run.log({k: v for k, v in summary.items() if isinstance(v, (int, float))})
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
