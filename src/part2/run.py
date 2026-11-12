import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from . import env
from .config import load_config
from .data import FeatureStore, edge_profile
from .train import run_once

DEFAULTS = {
    "data_root": "data/part2",
    "dataset": "gossipcop",
    "hidden": 128,
    "layers": 2,
    "heads": 4,
    "dropout": 0.5,
    "dropedge": 0.2,
    "batch_size": 32,
    "lr": 0.001,
    "weight_decay": 0.0005,
    "max_epochs": 100,
    "patience": 20,
    "seeds": [0, 1, 2, 3, 4],
    "normalize": True,
    "archs": ["mlp", "gcn", "sage", "gat", "bigcn"],
    "arch_feature": "bert",
    "ablation_backbone": "sage",
    "ablation_features": ["profile", "spacy", "bert", "content"],
    "output": "output/part2/results.json",
    "details": "results/part2/details.json",
    "curves": "results/part2/curves.json",
    "log": "logs/part2/run.jsonl",
    "wandb": True,
    "wandb_project": "anlp-a3-part2",
}

METRICS = ["accuracy", "precision", "recall", "f1"]


def mean_std(values):
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def aggregate(runs):
    means = {}
    stds = {}
    for metric in METRICS:
        values = [run["test"][metric] for run in runs]
        means[metric], stds[metric] = mean_std(values)
    return {"mean": means, "std": stds}


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_log(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def start_wandb(cfg, name, extra):
    if not cfg.get("wandb"):
        return None
    if not env.apply_wandb_key():
        return None
    import wandb

    return wandb.init(
        project=cfg.get("wandb_project") or env.wandb_project("anlp-a3-part2"),
        entity=env.wandb_entity(),
        name=name,
        config={key: value for key, value in cfg.items() if key != "seeds"},
        tags=[extra.get("kind", "part2")],
        reinit=True,
    )


def build_tasks(cfg):
    order = []
    for arch in cfg["archs"]:
        key = (arch, cfg["arch_feature"])
        if key not in order:
            order.append(key)
    for feature in cfg["ablation_features"]:
        key = (cfg["ablation_backbone"], feature)
        if key not in order:
            order.append(key)
    return order


def inspect(cfg):
    store = FeatureStore(cfg["data_root"], cfg["dataset"], cfg["normalize"])
    bundle = store.get(cfg["arch_feature"])
    splits = bundle["splits"]
    report = {
        "feature": cfg["arch_feature"],
        "classes": bundle["classes"],
        "orientation": bundle["orientation"],
        "edge_profile": edge_profile(splits["train"]),
        "sizes": {name: len(data) for name, data in splits.items()},
        "nodes": {
            name: int(sum(graph.x.size(0) for graph in data))
            for name, data in splits.items()
        },
        "dim": int(splits["train"][0].x.size(1)),
        "label_counts": {
            name: np.bincount(
                np.concatenate([graph.y.numpy() for graph in data])
            ).tolist()
            for name, data in splits.items()
        },
    }
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/part2/base.yaml")
    parser.add_argument("--ta", action="store_true")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--archs", default=None)
    parser.add_argument("--features", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    cfg = {**DEFAULTS, **load_config(args.config)}
    if args.seeds:
        cfg["seeds"] = [int(value) for value in args.seeds.split(",")]
    if args.max_epochs:
        cfg["max_epochs"] = args.max_epochs
    if args.archs:
        cfg["archs"] = args.archs.split(",")
    if args.features:
        cfg["ablation_features"] = args.features.split(",")
    if args.ta:
        cfg["seeds"] = cfg["seeds"][:1]
        cfg["output"] = "output/part2/results_ta.json"
        cfg["details"] = "results/part2/details_ta.json"
        cfg["curves"] = "results/part2/curves_ta.json"
        cfg["wandb"] = False
    if args.no_wandb:
        cfg["wandb"] = False

    if args.inspect:
        inspect(cfg)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    order = build_tasks(cfg)
    store = FeatureStore(cfg["data_root"], cfg["dataset"], cfg["normalize"])
    results = defaultdict(list)
    details = []
    started = time.time()
    remaining = defaultdict(int)
    for _, feature in order:
        remaining[feature] += 1

    print(
        json.dumps(
            {
                "device": str(device),
                "seeds": cfg["seeds"],
                "max_epochs": cfg["max_epochs"],
                "configurations": [
                    {"arch": arch, "feature": feature}
                    for arch, feature in order
                ],
            }
        )
    )

    for arch, feature in order:
        bundle = store.get(feature)
        name = f"part2-{arch}-{feature}"
        run = start_wandb(
            cfg, name, {"kind": "part2", "arch": arch, "feature": feature}
        )
        print(
            json.dumps(
                {
                    "configuration": name,
                    "orientation": bundle["orientation"],
                    "classes": bundle["classes"],
                }
            )
        )
        for index, seed in enumerate(cfg["seeds"]):
            def on_epoch(epoch, loss, val, seed=seed, index=index):
                if run is not None:
                    step = index * (int(cfg["max_epochs"]) + 1) + epoch
                    run.log(
                        {
                            f"seed{seed}/train_loss": loss,
                            f"seed{seed}/val_loss": val["loss"],
                            f"seed{seed}/val_f1": val["f1"],
                            f"seed{seed}/val_accuracy": val["accuracy"],
                        },
                        step=step,
                    )

            result = run_once(
                cfg, arch, feature, seed, bundle, device, on_epoch
            )
            results[(arch, feature)].append(result)
            details.append(
                {key: value for key, value in result.items() if key != "history"}
            )
            append_log(
                cfg["log"],
                {
                    "arch": arch,
                    "feature": feature,
                    "seed": seed,
                    "test": result["test"],
                    "best_epoch": result["best_epoch"],
                    "best_val": result["best_val"],
                    "seconds": result["seconds"],
                    "parameters": result["parameters"],
                },
            )
            print(
                json.dumps(
                    {
                        "arch": arch,
                        "feature": feature,
                        "seed": seed,
                        "best_epoch": result["best_epoch"],
                        "val_f1": round(result["best_val"]["f1"], 4),
                        "test": {
                            key: round(value, 4)
                            for key, value in result["test"].items()
                        },
                        "seconds": result["seconds"],
                    }
                )
            )
        if run is not None:
            run.finish()

        eval_run = start_wandb(
            cfg,
            f"{name}-evals",
            {"kind": "part2-evals", "arch": arch, "feature": feature},
        )
        stats = aggregate(results[(arch, feature)])
        if eval_run is not None:
            for seed, result in zip(cfg["seeds"], results[(arch, feature)]):
                eval_run.log(
                    {
                        f"seed{seed}/test_{key}": value
                        for key, value in result["test"].items()
                    }
                )
            eval_run.log(
                {
                    f"mean/test_{key}": stats["mean"][key]
                    for key in METRICS
                }
                | {
                    f"std/test_{key}": stats["std"][key]
                    for key in METRICS
                }
            )
            eval_run.finish()

        curves = {
            f"{key[0]}|{key[1]}": {
                str(item["seed"]): item["history"]
                for item in group
            }
            for key, group in results.items()
        }
        write_json(
            cfg["details"],
            {
                "config": cfg,
                "runs": details,
                "aggregate": {
                    f"{key[0]}|{key[1]}": aggregate(group)
                    for key, group in results.items()
                },
            },
        )
        write_json(cfg["curves"], curves)
        remaining[feature] -= 1
        if remaining[feature] == 0:
            store.release(feature)

    part_a = {
        arch: aggregate(results[(arch, cfg["arch_feature"])])["mean"]
        for arch in cfg["archs"]
    }
    part_b = {
        feature: aggregate(results[(cfg["ablation_backbone"], feature)])["mean"]
        for feature in cfg["ablation_features"]
    }
    payload = {"part_a": part_a, "part_b": part_b}
    write_json(cfg["output"], payload)
    print(json.dumps({"output": cfg["output"], "results": payload}, indent=2))
    print(
        json.dumps(
            {
                "elapsed_minutes": round((time.time() - started) / 60, 2),
                "details": cfg["details"],
                "curves": cfg["curves"],
            }
        )
    )


if __name__ == "__main__":
    main()
