import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import make_collate
from .metrics import binary_metrics, classification_metrics
from .models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(dataset, batch_size, shuffle, collate, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=0,
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def predict(model, loader, device, num_classes):
    model.eval()
    logits = []
    targets = []
    total_loss = 0.0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = F.cross_entropy(out, batch.y)
        total_loss += float(loss) * out.size(0)
        total += out.size(0)
        logits.append(out.detach().cpu())
        targets.append(batch.y.detach().cpu())
    logits = torch.cat(logits)
    targets = torch.cat(targets)
    predictions = logits.argmax(-1)
    metrics = classification_metrics(
        targets.numpy(), predictions.numpy(), num_classes=num_classes
    )
    metrics["loss"] = total_loss / max(total, 1)
    return metrics, targets.numpy(), predictions.numpy(), logits.numpy()


def run_once(cfg, arch, feature, seed, bundle, device, on_epoch=None):
    set_seed(seed)
    splits = bundle["splits"]
    mean, std = bundle["stats"]
    flip = bundle["flip"]
    classes = bundle["classes"]

    collate = make_collate(mean, std, flip)
    train_loader = make_loader(
        splits["train"], int(cfg["batch_size"]), True, collate, seed
    )
    val_loader = make_loader(
        splits["val"], int(cfg["batch_size"]), False, collate
    )
    test_loader = make_loader(
        splits["test"], int(cfg["batch_size"]), False, collate
    )

    in_dim = int(splits["train"][0].x.size(1))
    model = build_model(arch, in_dim, classes, cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    best_val = None
    history = []
    started = time.time()

    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
            running += float(loss) * out.size(0)
            seen += out.size(0)
        train_loss = running / max(seen, 1)
        val_metrics, _, _, _ = predict(model, val_loader, device, classes)
        record = {"epoch": epoch, "train_loss": train_loss}
        record.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(record)
        if on_epoch is not None:
            on_epoch(epoch, train_loss, val_metrics)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_val = dict(val_metrics)
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= int(cfg["patience"]):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, y_true, y_pred, _ = predict(
        model, test_loader, device, classes
    )
    return {
        "arch": arch,
        "feature": feature,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "test": test_metrics,
        "test_binary": binary_metrics(y_true, y_pred, positive=1),
        "epochs": len(history),
        "seconds": round(time.time() - started, 2),
        "parameters": sum(p.numel() for p in model.parameters()),
        "history": history,
    }
