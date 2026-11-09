import torch

from .models.build import build_model

BASE = {
    "d_model": 32,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 64,
    "dropout": 0.0,
    "max_position": 128,
    "d_state": 8,
    "chunk": 16,
}


def forward_backward(kind, vocab_size=64, length=48, batch=2):
    cfg = dict(BASE)
    cfg["arch"] = kind
    model = build_model(cfg, vocab_size)
    model.train()
    input_ids = torch.randint(0, vocab_size, (batch, length))
    attention_mask = torch.ones(batch, length, dtype=torch.long)
    if batch > 1:
        attention_mask[1, length // 2:] = 0
    logits = model(input_ids, attention_mask)
    assert logits.shape == (batch, length, vocab_size), (kind, tuple(logits.shape))
    assert torch.isfinite(logits).all(), f"{kind}: non-finite logits"
    logits.mean().backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad, f"{kind}: parameters without gradient: {no_grad[:5]}"
    params = sum(p.numel() for p in model.parameters())
    return model, params


def check_causality(kind, vocab_size=64, length=24):
    model, _ = forward_backward(kind, vocab_size, length, batch=1)
    model.eval()
    ids = torch.randint(0, vocab_size, (1, length))
    mask = torch.ones(1, length, dtype=torch.long)
    with torch.no_grad():
        first = model(ids, mask)
        perturbed = ids.clone()
        perturbed[0, -1] = (perturbed[0, -1] + 7) % vocab_size
        second = model(perturbed, mask)
    delta = (first[:, :-1] - second[:, :-1]).abs().max().item()
    assert delta < 1e-5, f"{kind}: future token changed past outputs (max delta {delta})"
    return delta


def check_padding(kind, vocab_size=64, length=32):
    model, _ = forward_backward(kind, vocab_size, length, batch=2)
    model.eval()
    ids = torch.randint(0, vocab_size, (1, length))
    mask = torch.ones(1, length, dtype=torch.long)
    mask[0, length // 2:] = 0
    with torch.no_grad():
        logits = model(ids, mask)
    assert torch.isfinite(logits).all(), f"{kind}: non-finite logits with padding"


def main():
    print(f"{'arch':<12}{'logits':<22}{'params':>10}{'causal_delta':>16}")
    for kind in ["transformer", "mamba", "deltanet", "hybrid"]:
        _, params = forward_backward(kind)
        delta = check_causality(kind)
        check_padding(kind)
        print(f"{kind:<12}{str((2, 48, 64)):<22}{params:>10}{delta:>16.2e}")
    print("all checks passed")


if __name__ == "__main__":
    main()
