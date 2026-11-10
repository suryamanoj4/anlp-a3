from .deltanet import DeltaNetMixer
from .mamba import MambaMixer
from .stack import MixerLM
from .transformer import TransformerLM


def build_model(cfg, vocab_size):
    arch = cfg.get("arch", "transformer")
    d_model = cfg["d_model"]
    n_layers = cfg["n_layers"]
    if arch == "transformer":
        return TransformerLM(
            vocab_size,
            d_model,
            cfg["n_heads"],
            n_layers,
            cfg["d_ff"],
            cfg["dropout"],
            cfg["max_position"],
        )
    if arch == "mamba":
        mixers = [
            (MambaMixer(d_model, cfg.get("d_state", 16), expand=cfg.get("expand", 4), scan_chunk=cfg.get("scan_chunk", 16), checkpoint_scan=cfg.get("checkpoint_scan", True), triton_scan=cfg.get("triton_scan", False)), False)
            for _ in range(n_layers)
        ]
    elif arch == "deltanet":
        mixers = [
            (DeltaNetMixer(d_model, cfg["n_heads"], cfg.get("chunk", 64)), True)
            for _ in range(n_layers)
        ]
    elif arch == "hybrid":
        mixers = []
        for i in range(n_layers):
            if i % 2 == 0:
                mixers.append((MambaMixer(d_model, cfg.get("d_state", 16), expand=cfg.get("expand", 4), scan_chunk=cfg.get("scan_chunk", 16), checkpoint_scan=cfg.get("checkpoint_scan", True), triton_scan=cfg.get("triton_scan", False)), False))
            else:
                mixers.append((DeltaNetMixer(d_model, cfg["n_heads"], cfg.get("chunk", 64)), True))
    else:
        raise ValueError(f"unknown architecture: {arch}")
    return MixerLM(vocab_size, d_model, mixers, cfg["d_ff"], cfg["dropout"])
