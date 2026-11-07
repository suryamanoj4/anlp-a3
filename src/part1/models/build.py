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
        mixers = [MambaMixer(d_model, cfg.get("d_state", 16)) for _ in range(n_layers)]
    elif arch == "deltanet":
        mixers = [
            DeltaNetMixer(d_model, cfg["n_heads"], cfg.get("chunk", 64))
            for _ in range(n_layers)
        ]
    elif arch == "hybrid":
        mixers = []
        for i in range(n_layers):
            if i % 2 == 0:
                mixers.append(MambaMixer(d_model, cfg.get("d_state", 16)))
            else:
                mixers.append(DeltaNetMixer(d_model, cfg["n_heads"], cfg.get("chunk", 64)))
    else:
        raise ValueError(f"unknown architecture: {arch}")
    return MixerLM(vocab_size, d_model, cfg["d_ff"], cfg["dropout"], mixers)
