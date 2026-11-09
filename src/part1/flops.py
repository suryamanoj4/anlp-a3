def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def transformer_flops(cfg, seq_len, vocab_size):
    d = cfg["d_model"]
    layers = cfg["n_layers"]
    params = layers * (12 * d * d + 13 * d) + 2 * vocab_size * d
    attention = 4 * layers * seq_len * seq_len * d
    return 2 * params * seq_len + attention


def mamba_flops(cfg, seq_len, vocab_size):
    d = cfg["d_model"]
    layers = cfg["n_layers"]
    d_inner = cfg.get("expand", 4) * d
    d_state = cfg.get("d_state", 16)
    params = layers * (6 * d * d_inner + 3 * d_inner * d_state + 2 * d * d) + 2 * vocab_size * d
    scan = 9 * layers * seq_len * d_inner * d_state
    return 2 * params * seq_len + scan


def deltanet_flops(cfg, seq_len, vocab_size):
    d = cfg["d_model"]
    layers = cfg["n_layers"]
    chunk = cfg.get("chunk", 64)
    params = layers * (12 * d * d) + 2 * vocab_size * d
    recurrent = layers * seq_len * (4 * chunk * d + 4 * d * d)
    return 2 * params * seq_len + recurrent


def hybrid_flops(cfg, seq_len, vocab_size):
    mamba_cfg = dict(cfg)
    mamba_cfg["n_layers"] = cfg["n_layers"] // 2
    delta_cfg = dict(cfg)
    delta_cfg["n_layers"] = cfg["n_layers"] - cfg["n_layers"] // 2
    total = mamba_flops(mamba_cfg, seq_len, vocab_size) + deltanet_flops(
        delta_cfg, seq_len, vocab_size
    )
    return total - 2 * 2 * vocab_size * cfg["d_model"] * seq_len


def estimate_flops(cfg, seq_len, vocab_size):
    arch = cfg.get("arch", "transformer")
    if arch == "transformer":
        return transformer_flops(cfg, seq_len, vocab_size)
    if arch == "mamba":
        return mamba_flops(cfg, seq_len, vocab_size)
    if arch == "deltanet":
        return deltanet_flops(cfg, seq_len, vocab_size)
    if arch == "hybrid":
        return hybrid_flops(cfg, seq_len, vocab_size)
    raise ValueError(f"unknown architecture: {arch}")
