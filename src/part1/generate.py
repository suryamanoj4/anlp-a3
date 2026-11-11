import torch

from .tokenizer import strip_source


@torch.no_grad()
def generate_ids(model, source_ids, max_new_tokens, eos_id, sep_id, device):
    prompt = list(source_ids) + [sep_id]
    total = len(prompt) + max_new_tokens
    buffer = torch.zeros((1, total), dtype=torch.long, device=device)
    buffer[0, :len(prompt)] = torch.tensor(prompt, dtype=torch.long, device=device)
    length = len(prompt)
    out = []
    for _ in range(max_new_tokens):
        input_ids = buffer[:, :length]
        attention_mask = torch.ones((1, length), dtype=torch.long, device=device)
        logits = model(input_ids, attention_mask)
        nxt = int(logits[0, length - 1].argmax())
        out.append(nxt)
        if nxt == eos_id:
            break
        buffer[0, length] = nxt
        length += 1
    return out


@torch.no_grad()
def decode_document(model, tokenizer, cfg, ciphertext, device, max_chunks=None):
    text = strip_source(ciphertext)
    size = cfg.get("patch_size") or cfg.get("max_source_len", 128)
    warmup = cfg.get("warmup_chars", 0)
    bounds = []
    for start in range(0, len(text), size):
        bounds.append((max(0, start - warmup), min(start + size, len(text))))
    if not bounds:
        bounds = [(0, 0)]
    if max_chunks is not None:
        bounds = bounds[:max_chunks]
    parts = []
    for src_start, end in bounds:
        ids = tokenizer.encode_source(text[src_start:end])
        out = generate_ids(
            model,
            ids,
            cfg.get("max_new_tokens", 128),
            tokenizer.eos_id,
            tokenizer.sep_id,
            device,
        )
        parts.append(tokenizer.decode_target(out))
    return "".join(parts)
