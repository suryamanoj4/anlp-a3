import torch

from .tokenizer import strip_source


@torch.no_grad()
def predict_document(model, tokenizer, cfg, ciphertext, device, max_chunks=None):
    text = strip_source(ciphertext)
    size = cfg.get("patch_size") or cfg.get("max_source_len", 64)
    warmup = cfg.get("warmup_chars", 0)
    offset = tokenizer.tgt_offset
    classes = len(tokenizer.target_chars)
    bounds = []
    for start in range(0, len(text), size):
        end = min(start + size, len(text))
        bounds.append((max(0, start - warmup), start, end))
    if not bounds:
        bounds = [(0, 0, 0)]
    if max_chunks is not None:
        bounds = bounds[:max_chunks]
    parts = []
    for src_start, start, end in bounds:
        chunk = text[src_start:end]
        ids = tokenizer.encode_source(chunk)
        length = len(ids)
        span = end - start
        if length == 0 or span <= 0:
            continue
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        logits = model(input_ids, attention_mask)[0, :, offset:offset + classes]
        preds = logits[length - span:length].argmax(dim=-1).tolist()
        parts.append("".join(tokenizer.target_chars[i] for i in preds))
    return "".join(parts)
