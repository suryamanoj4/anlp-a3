import torch

from .tokenizer import strip_source


@torch.no_grad()
def generate_ids(model, source_ids, max_new_tokens, eos_id, sep_id, device):
    ids = list(source_ids) + [sep_id]
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        logits = model(input_ids, attention_mask)
        nxt = int(logits[0, -1].argmax())
        ids.append(nxt)
        if nxt == eos_id:
            break
    return ids[len(source_ids) + 1:]


@torch.no_grad()
def decode_document(model, tokenizer, cfg, ciphertext, device, max_chunks=None):
    text = strip_source(ciphertext)
    size = cfg.get("patch_size") or cfg.get("max_source_len", 128)
    chunks = [text[i:i + size] for i in range(0, len(text), size)] or [""]
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    parts = []
    for chunk in chunks:
        ids = tokenizer.encode_source(chunk)
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
