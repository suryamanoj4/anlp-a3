import torch


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
