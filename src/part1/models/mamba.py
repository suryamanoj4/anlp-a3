import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaMixer(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None, scan_chunk=16):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_model // 16)
        self.scan_chunk = scan_chunk
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        b, t, _ = x.shape
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv1d(u.transpose(1, 2))[..., :t]
        u = F.silu(u).transpose(1, 2)
        projected = self.x_proj(u)
        delta, bmat, cmat = projected.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta)).clamp(1e-4, 0.1)
        A = -torch.exp(self.A_log)
        y = self.scan(u, delta, bmat, cmat, A)
        y = y + u * self.D
        return self.out_proj(y * F.silu(z))

    def scan(self, u, delta, bmat, cmat, A):
        b, t, d = u.shape
        n = self.d_state
        a = delta.unsqueeze(-1) * A
        bx = u.unsqueeze(-1) * bmat.unsqueeze(2) * delta.unsqueeze(-1)
        state = torch.zeros(b, d, n, device=u.device, dtype=u.dtype)
        outputs = []
        for start in range(0, t, self.scan_chunk):
            ac = a[:, start:start + self.scan_chunk]
            bc = bx[:, start:start + self.scan_chunk]
            cum = torch.cumsum(ac, dim=1)
            local = torch.exp(cum) * torch.cumsum(bc * torch.exp(-cum), dim=1)
            outputs.append(local + torch.exp(cum) * state.unsqueeze(1))
            state = torch.exp(cum[:, -1]) * state + local[:, -1]
        hidden = torch.cat(outputs, dim=1)
        return (hidden * cmat.unsqueeze(2)).sum(-1)
