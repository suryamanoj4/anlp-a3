import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaNetMixer(nn.Module):
    def __init__(self, d_model, n_heads, chunk=64, d_conv=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk = chunk
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.q_conv = nn.Conv1d(d_model, d_model, d_conv, groups=d_model, padding=d_conv - 1)
        self.k_conv = nn.Conv1d(d_model, d_model, d_conv, groups=d_model, padding=d_conv - 1)
        self.v_conv = nn.Conv1d(d_model, d_model, d_conv, groups=d_model, padding=d_conv - 1)
        self.o_norm = nn.RMSNorm(d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, t, c = x.shape
        h = self.n_heads
        d = self.head_dim
        q = self.q_conv(self.q_proj(x).transpose(1, 2))[..., :t]
        k = self.k_conv(self.k_proj(x).transpose(1, 2))[..., :t]
        v = self.v_conv(self.v_proj(x).transpose(1, 2))[..., :t]
        q = F.silu(q.transpose(1, 2))
        k = F.silu(k.transpose(1, 2))
        v = v.transpose(1, 2)
        q = F.normalize(q.view(b, t, h, d), dim=-1).transpose(1, 2)
        k = F.normalize(k.view(b, t, h, d), dim=-1).transpose(1, 2)
        v = v.view(b, t, h, d).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        state = torch.zeros(b, h, d, d, device=x.device, dtype=x.dtype)
        outputs = []
        for start in range(0, t, self.chunk):
            end = min(start + self.chunk, t)
            qc = q[:, :, start:end]
            kc = k[:, :, start:end]
            vc = v[:, :, start:end]
            bc = beta[:, :, start:end].unsqueeze(-1)
            span = end - start
            kb = kc * bc
            vb = vc * bc
            lower = torch.tril(kc @ kb.transpose(-1, -2), diagonal=-1)
            eye = torch.eye(span, device=x.device, dtype=x.dtype)
            mat = eye + lower
            w = torch.linalg.solve_triangular(mat, kb, upper=False, unitriangular=True)
            u = torch.linalg.solve_triangular(mat, vb, upper=False, unitriangular=True)
            u = u - w @ state
            intra = torch.tril(qc @ kc.transpose(-1, -2)) @ u
            outputs.append(qc @ state + intra)
            state = state + kc.transpose(-1, -2) @ u
        out = torch.cat(outputs, dim=2).transpose(1, 2).reshape(b, t, c)
        return self.o_proj(self.o_norm(out))
