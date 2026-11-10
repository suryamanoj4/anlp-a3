import torch
import triton
import triton.language as tl


@triton.jit
def _scan_fwd_kernel(
    x_ptr, dt_ptr, b_ptr, c_ptr, a_ptr, y_ptr, h_ptr,
    T, D, N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // D
    d = pid % D
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    a = tl.load(a_ptr + d * N + offs, mask=mask, other=0.0).to(tl.float32)
    h = tl.zeros([BLOCK_N], dtype=tl.float32)
    base_x = batch * T * D + d
    base_bc = batch * T * N
    base_h = (batch * T * D + d) * N
    for t in range(T):
        dt = tl.load(dt_ptr + base_x + t * D).to(tl.float32)
        xv = tl.load(x_ptr + base_x + t * D).to(tl.float32)
        bv = tl.load(b_ptr + base_bc + t * N + offs, mask=mask, other=0.0).to(tl.float32)
        cv = tl.load(c_ptr + base_bc + t * N + offs, mask=mask, other=0.0).to(tl.float32)
        h = tl.exp(dt * a) * h + dt * bv * xv
        y = tl.sum(h * cv, axis=0)
        tl.store(y_ptr + base_x + t * D, y)
        tl.store(h_ptr + base_h + t * D * N + offs, h, mask=mask)


@triton.jit
def _scan_bwd_kernel(
    x_ptr, dt_ptr, b_ptr, c_ptr, a_ptr, h_ptr, dy_ptr,
    dx_ptr, ddt_ptr, g_ptr, da_ptr,
    T, D, N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // D
    d = pid % D
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    a = tl.load(a_ptr + d * N + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.zeros([BLOCK_N], dtype=tl.float32)
    da = tl.zeros([BLOCK_N], dtype=tl.float32)
    base_x = batch * T * D + d
    base_bc = batch * T * N
    base_h = (batch * T * D + d) * N
    for idx in range(T):
        t = T - 1 - idx
        dt = tl.load(dt_ptr + base_x + t * D).to(tl.float32)
        xv = tl.load(x_ptr + base_x + t * D).to(tl.float32)
        bv = tl.load(b_ptr + base_bc + t * N + offs, mask=mask, other=0.0).to(tl.float32)
        cv = tl.load(c_ptr + base_bc + t * N + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + base_x + t * D).to(tl.float32)
        ht = tl.load(h_ptr + base_h + t * D * N + offs, mask=mask, other=0.0).to(tl.float32)
        if t > 0:
            hprev = tl.load(h_ptr + base_h + (t - 1) * D * N + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            hprev = tl.zeros([BLOCK_N], dtype=tl.float32)
        at = tl.exp(dt * a)
        g = g + cv * dy
        dxv = dt * tl.sum(bv * g, axis=0)
        ddtv = tl.sum(g * (a * at * hprev + bv * xv), axis=0)
        tl.store(dx_ptr + base_x + t * D, dxv)
        tl.store(ddt_ptr + base_x + t * D, ddtv)
        tl.store(g_ptr + base_h + t * D * N + offs, g, mask=mask)
        da += g * dt * at * hprev
        g = at * g
    tl.atomic_add(da_ptr + d * N + offs, da, mask=mask)


class MambaScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, bmat, cmat, A):
        b, t, d = x.shape
        n = A.shape[1]
        x = x.contiguous()
        delta = delta.contiguous()
        bmat = bmat.contiguous()
        cmat = cmat.contiguous()
        A = A.contiguous()
        y = torch.empty((b, t, d), device=x.device, dtype=torch.float32)
        hidden = torch.empty((b, t, d, n), device=x.device, dtype=torch.float32)
        block_n = triton.next_power_of_2(n)
        _scan_fwd_kernel[(b * d,)](
            x, delta, bmat, cmat, A, y, hidden, t, d, n, block_n
        )
        ctx.save_for_backward(x, delta, bmat, cmat, A, hidden)
        ctx.block_n = block_n
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        x, delta, bmat, cmat, A, hidden = ctx.saved_tensors
        b, t, d = x.shape
        n = A.shape[1]
        dy = dy.contiguous().to(torch.float32)
        dx = torch.empty((b, t, d), device=x.device, dtype=torch.float32)
        ddt = torch.empty((b, t, d), device=x.device, dtype=torch.float32)
        adjoint = torch.empty((b, t, d, n), device=x.device, dtype=torch.float32)
        da = torch.zeros((d, n), device=x.device, dtype=torch.float32)
        _scan_bwd_kernel[(b * d,)](
            x, delta, bmat, cmat, A, hidden, dy,
            dx, ddt, adjoint, da, t, d, n, ctx.block_n
        )
        db = (adjoint * delta.unsqueeze(-1) * x.unsqueeze(-1)).sum(dim=2)
        dc = (dy.unsqueeze(-1) * hidden).sum(dim=2)
        return (
            dx.to(x.dtype),
            ddt.to(delta.dtype),
            db.to(bmat.dtype),
            dc.to(cmat.dtype),
            da.to(A.dtype),
        )


def scan_triton(u, delta, bmat, cmat, A):
    return MambaScan.apply(u, delta, bmat, cmat, A)
