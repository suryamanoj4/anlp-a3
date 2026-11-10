import torch

from .models.mamba import MambaMixer


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")
    ref = MambaMixer(64, d_state=8, expand=2, checkpoint_scan=False, triton_scan=False).to(device)
    tri = MambaMixer(64, d_state=8, expand=2, checkpoint_scan=False, triton_scan=True).to(device)
    tri.load_state_dict(ref.state_dict())
    assert tri.triton_scan, "triton scan not active"

    x_ref = torch.randn(2, 40, 64, device=device, requires_grad=True)
    x_tri = x_ref.detach().clone().requires_grad_(True)

    y_ref = ref(x_ref)
    y_tri = tri(x_tri)
    print("forward   max abs diff: %.6e" % (y_ref - y_tri).abs().max().item())

    g_ref = torch.autograd.grad(y_ref.square().sum(), x_ref, retain_graph=True)[0]
    g_tri = torch.autograd.grad(y_tri.square().sum(), x_tri, retain_graph=True)[0]
    print("dx        max abs diff: %.6e" % (g_ref - g_tri).abs().max().item())

    params_ref = dict(ref.named_parameters())
    params_tri = dict(tri.named_parameters())
    for name, p in params_ref.items():
        gr = torch.autograd.grad(y_ref.square().sum(), p, retain_graph=True)[0]
        gt = torch.autograd.grad(y_tri.square().sum(), params_tri[name], retain_graph=True)[0]
        print("%-12s max abs diff: %.6e  scale %.3e" % (
            name, (gr - gt).abs().max().item(), gr.abs().max().item()))


if __name__ == "__main__":
    main()
