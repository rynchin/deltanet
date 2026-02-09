import torch
from deltanet_naive import deltanet_forward_naive
from deltanet_ref import deltanet_forward_ref
from deltanet_chunk import deltanet_forward_chunk

def main():
    torch.manual_seed(0)
    dtype = torch.float64
    device = "cpu"

    B, L, D = 10, 100, 100

    q = torch.randn(B, L, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, L, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, L, D, dtype=dtype, device=device, requires_grad=True)
    beta_logits = torch.randn(B, L, dtype=dtype, device=device, requires_grad=True)
    beta = torch.sigmoid(beta_logits)

    # forward
    out_naive = deltanet_forward_naive(q, k, v, beta)
    out_ref = deltanet_forward_ref(q, k, v, beta)
    out_chunk = deltanet_forward_chunk(q, k, v, beta)

    torch.testing.assert_close(out_chunk, out_ref, rtol=1e-10, atol=1e-12)
    print("forward passes!!")

    # backward
    loss_chunk = (out_chunk ** 2).mean()
    grads_chunk = torch.autograd.grad(loss_chunk, (q, k, v, beta_logits), retain_graph=True)

    loss_ref = (out_ref ** 2).mean()
    grads_ref = torch.autograd.grad(loss_ref, (q, k, v, beta_logits))

    names = ["q", "k", "v", "beta_logits"]
    for name, gn, gr in zip(names, grads_chunk, grads_ref):
        torch.testing.assert_close(gn, gr, rtol=1e-9, atol=1e-11)
        assert torch.isfinite(gn).all()
    print("backward grads match!!!")


if __name__ == "__main__":
    main()
