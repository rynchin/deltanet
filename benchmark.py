import time
import torch
from typing import cast

from deltanet_naive import deltanet_forward_naive
from deltanet_ref import deltanet_forward_ref
from deltanet import deltanet_forward, deltanet_backward

def bench_ms(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters

def zero_grads(*xs):
    for x in xs:
        if x.grad is not None:
            x.grad.zero_()

def grab_grad(x: torch.Tensor) -> torch.Tensor:
    assert x.grad is not None, "grad is None"
    return x.grad

class DeltaNetOurs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta_logits, c):
        beta = torch.sigmoid(beta_logits).to(q.dtype)
        out = deltanet_forward(q, k, v, beta, c=c)
        ctx.save_for_backward(q, k, v, beta)
        ctx.c = c
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, beta = ctx.saved_tensors
        c = ctx.c
        dq, dk, dv, dbeta = deltanet_backward(q, k, v, beta, dout, c=c)
        dbeta_logits = dbeta * beta * (1.0 - beta)
        return dq, dk, dv, dbeta_logits, None

def ours_forward(q, k, v, beta_logits, c=128):
    out = DeltaNetOurs.apply(q, k, v, beta_logits, c)
    return cast(torch.Tensor, out)  # tell pylance it's a Tensor

def main():
    torch.manual_seed(0)

    # CORRECTNESS
    dtype = torch.float64
    device = "cpu"
    B, L, D = 10, 100, 100
    c = 64

    q = torch.randn(B, L, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, L, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, L, D, dtype=dtype, device=device, requires_grad=True)
    beta_logits = torch.randn(B, L, dtype=dtype, device=device, requires_grad=True)
    beta = torch.sigmoid(beta_logits)

    out_naive = deltanet_forward_naive(q, k, v, beta)
    out_ref = deltanet_forward_ref(q, k, v, beta)
    out_ours = ours_forward(q, k, v, beta_logits, c=c)

    torch.testing.assert_close(out_naive, out_ref, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(out_ours, out_ref, rtol=1e-10, atol=1e-12)
    print("forward passes match!!")

    zero_grads(q, k, v, beta_logits)
    ((out_ref ** 2).mean()).backward(retain_graph=True)
    grads_ref = (grab_grad(q).clone(), grab_grad(k).clone(), grab_grad(v).clone(), grab_grad(beta_logits).clone())

    zero_grads(q, k, v, beta_logits)
    ((out_ours ** 2).mean()).backward()
    grads_ours = (grab_grad(q).clone(), grab_grad(k).clone(), grab_grad(v).clone(), grab_grad(beta_logits).clone())

    names = ["q", "k", "v", "beta_logits"]
    for name, go, gr in zip(names, grads_ours, grads_ref):
        torch.testing.assert_close(go, gr, rtol=1e-9, atol=1e-11)
        assert torch.isfinite(go).all()
    print("backward grads match!!")

    # BENCHMARK
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    B, L, D = (16, 2048, 128) if device == "cuda" else (8, 512, 128)
    c = 128

    q = torch.randn(B, L, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, L, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, L, D, dtype=dtype, device=device, requires_grad=True)
    beta_logits = torch.randn(B, L, dtype=torch.float32, device=device, requires_grad=True)

    def naive_fwd():
        beta = torch.sigmoid(beta_logits).to(q.dtype)
        return deltanet_forward_naive(q, k, v, beta)

    def ref_fwd():
        beta = torch.sigmoid(beta_logits).to(q.dtype)
        return deltanet_forward_ref(q, k, v, beta)

    def ours_fwd():
        return ours_forward(q, k, v, beta_logits, c=c)

    # backward = run loss=(out^2).mean(); backward()
    def naive_bwd():
        zero_grads(q, k, v, beta_logits)
        out = naive_fwd()
        ((out ** 2).mean()).backward()

    def ref_bwd():
        zero_grads(q, k, v, beta_logits)
        out = ref_fwd()
        ((out ** 2).mean()).backward()

    def ours_bwd():
        zero_grads(q, k, v, beta_logits)
        out = ours_fwd()
        ((out ** 2).mean()).backward()

    iters_fwd = 10 if device == "cpu" else 30
    iters_bwd = 3 if device == "cpu" else 10

    print(f"Benchmark ({device}, dtype={dtype}, B={B}, L={L}, D={D}, c={c})")
    print(f"naive forward:  {bench_ms(naive_fwd, iters=iters_fwd, warmup=2):.3f} ms/iter")
    print(f"ref forward:    {bench_ms(ref_fwd, iters=iters_fwd, warmup=2):.3f} ms/iter")
    print(f"ours forward:   {bench_ms(ours_fwd, iters=iters_fwd, warmup=2):.3f} ms/iter")
    print(f"\n")
    print(f"naive backward: {bench_ms(naive_bwd, iters=iters_bwd, warmup=1):.3f} ms/iter")
    print(f"ref backward:   {bench_ms(ref_bwd, iters=iters_bwd, warmup=1):.3f} ms/iter")
    print(f"ours backward:  {bench_ms(ours_bwd, iters=iters_bwd, warmup=1):.3f} ms/iter")


if __name__ == "__main__":
    main()