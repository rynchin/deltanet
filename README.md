# DeltaNet

PyTorch implementation of linear attention with the [delta rule](https://arxiv.org/abs/2406.06484) (Yang et al), which replaces additive state updates with a delta update for better associative recall.

## Overview

This repo includes:
- `deltanet_naive.py`: per-token update (simple, slow)
- `deltanet_ref.py`: correctness reference (unoptimized)
- `deltanet.py`: chunked implementation
- `benchmark.py`

## Benchmarks

`benchmark.py` measures **ms/iter** for `loss = (out^2).mean()` and `.backward()`.

**CPU (fp32)**
| impl | forward | backward |
|---|---:|---:|
| naive | 250.311 | 1305.558 |
| ref | 443.028 | 2080.503 |
| ours | 13.074 | 45.737 |

**CUDA (fp32)**
| impl | forward | backward |
|---|---:|---:|
| naive | 211.155 | 910.478 |
| ref | 1613.022 | 21249.556 |
| ours | 3.707 | 14.755 |


## Naive update

State $S_t \in \mathbb{R}^{d_v \times d_k}$. At step $t$:

```math
v_{\mathrm{old}} = S_{t-1} k_t,\quad
S_t \leftarrow S_{t-1} + \beta_t\, (v_t - v_{\mathrm{old}})\, k_t^\top,\quad
o_t = S_t q_t
```
See `deltanet_naive.py`.

## Chunked implementation
We chunk over the sequence to be hardware efficient. Code is at `deltanet.py`.

### Forward pass
Let `tril(·, k)` be lower-triangular including the \(k\)-th diagonal, `diag(β)` the diagonal from vector β.

For chunk `i`:

```math
A = I + \operatorname{tril}\!\big(\operatorname{diag}(\beta)\, K K^\top,\,-1\big)
```

```math
W = A^{-1}\big(\operatorname{diag}(\beta)\, K\big)
```

```math
U = A^{-1}\big(\operatorname{diag}(\beta)\, V\big)
```

```math
X = U - W S^\top
```

```math
O = Q S^\top + \operatorname{tril}\!\big(Q K^\top,\,0\big)\, X
```

```math
S \leftarrow S + X^\top K
```

### Backward pass

For the math, refer ![here](assets/deltanet-math.pdf).

