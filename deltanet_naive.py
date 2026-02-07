# v_old = S_{t-1} k_t
# S_t <- S_{t-1} - beta_t * (v_old - v_t) k_t^T
# o_t = S_t q_t

import torch

def deltanet_forward_naive(q, k, v, beta):
    assert q.shape == k.shape and q.shape == v.shape, "shapes must match!!"
    assert q.ndim >= 2, "increase dims!!"
    
    *batch, L, D = q.shape #bLD

    assert beta.shape[-1] == L, "beta has wrong dims!!"
    beta = beta.expand(*batch, L) #bL
    
    S = torch.zeros(*batch, D, D, dtype=q.dtype, device=q.device) #bDD
    
    outs = []
    
    for i in range(L):
        qi = q[..., i, :] #bD
        ki = k[..., i, :]
        vi = v[..., i, :]
        bi = beta[..., i] #b
        
        vold = torch.einsum("...de,...e->...d", S, ki) #bD
        S = S - bi[..., None, None] * torch.einsum("...d,...e->...de", (vold-vi), ki) #bDD
        
        outs.append(torch.einsum("...de,...e->...d", S, qi)) #bD

    return torch.stack(outs, dim=-2) #bLD
    