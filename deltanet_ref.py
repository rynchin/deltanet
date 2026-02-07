# S_t = sum_{i=1}^t u_i k_i^T
# u_1 = beta_1 v_1
# u_t = beta_t (v_t^new - v_t^old) = beta_t (v_t - S_{t-1}k_t) = beta_t (v_t - sum_{i=1}^{t-1} u_i(k_i^Tk_t))
# o_t = S_t q_t

import torch

def deltanet_forward_ref(q, k, v, beta):
    assert q.shape == k.shape and q.shape == v.shape, "shapes must match!!"
    assert q.ndim >= 2, "increase dims!!"
    
    *batch, L, D = q.shape #bLD

    assert beta.shape[-1] == L, "beta has wrong dims!!"
    beta = beta.expand(*batch, L) #bL
    
    S = torch.zeros(*batch, D, D, dtype=q.dtype, device=q.device) #bDD
    
    outs = []

    k_list = [] # BAD this grows memory
    u_list = [] # BAD this grows memory

    for i in range(L):
        qi = q[..., i, :] #bD
        ki = k[..., i, :]
        vi = v[..., i, :]
        bi = beta[..., i] #b
        
        if i == 0:
            ui = bi[..., None] * vi #bD
        else: 
            # sum_{i=1}^{t-1} u_i(k_i^Tk_t)
            ks = torch.stack(k_list, dim=-2) #biD
            us = torch.stack(u_list, dim=-2) #biD
            
            ktk = torch.einsum("...d,...id->...i", ki, ks) #bi
            p = torch.einsum("...i,...id->...d", ktk, us) #bD
            ui = bi[..., None] * (vi - p) #bD
        
        S = S + torch.einsum("...d,...e->...de", ui, ki) #bDD
        outs.append(torch.einsum("...de,...e->...d", S, qi)) #bD
        
        k_list.append(ki)
        u_list.append(ui)
    
    return torch.stack(outs, dim=-2) #bLD
