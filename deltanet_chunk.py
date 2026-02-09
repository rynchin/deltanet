# Chunked (with UT transform inside each chunk)
#
# Per chunk i:
#   A = I + tril(diag(beta) K K^T, -1)
#   W = A^{-1} (diag(beta) K)
#   U = A^{-1} (diag(beta) V)
#   X = U - W S^T
#   O = Q S^T + tril(Q K^T, 0) X
#   S <- S + X^T K

import torch

def deltanet_forward_chunk(q, k, v, beta, c=128):
    assert q.shape == k.shape and q.shape == v.shape, "shapes must match!!"
    assert q.ndim >= 2, "increase dims!!"
    
    *batch, L, D = q.shape #bLD

    assert beta.shape[-1] == L, "beta has wrong dims!!"
    beta = beta.expand(*batch, L) #bL

    n = (L + c - 1) // c
    pad = n * c - L

    if pad:
        padder = torch.zeros(*batch, pad, D, dtype=q.dtype, device=q.device)
        q = torch.cat([q, padder], dim=-2) # L is now multiple of chunk_size c
        k = torch.cat([k, padder], dim=-2)
        v = torch.cat([v, padder], dim=-2)

        b_padder = torch.zeros(*batch, pad, dtype=beta.dtype, device=beta.device)
        beta = torch.cat([beta, b_padder], dim=-1)

    # reshape
    qc = q.reshape(*batch, n, c, D) #bncD
    kc = k.reshape(*batch, n, c, D)
    vc = v.reshape(*batch, n, c, D)
    bc = beta.reshape(*batch, n, c) #bnc

    S = torch.zeros(*batch, D, D, dtype=q.dtype, device=q.device) #bDD

    # causal mask per chunk (dont recompute)
    M = torch.tril(torch.ones(c, c, dtype=q.dtype, device=q.device), diagonal=0) #cc

    outs = []

    for i in range(n):
        Q = qc[..., i, :, :] #bcD
        K = kc[..., i, :, :]
        V = vc[..., i, :, :]
        B = bc[..., i, :] #bc

        # KK^T
        G = K @ K.transpose(-1, -2) #bcc

        # A = I + tril(diag(beta) K K^T, -1)
        A = torch.tril(B[..., :, None] * G, diagonal=-1) #bcc
        I = torch.eye(c, dtype=q.dtype, device=q.device).expand(*batch, c, c) #bcc
        A = A + I

        # W = A^{-1} (diag(beta) K)
        W = torch.linalg.solve_triangular(A, B[..., :, None] * K, upper=False, unitriangular=True) #bcD
        # U = A^{-1} (diag(beta) V)
        U = torch.linalg.solve_triangular(A, B[..., :, None] * V, upper=False, unitriangular=True) #bcD

        # X = U - W S^T
        X = U - W @ S.transpose(-1, -2) #bcD

        # O = Q S^T + tril(Q K^T, 0) X
        base = Q @ S.transpose(-1, -2) #bcD
        attn = (Q @ K.transpose(-1, -2)) * M #bcc
        O = base + attn @ X #bcD

        # S <- S + X^T K
        S = S + X.transpose(-1, -2) @ K #bDD

        outs.append(O)

    out = torch.reshape(torch.stack(outs, dim=-3), (*batch, n*c, D))[..., :L, :] #bLD
    return out
        

        