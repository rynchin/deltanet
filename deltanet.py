import torch

def deltanet_forward(q, k, v, beta, c=128):
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

        # (1) A = I + tril(diag(beta) K K^T, -1)
        G = K @ K.transpose(-1, -2) #bcc
        A = torch.tril(B[..., :, None] * G, diagonal=-1) #bcc
        I = torch.eye(c, dtype=q.dtype, device=q.device).expand(*batch, c, c) #bcc
        A = A + I

        # (2) B_K = diag(beta) K, B_V = diag(beta) V
        B_K = B[..., :, None] * K #bcD
        B_V = B[..., :, None] * V #bcD

        # (3) W = A^{-1} B_K, U = A^{-1} B_V
        W = torch.linalg.solve_triangular(A, B_K, upper=False, unitriangular=True) #bcD
        U = torch.linalg.solve_triangular(A, B_V, upper=False, unitriangular=True) #bcD

        # (4) X = U - W S^T
        X = U - W @ S.transpose(-1, -2) #bcD

        # (5) P = tril(Q K^T, 0)
        P = (Q @ K.transpose(-1, -2)) * M #bcc
        
        # (6) O = Q S^T + P X
        base = Q @ S.transpose(-1, -2) #bcD
        O = base + P @ X #bcD

        # (7) S <- S + X^T K
        S = S + X.transpose(-1, -2) @ K #bDD

        outs.append(O)

    out = torch.reshape(torch.stack(outs, dim=-3), (*batch, n*c, D))[..., :L, :] #bLD
    return out

def deltanet_backward(q, k, v, beta, grad_output, c=128):
    assert q.shape == k.shape and q.shape == v.shape, "shapes must match!!"
    assert q.ndim >= 2, "increase dims!!"
    
    *batch, L, D = q.shape
    beta = beta.expand(*batch, L)
    n = (L + c - 1) // c
    pad = n * c - L
    if pad:
        padder = torch.zeros(*batch, pad, D, dtype=q.dtype, device=q.device)
        q = torch.cat([q, padder], dim=-2)
        k = torch.cat([k, padder], dim=-2)
        v = torch.cat([v, padder], dim=-2)
        b_padder = torch.zeros(*batch, pad, dtype=beta.dtype, device=beta.device)
        beta = torch.cat([beta, b_padder], dim=-1)
        grad_padder = torch.zeros(*batch, pad, D, dtype=grad_output.dtype, device=grad_output.device)
        grad_output = torch.cat([grad_output, grad_padder], dim=-2)
    
    qc = q.reshape(*batch, n, c, D)
    kc = k.reshape(*batch, n, c, D)
    vc = v.reshape(*batch, n, c, D)
    bc = beta.reshape(*batch, n, c)
    doc = grad_output.reshape(*batch, n, c, D)
    
    M = torch.tril(torch.ones(c, c, dtype=q.dtype, device=q.device), diagonal=0)
    
    # recompute forward
    S_list = []
    W_list = []
    U_list = []
    X_list = []
    A_list = []
    P_list = []
    
    S = torch.zeros(*batch, D, D, dtype=q.dtype, device=q.device)
    for i in range(n):
        Q = qc[..., i, :, :]
        K = kc[..., i, :, :]
        V = vc[..., i, :, :]
        B = bc[..., i, :]
        
        S_list.append(S.clone())
        
        G = K @ K.transpose(-1, -2)
        A = torch.tril(B[..., :, None] * G, diagonal=-1)
        I = torch.eye(c, dtype=q.dtype, device=q.device).expand(*batch, c, c)
        A = A + I
        
        W = torch.linalg.solve_triangular(A, B[..., :, None] * K, upper=False, unitriangular=True)
        U = torch.linalg.solve_triangular(A, B[..., :, None] * V, upper=False, unitriangular=True)
        X = U - W @ S.transpose(-1, -2)
        P = (Q @ K.transpose(-1, -2)) * M
        
        A_list.append(A)
        W_list.append(W)
        U_list.append(U)
        X_list.append(X)
        P_list.append(P)
        
        S = S + X.transpose(-1, -2) @ K
    
    # backward pass
    dq = torch.zeros_like(qc) #bncD
    dk = torch.zeros_like(kc)
    dv = torch.zeros_like(vc)
    dbeta = torch.zeros_like(bc) #bnc
    dS = torch.zeros(*batch, D, D, dtype=q.dtype, device=q.device) #bDD
    
    for i in range(n - 1, -1, -1): # iterate over chunks
        Q = qc[..., i, :, :]
        K = kc[..., i, :, :]
        V = vc[..., i, :, :]
        B = bc[..., i, :]
        dO = doc[..., i, :, :]
        S = S_list[i]
        W = W_list[i]
        U = U_list[i]
        X = X_list[i]
        A = A_list[i]
        P = P_list[i]
        
        # (7) S' = S + X^T K, dS is upstream gradient
        dX = K @ dS.transpose(-1, -2) #bcD
        dK_i = X @ dS #bcD
        
        # (6) O = Q S^T + P X
        dQ_i = dO @ S #bcD
        dS = dS + dO.transpose(-1, -2) @ Q #bDD
        dP = dO @ X.transpose(-1, -2) #bcc
        dX = dX + P.transpose(-1, -2) @ dO
        
        # (5) P = tril(Q K^T, 0) = (Q K^T) * M
        dT = dP * M #bcc
        dQ_i = dQ_i + dT @ K
        dK_i = dK_i + dT.transpose(-1, -2) @ Q
        
        # (4) X = U - W S^T
        dU = dX #bcD
        dW = -dX @ S #bcD
        dS = dS - dX.transpose(-1, -2) @ W
        
        # (3) W = A^{-1} B_K, U = A^{-1} B_V
        bK = torch.linalg.solve_triangular(A.transpose(-1, -2), dW, upper=True, unitriangular=True) #bcD
        bV = torch.linalg.solve_triangular(A.transpose(-1, -2), dU, upper=True, unitriangular=True) #bcD
        dA = -bK @ W.transpose(-1, -2) - bV @ U.transpose(-1, -2) #bcc

        # (2) B_K = diag(beta) K, B_V = diag(beta) V
        dK_i = dK_i + B[..., :, None] * bK
        dV_i = B[..., :, None] * bV #bcD
        dbeta_i = (bK * K).sum(-1) + (bV * V).sum(-1) #bc
        
        # (1) A = I + tril(diag(beta) K K^T, -1)
        dA = torch.tril(dA, diagonal=-1)

        G = K @ K.transpose(-1, -2)
        dbeta_i = dbeta_i + (dA * G).sum(-1)
        dG = dA * B[..., :, None] #bcc
        dK_i = dK_i + (dG + dG.transpose(-1,-2)) @ K
        
        dq[..., i, :, :] = dQ_i
        dk[..., i, :, :] = dK_i
        dv[..., i, :, :] = dV_i
        dbeta[..., i, :] = dbeta_i
    
    dq = dq.reshape(*batch, n * c, D)[..., :L, :] #bLD
    dk = dk.reshape(*batch, n * c, D)[..., :L, :] #bLD
    dv = dv.reshape(*batch, n * c, D)[..., :L, :] #bLD
    dbeta = dbeta.reshape(*batch, n * c)[..., :L] #bL
    return dq, dk, dv, dbeta