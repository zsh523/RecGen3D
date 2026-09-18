import torch
from datetime import datetime

@torch.no_grad()  # remove this if you want gradients through the fit
def umeyama_similarity(A: torch.Tensor, B: torch.Tensor, weights: torch.Tensor=None, eps: float=1e-8, tensor_point: torch.Tensor=None,
                       confidence_check: bool=False, min_confidence: float=0.90):
    """
    Compute batched 4x4 similarity transforms T s.t. B ≈ T @ A  (homogeneous coords).
    A, B: (B, N, 3)
    weights: optional (B, N) nonnegative weights (0 = ignore point)
    confidence_check: if True, return (None, None, None, None, B, confidence) when the fit
                      is untrustworthy -- either because confidence < min_confidence or
                      because the fit could not be computed at all.
    Returns:
      T: (B, 4, 4) with top-left = s*R, top-right = t, last row [0,0,0,1]
      s: (B,) uniform scales
      R: (B, 3, 3) rotations
      t: (B, 3) translations
      B_hat: (B, N, 3) transformed A

    A degenerate or numerically unusable fit is reported the same way as a
    low-confidence one, so callers need only one fallback path.
    """
    assert A.shape == B.shape and A.dim() == 3 and A.size(-1) == 3
    scale_factor = A.std(dim=1, keepdim=True).mean(dim=2, keepdim=True) + eps
    A = A / scale_factor
    Bsz, N, _ = A.shape
    device = A.device
    dtype  = A.dtype

    def failure_return(msg):
        """
        Bail out when the fit cannot be computed.

        Under confidence_check this signals failure exactly like a low-confidence
        fit does -- T is None and B is handed back untransformed -- so the caller
        takes the same fallback path in both cases. An identity transform would be
        worse than useless here: it would hand back points that are still in the
        reference frame while claiming they are aligned.

        Without confidence_check the caller expects a usable transform and has no
        fallback, so return the identity.
        """
        print(msg)
        if confidence_check:
            confidence = torch.zeros(Bsz, dtype=dtype, device=device)
            return None, None, None, None, B, confidence
        T = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).expand(Bsz, 4, 4).clone()
        s = torch.ones(Bsz, dtype=dtype, device=device)
        R = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(Bsz, 3, 3).clone()
        t = torch.zeros((Bsz, 3), dtype=dtype, device=device)
        return T, s, R, t, A.clone()

    try:
        if weights is None:
            # centroids
            muA = A.mean(dim=1, keepdim=True)            # (B,1,3)
            muB = B.mean(dim=1, keepdim=True)
            # demean
            A0 = A - muA
            B0 = B - muB
            # covariance
            C = A0.transpose(1, 2) @ B0 / N              # (B,3,3)
            # variance of A
            varA = (A0.square().sum(dim=(1,2)) / N)      # (B,)
        else:
            w = torch.clamp_min(weights, 0.0)            # (B,N)
            wsum = w.sum(dim=1, keepdim=True) + eps      # (B,1)
            wn = w / wsum                                # normalized per batch (B,N)
            # weighted centroids
            muA = (A * wn.unsqueeze(-1)).sum(dim=1, keepdim=True)
            muB = (B * wn.unsqueeze(-1)).sum(dim=1, keepdim=True)
            # demean
            A0 = A - muA
            B0 = B - muB
            # weighted covariance
            C = (A0 * wn.unsqueeze(-1)).transpose(1, 2) @ B0  # (B,3,3)
            # weighted variance of A
            varA = (A0.square() * wn.unsqueeze(-1)).sum(dim=(1,2))  # (B,)

        # Check for degenerate variance (all points collapsed)
        if torch.any(varA < eps):
            # debug
            degenerate_mask = varA < eps
            degenerate_indices = torch.where(degenerate_mask)[0].cpu().tolist()
            
            print(f"Degenerate case detected in batch items: {degenerate_indices}")
            print(f"Variance values: {varA[degenerate_mask].cpu().tolist()}")
            
            ## Save problematic A tensors to obj files
            
            #for idx in degenerate_indices:
                
                ## Use timestamp + process ID for unique filenames
                
                
                ## Write A to OBJ file
                    #for point in A_problem:
                
                ## Write B to OBJ file
                    #for point in B_problem:
                
                
                ## Also print statistics about the problematic point clouds
            
            return failure_return("Procrustes: degenerate input; falling back to the unaligned point cloud.")

        # Check for NaN or Inf in covariance matrix
        if not torch.isfinite(C).all():
            return failure_return("Procrustes: non-finite covariance; falling back to the unaligned point cloud.")

        # SVD of covariance - keep in original dtype
        U, S, Vh = torch.linalg.svd(C.to(dtype=torch.float32))                   # U @ diag(S) @ Vh
        
        # Check for NaN or Inf in SVD results
        if not (torch.isfinite(U).all() and torch.isfinite(S).all() and torch.isfinite(Vh).all()):
            return failure_return("Procrustes: SVD did not converge; falling back to the unaligned point cloud.")
        
        V = Vh.transpose(-2, -1)

        # Ensure right-handed rotation (avoid reflection)
        # Check det(V @ U^T) to decide if we need reflection correction
        det = torch.det((V @ U.transpose(-2, -1)).to(dtype=torch.float32))         # (B,)
        D = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(Bsz, 3, 3).clone()
        D[:, 2, 2] = torch.where(det < 0, torch.tensor(-1.0, dtype=dtype, device=device), 
                                          torch.tensor(1.0, dtype=dtype, device=device))

        # Rotation
        R = V @ D @ U.transpose(-2, -1)                  # (B,3,3)
        
        # tr(D * diag(S)) = S[0] + S[1] + D[2,2] * S[2]
        s = (S[:, 0] + S[:, 1] + D[:, 2, 2] * S[:, 2]) / (varA + eps)

        # Check for invalid scale
        if not torch.isfinite(s).all() or torch.any(s <= 0):
            return failure_return("Procrustes: invalid scale; falling back to the unaligned point cloud.")

        # Translation
        t = muB.squeeze(1) - s.unsqueeze(-1) * (R @ muA.squeeze(1).unsqueeze(-1)).squeeze(-1)  # (B,3)

        # Check for NaN or Inf in translation
        if not torch.isfinite(t).all():
            return failure_return("Procrustes: non-finite translation; falling back to the unaligned point cloud.")

        # Build 4x4
        T = torch.zeros((Bsz, 4, 4), dtype=dtype, device=device)
        T[:, :3, :3] = s.view(-1, 1, 1) * R
        T[:, :3, 3]  = t
        T[:, 3, 3]   = 1.0

        # Apply to A -> B_hat
        B_hat = s.view(-1, 1, 1) * (A @ R.transpose(1, 2)) + t.unsqueeze(1)   # (B,N,3)

        # Final sanity check
        if not torch.isfinite(B_hat).all():
            return failure_return("Procrustes: non-finite result; falling back to the unaligned point cloud.")

        # Optional confidence check (per batch item) based on post-fit RMSE.
        if confidence_check:
            diff = B_hat - B  # (B,N,3)
            sq = diff.square().sum(dim=2)  # (B,N)
            if weights is None:
                mse = sq.mean(dim=1)
            else:
                w = torch.clamp_min(weights, 0.0)
                wsum = w.sum(dim=1) + eps
                mse = (w * sq).sum(dim=1) / wsum

            rmse = torch.sqrt(mse + eps)  # (B,)
            b_scale = B.std(dim=1).mean(dim=1) + eps  # (B,)
            confidence = 1.0 / (1.0 + (rmse / b_scale))  # (B,)
            print(f"Procrustes confidence scores: {confidence.cpu().tolist()}")

            if torch.any(confidence < min_confidence):
                return None, None, None, None, B, confidence

        # Adjust scale to account for the initial normalization of A
        s = s / scale_factor
        T[:, :3, :3] /= scale_factor

        if confidence_check:
            return T, s, R, t, B_hat, confidence
        else:
            return T, s, R, t, B_hat
    
    except Exception as e:
        # Catch any other errors (SVD convergence failure, etc.)
        print(f"Error in umeyama_similarity: {e}")
        return failure_return("Procrustes: fit failed; falling back to the unaligned point cloud.")