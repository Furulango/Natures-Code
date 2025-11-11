# dtwin/optim/utils.py
import torch

def clamp_params(params, lower, upper):
    return torch.max(torch.min(params, upper), lower)

def masked_params(base_params, candidate_delta, mask):
    # actualiza solo dimensiones con mask=1
    return base_params * (1-mask) + (base_params + candidate_delta) * mask

def build_bounds(mcfg, device, B):
    lower = torch.tensor([mcfg.rs_bounds[0], mcfg.rr_bounds[0], mcfg.lls_bounds[0], mcfg.llr_bounds[0], mcfg.lm_bounds[0], mcfg.j_bounds[0], mcfg.b_bounds[0]],
                         device=device).unsqueeze(0).repeat(B,1)
    upper = torch.tensor([mcfg.rs_bounds[1], mcfg.rr_bounds[1], mcfg.lls_bounds[1], mcfg.llr_bounds[1], mcfg.lm_bounds[1], mcfg.j_bounds[1], mcfg.b_bounds[1]],
                         device=device).unsqueeze(0).repeat(B,1)
    return lower, upper
