# dtwin/scenarios/temperature.py
import torch

def apply_temperature(params, temp_c, coeffs, tref=20.0):
    """
    params: [B,7] base
    temp_c: escalar o [B]
    coeffs: dict con alfas por índice (0:Rs,1:Rr,4:Lm)
    """
    B = params.shape[0]
    dT = (temp_c - tref).reshape(-1) if isinstance(temp_c, torch.Tensor) else torch.tensor([temp_c - tref], device=params.device, dtype=params.dtype).repeat(B)
    P = params.clone()
    if 0 in coeffs:
        P[:,0] = params[:,0] * (1.0 + coeffs[0]*dT)
    if 1 in coeffs:
        P[:,1] = params[:,1] * (1.0 + coeffs[1]*dT)
    if 4 in coeffs:
        P[:,4] = params[:,4] * (1.0 + coeffs[4]*dT)
    return P
