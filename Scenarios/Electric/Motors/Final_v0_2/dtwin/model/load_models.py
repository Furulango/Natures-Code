# dtwin/model/load_models.py
import torch

def constant_load(B, torque_val, device, dtype):
    return torch.full((B,), float(torque_val), device=device, dtype=dtype)

def fan_load(wr, kf=1e-4):
    # par ~ kf * wr^2 típico de ventilador/bomba centrífuga
    return kf * (wr**2)

def piecewise_load(time_idx, profile, device, dtype):
    # profile: lista [(t_end, torque_value), ...] en índices
    val = 0.0
    cum = 0
    for t_end, tv in profile:
        cum += t_end
        if time_idx < cum:
            val = tv
            break
    return torch.full((1,), float(val), device=device, dtype=dtype).squeeze(0)
