# dtwin/pipeline/stage2_adaptation.py
import torch
from ..config import motor_cfg, loss_w
from ..sim.simulator import simulate_time_series
from ..objective.losses import multichannel_loss
from ..optim.pso_torch import PSO
from ..optim.utils import build_bounds

def run_stage2(meas_hot, params_stage1, dt=1e-4, device="cuda", mask_adapt=(1,1,0,0,1,0,0)):
    mask = torch.tensor(mask_adapt, device=device, dtype=torch.float32)
    base = params_stage1.to(device).unsqueeze(0).repeat(256,1)
    lower, upper = build_bounds(motor_cfg, device, B=256)

    # Restringe búsqueda a dimensiones con mask=1 alrededor de base
    span = 0.4*(upper - lower)[0]
    low_local = torch.maximum(lower[0], base - span*mask)
    up_local  = torch.minimum(upper[0], base + span*mask)

    def obj_fn_torch(pop_params):
        P = base.clone()
        # pisa solo dimensiones adaptables
        P = P * (1-mask) + pop_params * mask
        B = P.shape[0]
        vqd_series = meas_hot["vqd"].to(device).repeat(B,1,1)
        TL_series = torch.zeros(B, meas_hot["time"].shape[1], device=device, dtype=torch.float32)
        out = simulate_time_series(P, vqd_series, TL_series, fe_hz=motor_cfg.fe_hz, poles=motor_cfg.poles, dt=dt, device=device)
        loss = multichannel_loss(out, {k: v.to(device).repeat(B,1) for k,v in {"iqs":meas_hot["iqs"],"ids":meas_hot["ids"],"torque":meas_hot["torque"],"rpm":meas_hot["rpm"]}.items()},
                                 loss_w, P, penalty_scale=loss_w.penalty_phys)
        return loss

    pso = PSO(dim=7, bounds=(low_local, up_local), swarm_size=256)
    gbest, gfit = pso.optimize(lambda P: obj_fn_torch(P))
    # combina con base
    adapted = base[0]*(1-mask) + gbest*mask
    return adapted.detach(), gfit.detach()
