# dtwin/pipeline/stage1_identification.py
import torch
from ..config import motor_cfg, loss_w
from ..sim.simulator import simulate_time_series
from ..objective.losses import multichannel_loss
from ..optim.pso_torch import PSO
from ..optim.utils import build_bounds

def build_objective_stage1(meas, dt, device):
    B = 1
    S = meas["time"].shape[1]
    vqd_series = meas["vqd"].to(device)
    TL_series = torch.zeros(B, S, device=device, dtype=torch.float32)  # ajusta si tienes carga medida
    tvec = meas["time"].to(device)
    poles = motor_cfg.poles
    fe_hz = motor_cfg.fe_hz

    def obj_fn_torch(pop_params):
        # pop_params: [S,7] S=swarm_size
        swarm = pop_params.shape[0]
        params = pop_params  # [S,7]
        params_batched = params  # B==S
        vqd_rep = vqd_series.repeat(swarm,1,1)
        TL_rep = TL_series.repeat(swarm,1)
        out = simulate_time_series(params_batched, vqd_rep, TL_rep, fe_hz, poles, dt, device)
        loss = multichannel_loss(out, {k: v.repeat(swarm,1) if k in ["iqs","ids","torque","rpm"] else v for k,v in {"iqs":meas["iqs"].to(device),
                                                                                                                    "ids":meas["ids"].to(device),
                                                                                                                    "torque":meas["torque"].to(device),
                                                                                                                    "rpm":meas["rpm"].to(device)}.items()},
                                 loss_w, params_batched, penalty_scale=loss_w.penalty_phys)
        return loss  # [S]
    return obj_fn_torch

def run_stage1(meas, dt=1e-4, device="cuda"):
    lower, upper = build_bounds(motor_cfg, device, B=256)
    pso = PSO(dim=7, bounds=(lower[0], upper[0]), swarm_size=256)
    obj = build_objective_stage1(meas, dt, device)
    gbest, gfit = pso.optimize(lambda P: obj(P))
    return gbest.detach(), gfit.detach()
