# dtwin/optim/pso_torch.py
import torch
import math
from .utils import clamp_params

class PSO:
    def __init__(self, dim, bounds, swarm_size=256, iters=400, w=0.72, c1=1.49, c2=1.49, vmax_frac=0.2, device="cuda", seed=42):
        self.dim = dim
        self.lower, self.upper = bounds
        self.S = swarm_size
        self.iters = iters
        self.w, self.c1, self.c2 = w, c1, c2
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.vmax = (self.upper - self.lower) * vmax_frac
        self.vmin = -self.vmax

    def optimize(self, objective_fn):
        # Inicialización
        pos = self.lower + (self.upper - self.lower) * torch.rand(self.S, self.dim, device=self.device, generator=self.gen)
        vel = torch.zeros_like(pos, device=self.device)
        fit = objective_fn(pos)  # [S]
        pbest = pos.clone()
        pbest_fit = fit.clone()
        gbest_fit, idx = torch.min(pbest_fit, dim=0)
        gbest = pbest[idx].clone()

        for it in range(self.iters):
            r1 = torch.rand(pos.shape, device=self.device, dtype=pos.dtype, generator=self.gen)
            r2 = torch.rand(pos.shape, device=self.device, dtype=pos.dtype, generator=self.gen)
            
            vel = self.w*vel + self.c1*r1*(pbest - pos) + self.c2*r2*(gbest.unsqueeze(0) - pos)
            vel = torch.max(torch.min(vel, self.vmax), self.vmin)
            pos = pos + vel
            pos = clamp_params(pos, self.lower, self.upper)

            fit = objective_fn(pos)
            improved = fit < pbest_fit
            pbest[improved] = pos[improved]
            pbest_fit[improved] = fit[improved]
            gbest_fit_new, idx = torch.min(pbest_fit, dim=0)
            if gbest_fit_new < gbest_fit:
                gbest_fit = gbest_fit_new
                gbest = pbest[idx].clone()

        return gbest, gbest_fit
