# dtwin/model/integrators.py
import torch

def rk4_step(state, rhs_fn, dt):
    k1, Te1 = rhs_fn(state)
    k2, _   = rhs_fn(state + 0.5*dt*k1)
    k3, _   = rhs_fn(state + 0.5*dt*k2)
    k4, Te4 = rhs_fn(state + dt*k3)
    next_state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    # Retornar también un torque representativo (usar Te4 por simplicidad)
    return next_state, Te4

def integrate_time(state0, rhs_fn, dt, steps):
    B = state0.shape[0]
    S = steps
    D = state0.shape[1]
    device = state0.device
    states = torch.zeros(B, S, D, device=device, dtype=state0.dtype)
    torques = torch.zeros(B, S, device=device, dtype=state0.dtype)
    s = state0
    for t in range(S):
        s, Te = rk4_step(s, rhs_fn, dt)
        states[:, t, :] = s
        torques[:, t] = Te
    return states, torques
