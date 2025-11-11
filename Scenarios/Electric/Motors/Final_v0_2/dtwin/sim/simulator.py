# dtwin/sim/simulator.py
import torch
from ..model.motor_dq import dq_step
from ..model.integrators import integrate_time

def simulate_time_series(params, vqd_series, TL_series, fe_hz, poles, dt, device, state0=None):
    """
    params: [B,7]
    vqd_series: [B,S,2]
    TL_series: [B,S] (par de carga por paso) o escalar broadcast
    state0: [B,5] o None -> ceros
    return: dict con states[B,S,5], torque[B,S], i_mag[B,S], rpm[B,S]
    """
    B, S, _ = vqd_series.shape
    if state0 is None:
        state0 = torch.zeros(B, 5, device=device, dtype=params.dtype)
    def rhs_factory(t_idx_holder):
        # t_idx_holder será un cierre para seleccionar vqd/TL por índice actual
        def rhs(state):
            t = t_idx_holder["t"]
            vqd = vqd_series[:, t, :]
            TL = TL_series[:, t] if TL_series.dim()==2 else TL_series
            dst, Te = dq_step(state, params, vqd, fe_hz, poles, TL, dt=0.0)  # dt=0 aquí; RHS no usa dt
            return dst, Te
        return rhs

    states = torch.zeros(B, S, 5, device=device, dtype=params.dtype)
    torques = torch.zeros(B, S, device=device, dtype=params.dtype)

    s = state0
    t_holder = {"t": 0}
    for t in range(S):
        t_holder["t"] = t
        rhs = rhs_factory(t_holder)
        s, Te = integrate_time_step(s, rhs, dt)  # pequeño wrapper debajo
        states[:, t, :] = s
        torques[:, t] = Te

    iqs = states[:,:,0]
    ids = states[:,:,1]
    i_mag = torch.sqrt(iqs**2 + ids**2)
    wr = states[:,:,4]
    rpm = wr * (60.0 / (2.0*torch.pi))
    return {"states": states, "torque": torques, "i_mag": i_mag, "rpm": rpm}

def integrate_time_step(state, rhs, dt):
    # un paso RK4 embebido (idéntico a rk4_step, separado para claridad)
    k1, Te1 = rhs(state)
    k2, _   = rhs(state + 0.5*dt*k1)
    k3, _   = rhs(state + 0.5*dt*k2)
    k4, Te4 = rhs(state + dt*k3)
    next_state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return next_state, Te4
