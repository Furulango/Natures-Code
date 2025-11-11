# dtwin/model/motor_dq.py
import torch

def electrical_rhs(currents, wr, params, vqd, we, poles):
    """
    currents: [B,4] -> [iqs, ids, iqr, idr]
    wr: [B] mecánica
    params: [B,7] -> [Rs, Rr, Lls, Llr, Lm, J, Bm]
    vqd: [B,2] -> [vqs, vds] en marco síncrono
    we: escalar (rad/s)
    poles: int
    return di/dt: [B,4]
    """
    iqs, ids, iqr, idr = currents.unbind(dim=1)
    Rs, Rr, Lls, Llr, Lm, J, Bm = params.unbind(dim=1)
    Ls = Lls + Lm
    Lr = Llr + Lm
    ws = we - (poles/2.0) * wr  # slip eléctrico
    vqs, vds = vqd.unbind(dim=1)

    # Matriz A * d(i)/dt = b  (A tamaño 4x4 por batch)
    # Derivadas de flujos: dλ = L * di + acoples Lm
    # Ecuaciones:
    # vqs = Rs*iqs + dλqs/dt - we*λds
    # vds = Rs*ids + dλds/dt + we*λqs
    # 0   = Rr*iqr + dλqr/dt - ws*λdr
    # 0   = Rr*idr + dλdr/dt + ws*λqr
    # λqs = Ls*iqs + Lm*iqr; λds = Ls*ids + Lm*idr
    # λqr = Lr*iqr + Lm*iqs; λdr = Lr*idr + Lm*ids

    B = currents.shape[0]
    A = torch.zeros(B,4,4, device=currents.device, dtype=currents.dtype)
    b = torch.zeros(B,4, device=currents.device, dtype=currents.dtype)

    # Construcción de A a partir de derivadas de λ: dλ/dt = L*di/dt + ...
    # dλqs/dt = Ls*diqs + Lm*diqr
    # dλds/dt = Ls*dids + Lm*didr
    # dλqr/dt = Lr*diqr + Lm*diqs
    # dλdr/dt = Lr*didr + Lm*dids

    # E1: vqs = Rs*iqs + (Ls*diqs + Lm*diqr) - we*λds
    A[:,0,0] = Ls
    A[:,0,2] = Lm
    b[:,0] = vqs - Rs*iqs + we*(Ls*ids + Lm*idr)

    # E2: vds = Rs*ids + (Ls*dids + Lm*didr) + we*λqs
    A[:,1,1] = Ls
    A[:,1,3] = Lm
    b[:,1] = vds - Rs*ids - we*(Ls*iqs + Lm*iqr)

    # E3: 0 = Rr*iqr + (Lr*diqr + Lm*diqs) - ws*λdr
    A[:,2,0] = Lm
    A[:,2,2] = Lr
    b[:,2] = -Rr*iqr + ws*(Lr*idr + Lm*ids)

    # E4: 0 = Rr*idr + (Lr*didr + Lm*dids) + ws*λqr
    A[:,3,1] = Lm
    A[:,3,3] = Lr
    b[:,3] = -Rr*idr - ws*(Lr*iqr + Lm*iqs)

    # Resolver A * di = b
    di = torch.linalg.solve(A, b)
    return di

def torque_electromagnetic(currents, params, poles):
    iqs, ids, iqr, idr = currents.unbind(dim=1)
    Rs, Rr, Lls, Llr, Lm, J, Bm = params.unbind(dim=1)
    Te = 1.5 * (poles/2.0) * Lm * (iqs*idr - ids*iqr)
    return Te

def mechanical_rhs(wr, Te, TL, params):
    Rs, Rr, Lls, Llr, Lm, J, Bm = params.unbind(dim=1)
    dwr = (Te - TL - Bm*wr) / J
    return dwr

def dq_step(state, params, vqd, fe_hz, poles, TL, dt):
    """
    state: [B,5] -> [iqs, ids, iqr, idr, wr]
    params: [B,7]
    vqd: [B,2]
    TL: [B] carga mecánica
    """
    we = 2.0 * torch.pi * fe_hz
    currents = state[:, :4]
    wr = state[:, 4]
    di = electrical_rhs(currents, wr, params, vqd, we, poles)
    Te = torque_electromagnetic(currents, params, poles)
    dwr = mechanical_rhs(wr, Te, TL, params)
    dst = torch.cat([di, dwr.unsqueeze(1)], dim=1)
    return dst, Te
