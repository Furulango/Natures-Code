# dtwin/data/synthetic.py
import os
import math
import pandas as pd
import torch

from dtwin.config import motor_cfg, thermal, paths
from dtwin.sim.simulator import simulate_time_series
from dtwin.scenarios.temperature import apply_temperature

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def build_true_params(device="cpu", dtype=torch.float32):
    """
    Vector verdadero base [Rs, Rr, Lls, Llr, Lm, J, B]
    Tomado del ejemplo académico para un motor ~2 HP a 60 Hz.
    """
    base = [2.45, 1.83, 0.008, 0.008, 0.203, 0.02, 0.001]
    return torch.tensor(base, device=device, dtype=dtype).unsqueeze(0)  # [1,7]

def make_vqd_series(S, device="cpu", dtype=torch.float32, v_amp=None):
    """
    Señal síncrona con vqs constante y vds=0.
    v_amp por defecto ~ 220*sqrt(2)/sqrt(3) (fase pico supuesto).
    """
    if v_amp is None:
        v_amp = 220.0 * math.sqrt(2.0) / math.sqrt(3.0)
    vqs = torch.full((1, S), float(v_amp), device=device, dtype=dtype)
    vds = torch.zeros(1, S, device=device, dtype=dtype)
    vqd = torch.stack([vqs, vds], dim=2)  # [1,S,2]
    return vqd

def make_load_series(S, device="cpu", dtype=torch.float32, kind="constant", torque_val=0.5, fan_k=1e-4):
    """
    Perfila el par de carga por paso de tiempo.
    - constant: TL = torque_val
    - fan: TL = k * wr^2 (se reevalúa fuera si se desea lazo)
    Aquí: devolvemos TL constante para simplificar generación de dataset reproducible.
    """
    if kind == "constant":
        TL = torch.full((1, S), float(torque_val), device=device, dtype=dtype)
    else:
        TL = torch.full((1, S), float(torque_val), device=device, dtype=dtype)
    return TL

def add_noise(meas_dict, seed=123, sigma_i_frac=0.01, sigma_torque=0.02, sigma_rpm=5.0):
    g = torch.Generator(device=meas_dict["iqs"].device)
    g.manual_seed(seed)

    iqs = meas_dict["iqs"].clone()
    ids = meas_dict["ids"].clone()
    torque = meas_dict["torque"].clone()
    rpm = meas_dict["rpm"].clone()

    i_mag = torch.sqrt(iqs**2 + ids**2)
    std_i = torch.std(i_mag, dim=1, keepdim=True)       # [B,1]
    std_iqs = (sigma_i_frac * std_i).expand_as(iqs)     # [B,S]
    std_ids = (sigma_i_frac * std_i).expand_as(ids)     # [B,S]

    # randn_like no soporta 'generator' en tu versión → usa torch.randn con shape + generator
    noise_iqs = torch.randn(iqs.shape, generator=g, device=iqs.device, dtype=iqs.dtype) * std_iqs
    noise_ids = torch.randn(ids.shape, generator=g, device=ids.device, dtype=ids.dtype) * std_ids
    noise_T   = torch.randn(torque.shape, generator=g, device=torque.device, dtype=torque.dtype) * sigma_torque
    noise_r   = torch.randn(rpm.shape, generator=g, device=rpm.device, dtype=rpm.dtype) * sigma_rpm

    iqs += noise_iqs
    ids += noise_ids
    torque += noise_T
    rpm += noise_r

    out = dict(meas_dict)
    out["iqs"], out["ids"], out["torque"], out["rpm"] = iqs, ids, torque, rpm
    return out


def simulate_measurement(params, duration_s=1.0, dt=1e-3, v_amp=None, torque_load_kind="constant", torque_val=0.5, device="cpu"):
    """
    Simula una medición con el modelo DQ y retorna dict con tensores [1,S].
    """
    S = int(duration_s / dt)
    vqd = make_vqd_series(S, device=device, v_amp=v_amp)
    TL = make_load_series(S, device=device, kind=torque_load_kind, torque_val=torque_val)
    sim = simulate_time_series(params, vqd, TL, fe_hz=motor_cfg.fe_hz, poles=motor_cfg.poles, dt=dt, device=device)
    out = {
        "time": torch.linspace(0.0, duration_s - dt, S, device=device).unsqueeze(0),
        "iqs": sim["states"][:,:,0],
        "ids": sim["states"][:,:,1],
        "torque": sim["torque"],
        "rpm": sim["rpm"],
        "vqs": vqd[:,:,0],
        "vds": vqd[:,:,1],
    }
    return out

def to_dataframe(meas_dict):
    """
    Convierte dict [1,S] a DataFrame con columnas esperadas.
    """
    t = meas_dict["time"].squeeze(0).detach().cpu().numpy()
    iqs = meas_dict["iqs"].squeeze(0).detach().cpu().numpy()
    ids = meas_dict["ids"].squeeze(0).detach().cpu().numpy()
    T = meas_dict["torque"].squeeze(0).detach().cpu().numpy()
    rpm = meas_dict["rpm"].squeeze(0).detach().cpu().numpy()
    vqs = meas_dict["vqs"].squeeze(0).detach().cpu().numpy()
    vds = meas_dict["vds"].squeeze(0).detach().cpu().numpy()
    df = pd.DataFrame({
        "time": t,
        "iqs": iqs,
        "ids": ids,
        "torque": T,
        "rpm": rpm,
        "vqs": vqs,
        "vds": vds,
    })
    return df

def generate_stage1_csv(out_path=None, duration_s=1.0, dt=1e-3, v_amp=None, torque_val=0.5, noise=True, seed=123):
    """
    Genera dataset Stage 1 (condición base ~20°C).
    """
    device = "cpu"
    params_true = build_true_params(device=device)
    meas = simulate_measurement(params_true, duration_s=duration_s, dt=dt, v_amp=v_amp, torque_val=torque_val, device=device)
    if noise:
        meas = add_noise(meas, seed=seed, sigma_i_frac=0.01, sigma_torque=0.02, sigma_rpm=5.0)
    df = to_dataframe(meas)
    if out_path is None:
        out_path = paths.data_stage1_csv
    _ensure_dir(out_path)
    df.to_csv(out_path, index=False)
    return out_path

def generate_stage2_hot_csv(out_path=None, temp_c=80.0, duration_s=1.0, dt=1e-3, v_amp=None, torque_val=0.5, noise=True, seed=321):
    """
    Genera dataset Stage 2 (alta temperatura) a partir de parámetros base con compensación térmica.
    """
    device = "cpu"
    base = build_true_params(device=device)
    coeffs = {0: thermal.rs_alpha, 1: thermal.rr_alpha, 4: thermal.lm_alpha}
    params_hot = apply_temperature(base, temp_c=temp_c, coeffs=coeffs, tref=thermal.tref_c)
    meas = simulate_measurement(params_hot, duration_s=duration_s, dt=dt, v_amp=v_amp, torque_val=torque_val, device=device)
    if noise:
        meas = add_noise(meas, seed=seed, sigma_i_frac=0.015, sigma_torque=0.03, sigma_rpm=7.0)
    df = to_dataframe(meas)
    if out_path is None:
        out_path = paths.data_stage2_csv
    _ensure_dir(out_path)
    df.to_csv(out_path, index=False)
    return out_path
