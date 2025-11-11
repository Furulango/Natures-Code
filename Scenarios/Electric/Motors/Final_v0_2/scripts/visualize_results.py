# scripts/visualize_results.py
import os, json
import torch
import matplotlib.pyplot as plt
from dtwin.config import paths, motor_cfg
from dtwin.data.io import load_measurements_csv
from dtwin.sim.simulator import simulate_time_series

def plot_series(time, y_mea, y_sim, title):
    plt.figure()
    plt.plot(time, y_mea, label="medido")
    plt.plot(time, y_sim, label="simulado", alpha=0.8)
    plt.title(title)
    plt.legend()
    plt.grid(True)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(paths.out_dir, "stage1_params.json")) as f:
        d = json.load(f)
    params = torch.tensor(d["params"], device=device).unsqueeze(0)
    meas = load_measurements_csv(paths.data_stage1_csv, device=device)
    S = meas["time"].shape[1]
    vqd = meas["vqd"]
    TL = torch.zeros(1, S, device=device)
    sim = simulate_time_series(params, vqd, TL, fe_hz=motor_cfg.fe_hz, poles=motor_cfg.poles, dt=1e-4, device=device)
    t = meas["time"].squeeze(0).cpu().numpy()
    plot_series(t, (meas["iqs"].squeeze(0)**2+meas["ids"].squeeze(0)**2).sqrt().cpu().numpy(), sim["i_mag"].squeeze(0).cpu().numpy(), "Corriente |Is|")
    plot_series(t, meas["torque"].squeeze(0).cpu().numpy(), sim["torque"].squeeze(0).cpu().numpy(), "Torque")
    plot_series(t, meas["rpm"].squeeze(0).cpu().numpy(), sim["rpm"].squeeze(0).cpu().numpy(), "RPM")
    plt.show()

if __name__ == "__main__":
    main()
