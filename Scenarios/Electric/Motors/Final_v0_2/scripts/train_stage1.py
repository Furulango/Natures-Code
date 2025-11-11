# scripts/train_stage1.py
import torch, os, json
from dtwin.config import paths
from dtwin.data.io import load_measurements_csv
from dtwin.pipeline.stage1_identification import run_stage1

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meas = load_measurements_csv(paths.data_stage1_csv, device=device)
    params, fit = run_stage1(meas, dt=1e-4, device=device)
    os.makedirs(paths.out_dir, exist_ok=True)
    outp = os.path.join(paths.out_dir, "stage1_params.json")
    with open(outp, "w") as f:
        json.dump({"params": params.tolist(), "fitness": float(fit)}, f, indent=2)
    print(f"Guardado: {outp}")

if __name__ == "__main__":
    main()
