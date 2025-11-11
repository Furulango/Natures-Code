# scripts/adapt_stage2.py
import torch, os, json
from dtwin.config import paths
from dtwin.data.io import load_measurements_csv
from dtwin.pipeline.stage2_adaptation import run_stage2

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(paths.out_dir, "stage1_params.json")) as f:
        d = json.load(f)
    params_stage1 = torch.tensor(d["params"], device=device)
    meas_hot = load_measurements_csv(paths.data_stage2_csv, device=device)
    adapted, fit = run_stage2(meas_hot, params_stage1, dt=1e-4, device=device)
    outp = os.path.join(paths.out_dir, "stage2_params.json")
    with open(outp, "w") as f:
        json.dump({"params": adapted.tolist(), "fitness": float(fit)}, f, indent=2)
    print(f"Guardado: {outp}")

if __name__ == "__main__":
    main()
