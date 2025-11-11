# dtwin/data/io.py
import pandas as pd
import torch

def load_measurements_csv(path, device="cuda", dtype=torch.float32):
    df = pd.read_csv(path)
    # columnas esperadas: time, iqs, ids, torque, rpm, vqs, vds
    required = ["time","iqs","ids","torque","rpm","vqs","vds"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Falta columna {c} en {path}")
    t = torch.tensor(df["time"].values, device=device, dtype=dtype)
    iqs = torch.tensor(df["iqs"].values, device=device, dtype=dtype)
    ids = torch.tensor(df["ids"].values, device=device, dtype=dtype)
    torque = torch.tensor(df["torque"].values, device=device, dtype=dtype)
    rpm = torch.tensor(df["rpm"].values, device=device, dtype=dtype)
    vqs = torch.tensor(df["vqs"].values, device=device, dtype=dtype)
    vds = torch.tensor(df["vds"].values, device=device, dtype=dtype)
    # formato [B=1,S,*]
    S = t.shape[0]
    vqd = torch.stack([vqs, vds], dim=1).unsqueeze(0)
    meas = {
        "time": t.unsqueeze(0),
        "iqs": iqs.unsqueeze(0),
        "ids": ids.unsqueeze(0),
        "torque": torque.unsqueeze(0),
        "rpm": rpm.unsqueeze(0),
        "vqd": vqd
    }
    return meas
