# scripts/generate_csv_data.py
import argparse
from dtwin.data.synthetic import generate_stage1_csv, generate_stage2_hot_csv
from dtwin.config import paths

def main():
    parser = argparse.ArgumentParser(description="Generador de CSV sintéticos para Stage 1 y Stage 2 (alta temperatura)")
    parser.add_argument("--dur", type=float, default=1.0, help="Duración en segundos")
    parser.add_argument("--dt", type=float, default=1e-3, help="Paso de integración/registro")
    parser.add_argument("--v_amp", type=float, default=None, help="Amplitud vqs, por defecto 220*sqrt(2)/sqrt(3)")
    parser.add_argument("--tl", type=float, default=0.5, help="Par de carga constante en N·m")
    parser.add_argument("--hot", type=float, default=80.0, help="Temperatura Stage 2 (°C)")
    parser.add_argument("--no-noise", action="store_true", help="Desactivar ruido")
    args = parser.parse_args()

    print("Generando Stage 1...")
    p1 = generate_stage1_csv(out_path=paths.data_stage1_csv, duration_s=args.dur, dt=args.dt, v_amp=args.v_amp, torque_val=args.tl, noise=not args.no_noise)
    print(f"CSV Stage 1 -> {p1}")

    print("Generando Stage 2 (alta temperatura)...")
    p2 = generate_stage2_hot_csv(out_path=paths.data_stage2_csv, temp_c=args.hot, duration_s=args.dur, dt=args.dt, v_amp=args.v_amp, torque_val=args.tl, noise=not args.no_noise)
    print(f"CSV Stage 2 -> {p2}")

if __name__ == "__main__":
    main()
