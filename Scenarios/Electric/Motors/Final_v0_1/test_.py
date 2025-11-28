"""
Script para analizar resultados de experimentos

EJECUCIÓN:
    python analyze_results.py --phase phase1
    python analyze_results.py --phase phase1 --plot
    python analyze_results.py --algorithm GA --phase phase1
    python analyze_results.py --algorithm HYBRID_PSO_LBFGS --phase phase1 --dispersion
    python analyze_results.py --phase phase1 --generate_analyze
"""

import json
import argparse
import numpy as np
import os

import torch as th
from torchdiffeq import odeint
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# SVG de alta calidad por defecto
plt.rcParams['savefig.format'] = 'svg'
plt.rcParams['svg.fonttype'] = 'none'   # texto como texto (bueno para JCR)

# Config y utilidades
from motor_dynamic_batch import InductionMotorModelBatch
from config import (
    PHASES, PARAM_NAMES, MOTOR_VOLTAGE, OPTIMIZATION_CONFIG, PYTORCH_CONFIG,
    DATA_FILES, TRUE_PARAMS, ALGORITHM_CONFIGS
)
from utils import setup_pytorch, load_measurement_data


# =========================
# Carga y estadísticas
# =========================
def load_algorithm_results(results_dir, algorithm_name):
    """
    Carga resultados de un algoritmo
    """
    filepath = os.path.join(results_dir, f"{algorithm_name}_results.json")
    if not os.path.exists(filepath):
        print(f"No se encontró: {filepath}")
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def get_best_run(runs):
    """Obtiene el número del mejor run"""
    best_fitness = min(run['best_fitness'] for run in runs)
    for run in runs:
        if run['best_fitness'] == best_fitness:
            return run['run_id']
    return 1


def get_worst_run(runs):
    """Obtiene el número del peor run"""
    worst_fitness = max(run['best_fitness'] for run in runs)
    for run in runs:
        if run['best_fitness'] == worst_fitness:
            return run['run_id']
    return 1


def print_detailed_statistics(algorithm_name, data):
    """
    Imprime estadísticas detalladas de un algoritmo
    """
    print(f"\n{'-'*70}")
    print(f" {algorithm_name}")
    print(f"{'-'*70}")

    runs = data['runs']
    stats = data['statistics']

    # Información general
    print(f"\n Información General:")
    print(f"   Total de runs: {len(runs)}")
    print(f"   Tiempo promedio por run: {stats['execution_time']['mean']:.2f}s")
    print(f"   Tiempo total: {stats['execution_time']['total']/60:.2f} min")

    # Estadísticas de fitness
    print(f"\n Fitness:")
    fs = stats['fitness']
    print(f"   Mean ± Std:  {fs['mean']:.6f} ± {fs['std']:.6f}")
    print(f"   Median:      {fs['median']:.6f}")
    print(f"   Best:        {fs['min']:.6f} (Run {get_best_run(runs)})")
    print(f"   Worst:       {fs['max']:.6f} (Run {get_worst_run(runs)})")
    print(f"   IQR:         {fs['iqr']:.6f}")
    print(f"   CV:          {fs['cv']:.4f}")

    # Estadísticas por parámetro
    print(f"\n Error de Parámetros (%):")
    print(f"   {'Parámetro':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"   {'-'*50}")

    for param_name in PARAM_NAMES:
        ps = stats['parameters'][param_name]
        print(f"   {param_name:<10} "
              f"{ps['mean']:>9.2f} "
              f"{ps['std']:>9.2f} "
              f"{ps['min']:>9.2f} "
              f"{ps['max']:>9.2f}")

    # Convergencia
    print(f"\n Convergencia:")
    convergences = [run['final_generation'] for run in runs]
    stagnations = [run['stagnation_count'] for run in runs]
    print(f"   Generaciones promedio: {np.mean(convergences):.1f}")
    print(f"   Estancamiento promedio: {np.mean(stagnations):.1f} generaciones")

    # Mejor solución
    print(f"\n Mejor Solución (Run {get_best_run(runs)}):")
    best_run = runs[get_best_run(runs) - 1]
    print(f"   Fitness: {best_run['best_fitness']:.6f}")
    print(f"   Parámetros:")
    for param_name in PARAM_NAMES:
        param_val = best_run['best_params'][param_name]
        error_pct = best_run['param_errors_percent'][param_name]
        print(f"      {param_name:<6} = {param_val:.6f}  (error: {error_pct:>6.2f}%)")


def compare_algorithms(results_dir):
    """
    Compara todos los algoritmos
    """
    algorithms = ['GA', 'PSO', 'HYBRID_PSO_LBFGS', 'BEE_MEMETIC']

    print(f"\n{'-'*70}")
    print(f" COMPARACIÓN DE ALGORITMOS")
    print(f"{'-'*70}\n")

    # Tabla comparativa
    print(f"{'Algoritmo':<12} {'Mean':<12} {'Std':<12} {'Best':<12} {'Median':<12} {'CV':<10}")
    print(f"{'-'*70}")

    all_stats = {}

    for algo in algorithms:
        data = load_algorithm_results(results_dir, algo)
        if data is None:
            continue

        stats = data['statistics']['fitness']
        all_stats[algo] = stats

        print(f"{algo:<12} "
              f"{stats['mean']:<12.6f} "
              f"{stats['std']:<12.6f} "
              f"{stats['min']:<12.6f} "
              f"{stats['median']:<12.6f} "
              f"{stats['cv']:<10.4f}")

    print(f"{'-'*70}\n")

    # Ranking
    if all_stats:
        print(" Ranking (por fitness promedio):")
        ranked = sorted(all_stats.items(), key=lambda x: x[1]['mean'])
        for i, (algo, stats) in enumerate(ranked, 1):
            print(f"   {i}. {algo:<6} - {stats['mean']:.6f}")

        print("\n Ranking (por mejor solución):")
        ranked_best = sorted(all_stats.items(), key=lambda x: x[1]['min'])
        for i, (algo, stats) in enumerate(ranked_best, 1):
            print(f"   {i}. {algo:<6} - {stats['min']:.6f}")

        print("\n Ranking (por robustez - menor CV):")
        ranked_cv = sorted(all_stats.items(), key=lambda x: x[1]['cv'])
        for i, (algo, stats) in enumerate(ranked_cv, 1):
            print(f"   {i}. {algo:<6} - CV: {stats['cv']:.4f}")


def export_convergence_data(results_dir, output_file):
    """
    Exporta datos de convergencia para graficar externamente
    """
    algorithms = ['GA', 'PSO', 'HYBRID_PSO_LBFGS']

    convergence_data = {}

    for algo in algorithms:
        data = load_algorithm_results(results_dir, algo)
        if data is None:
            continue

        # Extraer convergencia de todos los runs
        convergence_data[algo] = {
            'runs': [run['convergence_history'] for run in data['runs']],
            'mean_convergence': None,  # Calcular después
            'std_convergence': None
        }

        # Calcular promedio y std por generación
        max_gen = max(len(h) for h in convergence_data[algo]['runs'])

        mean_conv = []
        std_conv = []

        for gen in range(max_gen):
            values_at_gen = []
            for history in convergence_data[algo]['runs']:
                if gen < len(history):
                    values_at_gen.append(history[gen])

            if values_at_gen:
                mean_conv.append(float(np.mean(values_at_gen)))
                std_conv.append(float(np.std(values_at_gen)))

        convergence_data[algo]['mean_convergence'] = mean_conv
        convergence_data[algo]['std_convergence'] = std_conv

    # Guardar
    with open(output_file, 'w') as f:
        json.dump(convergence_data, f, indent=2)

    print(f"-> Datos de convergencia exportados a: {output_file}")


# =========================
# Análisis estadístico avanzado (ANOVA, histogramas, ECDF)
# =========================

ALGORITHMS_TO_ANALYZE = ['GA', 'PSO', 'HYBRID_PSO_LBFGS']
PARAM_ERROR_NAMES = ["rs", "rr", "Lls", "Llr", "Lm", "J", "B"]  # coincide con PARAM_NAMES


def load_runs_dataframe(results_dir, algorithms=ALGORITHMS_TO_ANALYZE):
    """
    Construye un DataFrame con:
      - algorithm
      - run_id
      - best_fitness
      - E_param (error medio absoluto en % de todos los parámetros)
    a partir de los *_results.json ya generados.
    """
    rows = []
    for algo in algorithms:
        path = os.path.join(results_dir, f"{algo}_results.json")
        if not os.path.exists(path):
            print(f"(Aviso) No se encontró {path}, se omite {algo}")
            continue

        with open(path, "r") as f:
            data = json.load(f)

        for run in data["runs"]:
            bf = run["best_fitness"]
            perr = run["param_errors_percent"]
            e_param = float(
                np.mean([abs(perr[p]) for p in PARAM_ERROR_NAMES])
            )
            rows.append({
                "algorithm": algo,
                "run_id": run["run_id"],
                "best_fitness": bf,
                "E_param": e_param,
            })

    df = pd.DataFrame(rows)
    return df


def run_anova_best_fitness(df, results_dir):
    """
    ANOVA de una vía y Tukey HSD sobre best_fitness entre algoritmos.
    """
    groups = []
    labels = []
    for algo in df["algorithm"].unique():
        vals = df[df["algorithm"] == algo]["best_fitness"].values
        if len(vals) == 0:
            continue
        groups.append(vals)
        labels.append(algo)

    if len(groups) < 2:
        print("No hay suficientes algoritmos para ANOVA.")
        return

    F, p = f_oneway(*groups)
    print("\n=== ANOVA una vía (best_fitness) ===")
    print("Algoritmos:", ", ".join(labels))
    print(f"F = {F:.4f}, p = {p:.4e}")

    print("\n=== Tukey HSD (post-hoc) ===")
    tukey = pairwise_tukeyhsd(
        endog=df["best_fitness"],
        groups=df["algorithm"],
        alpha=0.05
    )
    print(tukey.summary())


def plot_hist_kde(df, metric, xlabel, results_dir):
    """
    Histogramas + KDE de una métrica (best_fitness o E_param),
    un subplot por algoritmo.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    algos = df["algorithm"].unique()
    n = len(algos)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos):
        sub = df[df["algorithm"] == algo]
        sns.histplot(
            sub[metric],
            kde=True,
            stat="density",
            bins=10,
            ax=ax,
            color="tab:red",
            alpha=0.4
        )
        ax.set_title(algo)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Densidad")

    fig.tight_layout()
    out_dir = os.path.join(results_dir, "distributions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{metric}_hist_kde.svg")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Figura hist+KDE guardada en: {out_path}")
    plt.show()


def compute_ecdf(values):
    x = np.sort(np.asarray(values))
    n = x.size
    y = np.arange(1, n + 1) / n
    return x, y


def plot_ecdf_metric(df, metric, xlabel, logx, results_dir):
    """
    ECDF de una métrica para todos los algoritmos.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.figure(figsize=(6, 4))
    for algo in ALGORITHMS_TO_ANALYZE:
        sub = df[df["algorithm"] == algo]
        if sub.empty:
            continue
        x, y = compute_ecdf(sub[metric].values)
        plt.plot(x, y, marker=".", linestyle="-", label=algo)

    if logx:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Proporción de runs (ECDF)")
    plt.title(f"ECDF de {metric}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_dir = os.path.join(results_dir, "distributions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{metric}_ecdf.svg")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Figura ECDF guardada en: {out_path}")
    plt.show()


def generate_full_analysis(results_dir):
    """
    Ejecuta:
      - DataFrame de runs (best_fitness, E_param)
      - ANOVA + Tukey de best_fitness
      - Histogramas/KDE de best_fitness y E_param
      - ECDF de best_fitness y E_param
    """
    df = load_runs_dataframe(results_dir)
    if df.empty:
        print("No se pudieron cargar runs para el análisis.")
        return

    print("\n=== DataFrame de runs (primeras filas) ===")
    print(df.head())

    # ANOVA + Tukey
    run_anova_best_fitness(df, results_dir)

    # Histogramas/KDE
    plot_hist_kde(df, "best_fitness", "Best fitness", results_dir)
    plot_hist_kde(df, "E_param", "Error medio de parámetros (%)", results_dir)

    # ECDF
    plot_ecdf_metric(df, "best_fitness", "Best fitness (escala log)",
                     logx=True, results_dir=results_dir)
    plot_ecdf_metric(df, "E_param", "Error medio de parámetros (%)",
                     logx=False, results_dir=results_dir)


# =========================
# Simulación y gráficos (mejor run / overlay)
# =========================
def plot_best_parameters_simulation(results_dir, algorithm_name):
    """
    Simula y grafica las señales del motor con los mejores parámetros encontrados
    frente a las mediciones originales (solo un algoritmo).
    """
    data = load_algorithm_results(results_dir, algorithm_name)
    if data is None:
        print(f" No se encontraron resultados para {algorithm_name}")
        return

    best_run = data['runs'][get_best_run(data['runs']) - 1]
    best_params = th.tensor(
        [best_run['best_params'][name] for name in PARAM_NAMES],
        dtype=th.float32
    ).unsqueeze(0)

    print(f"\n Simulando con los mejores parámetros del {algorithm_name}:")
    for name, val in zip(PARAM_NAMES, best_params.squeeze().tolist()):
        print(f"   {name:<8} = {val:.6f}")

    # Configurar PyTorch y modelo
    device = setup_pytorch(PYTORCH_CONFIG)
    best_params = best_params.to(device)
    model = InductionMotorModelBatch(
        vqs=MOTOR_VOLTAGE['vqs'],
        vds=MOTOR_VOLTAGE['vds']
    ).to(device)
    model.update_params_batch(best_params)

    # Cargar mediciones
    current_measured, rpm_measured, torque_measured = load_measurement_data(DATA_FILES, device)

    # Simular con los mejores parámetros
    x0 = th.zeros(1, 5, dtype=th.float32, device=device)
    t = th.linspace(
        0, OPTIMIZATION_CONFIG['time_total'],
        OPTIMIZATION_CONFIG['time_steps'],
        device=device
    )
    sol = odeint(
        model, x0, t,
        method=OPTIMIZATION_CONFIG['ode_method'],
        rtol=OPTIMIZATION_CONFIG['rtol'],
        atol=OPTIMIZATION_CONFIG['atol']
    )
    sol = sol.permute(1, 0, 2)

    current_sim = model.calculate_stator_current(sol).squeeze(0).cpu().numpy()
    rpm_sim = model.calculate_rpm(sol).squeeze(0).cpu().numpy()
    torque_sim = model.calculate_torque(sol).squeeze(0).cpu().numpy()

    t_np = t.detach().cpu().numpy()

    def to_cpu_numpy(x):
        return x.detach().cpu().numpy() if th.is_tensor(x) else x

    current_measured = to_cpu_numpy(current_measured)
    rpm_measured = to_cpu_numpy(rpm_measured)
    torque_measured = to_cpu_numpy(torque_measured)

    # Graficar
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t_np, current_measured[:len(t_np)], label="Medido",
             color='gray', linestyle='--')
    plt.plot(t_np, current_sim, label="Simulado",
             color=ALGORITHM_CONFIGS.get(algorithm_name, {}).get('color', 'blue'))
    plt.ylabel("Corriente (A)")
    plt.title(f"Simulación con mejores parámetros - {algorithm_name}")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t_np, rpm_measured[:len(t_np)], label="Medido",
             color='gray', linestyle='--')
    plt.plot(t_np, rpm_sim, label="Simulado",
             color=ALGORITHM_CONFIGS.get(algorithm_name, {}).get('color', 'orange'))
    plt.ylabel("RPM")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(t_np, torque_measured[:len(t_np)], label="Medido",
             color='gray', linestyle='--')
    plt.plot(t_np, torque_sim, label="Simulado",
             color=ALGORITHM_CONFIGS.get(algorithm_name, {}).get('color', 'green'))
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Torque (N·m)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_overlay_algorithms(results_dir, algorithms=('GA', 'PSO', 'HYBRID_PSO_LBFGS', 'CS')):
    """
    Carga el mejor individuo de cada algoritmo, simula y sobrepone
    corriente, rpm y torque en la misma figura usando colores de config.
    """
    # Configurar PyTorch y modelo
    device = setup_pytorch(PYTORCH_CONFIG)
    model = InductionMotorModelBatch(
        vqs=MOTOR_VOLTAGE['vqs'],
        vds=MOTOR_VOLTAGE['vds']
    ).to(device)

    # Tiempo y CI
    t = th.linspace(
        0, OPTIMIZATION_CONFIG['time_total'],
        OPTIMIZATION_CONFIG['time_steps'],
        device=device
    )
    x0 = th.zeros(1, 5, dtype=th.float32, device=device)

    # Mediciones
    cur_meas, rpm_meas, torq_meas = load_measurement_data(DATA_FILES, device)

    # Series por algoritmo
    sims = {}
    for algo in algorithms:
        data = load_algorithm_results(results_dir, algo)
        if not data:
            continue
        best_run = data['runs'][get_best_run(data['runs']) - 1]
        best_params = th.tensor(
            [best_run['best_params'][name] for name in PARAM_NAMES],
            dtype=th.float32, device=device
        ).unsqueeze(0)
        model.update_params_batch(best_params)
        sol = odeint(
            model, x0, t,
            method=OPTIMIZATION_CONFIG['ode_method'],
            rtol=OPTIMIZATION_CONFIG['rtol'],
            atol=OPTIMIZATION_CONFIG['atol']
        )
        sol = sol.permute(1, 0, 2)
        sims[algo] = {
            'cur': model.calculate_stator_current(sol).squeeze(0).detach().cpu().numpy(),
            'rpm': model.calculate_rpm(sol).squeeze(0).detach().cpu().numpy(),
            'tor': model.calculate_torque(sol).squeeze(0).detach().cpu().numpy(),
        }

    # Numpy
    t_np = t.detach().cpu().numpy()
    cur_meas_np = cur_meas.detach().cpu().numpy()
    rpm_meas_np = rpm_meas.detach().cpu().numpy()
    torq_meas_np = torq_meas.detach().cpu().numpy()

    # Figura
    plt.figure(figsize=(12, 8))

    # Corriente
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t_np, cur_meas_np[:len(t_np)], label="Medido",
             color='gray', linestyle='--', linewidth=1.5)
    for algo, series in sims.items():
        color = ALGORITHM_CONFIGS.get(algo, {}).get('color', None)
        label = ALGORITHM_CONFIGS.get(algo, {}).get('short_name', algo)
        ax1.plot(t_np, series['cur'], label=label,
                 color=color, linewidth=1.8)
    ax1.set_ylabel("Corriente (A)")
    ax1.set_title("Simulación sobrepuesta por algoritmo")
    ax1.grid(True)
    ax1.legend(ncol=2)

    # RPM
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(t_np, rpm_meas_np[:len(t_np)], label="Medido",
             color='gray', linestyle='--', linewidth=1.5)
    for algo, series in sims.items():
        color = ALGORITHM_CONFIGS.get(algo, {}).get('color', None)
        label = ALGORITHM_CONFIGS.get(algo, {}).get('short_name', algo)
        ax2.plot(t_np, series['rpm'], label=label,
                 color=color, linewidth=1.8)
    ax2.set_ylabel("RPM")
    ax2.grid(True)
    ax2.legend(ncol=2)

    # Torque
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(t_np, torq_meas_np[:len(t_np)], label="Medido",
             color='gray', linestyle='--', linewidth=1.5)
    for algo, series in sims.items():
        color = ALGORITHM_CONFIGS.get(algo, {}).get('color', None)
        label = ALGORITHM_CONFIGS.get(algo, {}).get('short_name', algo)
        ax3.plot(t_np, series['tor'], label=label,
                 color=color, linewidth=1.8)
    ax3.set_xlabel("Tiempo (s)")
    ax3.set_ylabel("Torque (N·m)")
    ax3.grid(True)
    ax3.legend(ncol=2)

    plt.tight_layout()
    plt.show()


# ========= Dispersión de señales por runs =========
def simulate_all_runs_signals(results_dir, algorithm_name):
    """
    Simula TODAS las runs de un algoritmo y devuelve:
      t_np              -> vector de tiempo
      cur_sims, rpm_sims, tor_sims -> arrays [n_runs, T]
      cur_meas, rpm_meas, tor_meas -> arrays [T]
      best_idx          -> índice de la mejor run
    """
    data = load_algorithm_results(results_dir, algorithm_name)
    if data is None:
        raise RuntimeError(f"No se encontraron resultados para {algorithm_name}")

    runs = data['runs']
    n_runs = len(runs)

    # PyTorch y modelo
    device = setup_pytorch(PYTORCH_CONFIG)
    model = InductionMotorModelBatch(
        vqs=MOTOR_VOLTAGE['vqs'],
        vds=MOTOR_VOLTAGE['vds']
    ).to(device)

    # Tiempo y condiciones iniciales
    t = th.linspace(
        0, OPTIMIZATION_CONFIG['time_total'],
        OPTIMIZATION_CONFIG['time_steps'],
        device=device
    )
    x0 = th.zeros(1, 5, dtype=th.float32, device=device)

    # Mediciones
    cur_meas, rpm_meas, tor_meas = load_measurement_data(DATA_FILES, device)

    T = t.numel()
    cur_sims = np.zeros((n_runs, T))
    rpm_sims = np.zeros((n_runs, T))
    tor_sims = np.zeros((n_runs, T))

    # Índice de la mejor run
    best_run_id = get_best_run(runs)   # run_id (1..N)
    best_idx = best_run_id - 1        # índice 0..N-1

    for i, run in enumerate(runs):
        best_params = th.tensor(
            [run["best_params"][name] for name in PARAM_NAMES],
            dtype=th.float32, device=device
        ).unsqueeze(0)

        model.update_params_batch(best_params)

        sol = odeint(
            model, x0, t,
            method=OPTIMIZATION_CONFIG['ode_method'],
            rtol=OPTIMIZATION_CONFIG['rtol'],
            atol=OPTIMIZATION_CONFIG['atol']
        )
        sol = sol.permute(1, 0, 2)

        cur = model.calculate_stator_current(sol).squeeze(0).detach().cpu().numpy()
        rpm = model.calculate_rpm(sol).squeeze(0).detach().cpu().numpy()
        tor = model.calculate_torque(sol).squeeze(0).detach().cpu().numpy()

        cur_sims[i, :] = cur
        rpm_sims[i, :] = rpm
        tor_sims[i, :] = tor

    t_np = t.detach().cpu().numpy()
    cur_meas_np = cur_meas.detach().cpu().numpy()
    rpm_meas_np = rpm_meas.detach().cpu().numpy()
    tor_meas_np = tor_meas.detach().cpu().numpy()

    return t_np, cur_sims, rpm_sims, tor_sims, cur_meas_np, rpm_meas_np, tor_meas_np, best_idx


def plot_signal_with_dispersion(t, sims, meas, best_idx, ylabel, title, color, save_path=None):
    """
    t        -> [T]
    sims     -> [n_runs, T]
    meas     -> [T]
    best_idx -> índice de mejor run
    """
    # Banda de dispersión (percentiles 2.5 y 97.5)
    lower = np.percentile(sims, 2.5, axis=0)
    upper = np.percentile(sims, 97.5, axis=0)
    best = sims[best_idx, :]

    plt.figure(figsize=(8, 4))

    # Banda
    plt.fill_between(t, lower, upper, color=color, alpha=0.2,
                     label="Rango 95% runs")

    # Mejor run
    plt.plot(t, best, color=color, linewidth=2.0, label="Mejor run")

    # Señal medida
    plt.plot(t, meas[:len(t)], color="gray", linestyle="--",
             linewidth=1.2, label="Medida")

    plt.xlabel("Tiempo (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path is not None:
        # Guardar siempre en SVG dentro de la ruta indicada
        base, _ = os.path.splitext(save_path)
        svg_path = base + ".svg"
        plt.savefig(svg_path, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def plot_dispersion_three_signals(results_dir, algorithm_name):
    """
    Genera 3 figuras: Corriente, RPM y Torque con dispersión de runs
    alrededor del mejor resultado para un algoritmo.
    """
    (t, cur_sims, rpm_sims, tor_sims,
     cur_meas, rpm_meas, tor_meas, best_idx) = simulate_all_runs_signals(
        results_dir, algorithm_name
    )

    algo_cfg = ALGORITHM_CONFIGS.get(algorithm_name, {})
    base_color = algo_cfg.get("color", "tab:blue")
    short_name = algo_cfg.get("short_name", algorithm_name)

    # Carpeta por algoritmo: results_dir/dispersion_plots/ALGO/
    out_dir = os.path.join(results_dir, "dispersion_plots", algorithm_name)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Corriente
    plot_signal_with_dispersion(
        t, cur_sims, cur_meas, best_idx,
        ylabel="Corriente (A)",
        title=f"Corriente - {short_name}",
        color=base_color,
        save_path=os.path.join(out_dir, "current_dispersion")
    )

    # 2) RPM
    plot_signal_with_dispersion(
        t, rpm_sims, rpm_meas, best_idx,
        ylabel="Velocidad (rpm)",
        title=f"Velocidad - {short_name}",
        color=base_color,
        save_path=os.path.join(out_dir, "rpm_dispersion")
    )

    # 3) Torque
    plot_signal_with_dispersion(
        t, tor_sims, tor_meas, best_idx,
        ylabel="Torque (N·m)",
        title=f"Torque - {short_name}",
        color=base_color,
        save_path=os.path.join(out_dir, "torque_dispersion")
    )


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description='Analizar resultados de experimentos')
    parser.add_argument('--phase', type=str, default='phase1',
                        help='Fase a analizar (phase1, phase2, phase3)')
    parser.add_argument('--algorithm', type=str, default=None,
                        help='Algoritmo específico a analizar (GA, PSO, HYBRID_PSO_LBFGS, etc.)')
    parser.add_argument('--export-convergence', action='store_true',
                        help='Exportar datos de convergencia')
    parser.add_argument('--plot', action='store_true',
                        help='Mostrar gráficas de simulación (mejor run / overlay)')
    parser.add_argument('--dispersion', action='store_true',
                        help='Mostrar/grabar gráficas de dispersión por runs')
    parser.add_argument('--generate_analyze', action='store_true',
                        help='Generar análisis estadístico (ANOVA, histogramas, ECDF)')
    args = parser.parse_args()

    # Directorio de resultados
    if args.phase not in PHASES:
        print(f" Fase '{args.phase}' no reconocida")
        print(f"   Fases disponibles: {', '.join(PHASES.keys())}")
        return

    results_dir = PHASES[args.phase]['output_dir']
    if not os.path.exists(results_dir):
        print(f" No se encontró el directorio: {results_dir}")
        print(f"   Ejecuta primero: python main_{args.phase}.py")
        return

    print(f"\n{'='*70}")
    print(f"Analizando: {results_dir}")
    print(f"{'='*70}")

    # Analizar algoritmo específico o todos
    if args.algorithm:
        data = load_algorithm_results(results_dir, args.algorithm)
        if data:
            print_detailed_statistics(args.algorithm, data)
    else:
        compare_algorithms(results_dir)
        for algo in ['GA', 'PSO', 'HYBRID_PSO_LBFGS']:
            data = load_algorithm_results(results_dir, algo)
            if data:
                print_detailed_statistics(algo, data)

    # Exportar convergencia si se solicita
    if args.export_convergence:
        output_file = os.path.join(results_dir, 'convergence_data.json')
        export_convergence_data(results_dir, output_file)

    # Análisis estadístico avanzado (ANOVA, histogramas, ECDF)
    if getattr(args, "generate_analyze", False):
        generate_full_analysis(results_dir)

    # Gráficas clásicas
    if getattr(args, "plot", False):
        if args.algorithm:
            plot_best_parameters_simulation(results_dir, args.algorithm)
        else:
            plot_overlay_algorithms(results_dir, algorithms=['GA', 'PSO', 'HYBRID_PSO_LBFGS'])

    # Gráficas de dispersión (SVG, carpeta por algoritmo)
    if getattr(args, "dispersion", False):
        if args.algorithm:
            algos = [args.algorithm]
        else:
            algos = ['GA', 'PSO', 'HYBRID_PSO_LBFGS']

        for algo in algos:
            data = load_algorithm_results(results_dir, algo)
            if data is None:
                print(f"  (Saltando {algo}: no hay resultados)")
                continue
            print(f"\nGenerando figuras de dispersión para {algo}...")
            plot_dispersion_three_signals(results_dir, algo)

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
