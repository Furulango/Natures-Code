import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import f_oneway          # ANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd  # Tukey HSD


# === CONFIGURACIÓN ===
RESULTS_DIR = "results/phase1_synthetic"   # mismo que PHASES["phase1"]["output_dir"]
ALGORITHMS = ["GA", "PSO", "HYBRID_PSO_LBFGS"]


def load_best_fitness(results_dir, algo_name):
    """
    Carga el vector de best_fitness para todos los runs de un algoritmo.
    Ajusta la clave si en tu JSON se llama distinto (ej. 'bestfitness').
    """
    fname = f"{algo_name}_results.json"
    path = os.path.join(results_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # Si tu JSON usa 'bestfitness' en lugar de 'best_fitness', cambia la línea:
    # return [run["bestfitness"] for run in data["runs"]]
    return [run["best_fitness"] for run in data["runs"]]


def build_long_dataframe(results_dir, algorithms):
    """
    Construye un DataFrame tipo 'long' con columnas:
    - algorithm
    - best_fitness
    Para usarlo en ANOVA y Tukey.
    """
    rows = []
    for algo in algorithms:
        fitness_vals = load_best_fitness(results_dir, algo)
        for val in fitness_vals:
            rows.append({"algorithm": algo, "best_fitness": val})

    df = pd.DataFrame(rows)
    return df


def run_anova(df):
    """
    Ejecuta ANOVA de una vía sobre best_fitness ~ algorithm.
    Devuelve F, p y un diccionario con los vectores por algoritmo.
    """
    groups = {}
    for algo in df["algorithm"].unique():
        groups[algo] = df.loc[df["algorithm"] == algo, "best_fitness"].values

    # ANOVA de una vía
    F, p = f_oneway(*(groups[algo] for algo in groups.keys()))
    print(f"\n=== ANOVA una vía (best_fitness) ===")
    print(f"Algoritmos: {', '.join(groups.keys())}")
    print(f"F = {F:.4f}, p = {p:.4e}")

    return F, p, groups


def run_tukey(df):
    """
    Ejecuta prueba post-hoc de Tukey HSD entre los algoritmos.
    """
    print("\n=== Tukey HSD (post-hoc) ===")
    tukey = pairwise_tukeyhsd(
        endog=df["best_fitness"],      # variable dependiente
        groups=df["algorithm"],        # factor (algoritmo)
        alpha=0.05
    )
    print(tukey.summary())
    return tukey


def plot_boxplot(df, output_dir=None):
    """
    Boxplot del best_fitness por algoritmo.
    """
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    df.boxplot(column="best_fitness", by="algorithm", ax=ax)
    plt.title("Distribución de best_fitness por algoritmo")
    plt.suptitle("")  # quita el título extra de pandas
    plt.xlabel("Algoritmo")
    plt.ylabel("best_fitness (menor es mejor)")
    plt.grid(True, axis="y", alpha=0.3)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "anova_boxplot_best_fitness.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Boxplot guardado en: {out_path}")

    plt.show()


def plot_means_with_ci(df, output_dir=None):
    """
    Grafica la media de best_fitness por algoritmo con barras de error
    (intervalos aproximados 95% usando 1.96 * SEM).
    """
    summary = (
        df.groupby("algorithm")["best_fitness"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Error estándar y CI ~95% (aprox normal)
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95"] = 1.96 * summary["sem"]

    plt.figure(figsize=(8, 5))
    x = np.arange(len(summary))
    means = summary["mean"].values
    ci = summary["ci95"].values

    plt.bar(x, means, yerr=ci, capsize=5, alpha=0.8)
    plt.xticks(x, summary["algorithm"])
    plt.ylabel("Media de best_fitness")
    plt.title("Medias de best_fitness con IC≈95% por algoritmo")
    plt.grid(True, axis="y", alpha=0.3)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "anova_means_ci_best_fitness.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Gráfica de medias guardada en: {out_path}")

    plt.show()


def main():
    # 1) Construir DataFrame
    df = build_long_dataframe(RESULTS_DIR, ALGORITHMS)
    print("Primeras filas de datos:")
    print(df.head())

    # 2) ANOVA
    F, p, groups = run_anova(df)

    # 3) Tukey solo si ANOVA es significativo (opcional)
    if p < 0.05:
        tukey = run_tukey(df)
    else:
        print("\nANOVA no significativo (p >= 0.05); Tukey opcional.")

    # 4) Gráficas
    plots_dir = os.path.join(RESULTS_DIR, "anova_plots")
    plot_boxplot(df, output_dir=plots_dir)
    plot_means_with_ci(df, output_dir=plots_dir)


if __name__ == "__main__":
    main()
