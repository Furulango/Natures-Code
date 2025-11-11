"""
Script para analizar resultados de experimentos

EJECUCIÓN:
    python analyze_results.py --phase phase1
    python analyze_results.py --phase phase1 --plot
    python analyze_results.py --algorithm GA --phase phase1
"""
import json
import argparse
import numpy as np
import os
from config import PHASES, PARAM_NAMES

import torch as th
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from motor_dynamic_batch import InductionMotorModelBatch
from config import MOTOR_VOLTAGE, OPTIMIZATION_CONFIG, PYTORCH_CONFIG, PARAM_NAMES, DATA_FILES, TRUE_PARAMS
from utils import setup_pytorch, load_measurement_data



def load_algorithm_results(results_dir, algorithm_name):
    """
    Carga resultados de un algoritmo
    """
    filepath = os.path.join(results_dir, f"{algorithm_name}_results.json")
    
    if not os.path.exists(filepath):
        print(f"❌ No se encontró: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data


def print_detailed_statistics(algorithm_name, data):
    """
    Imprime estadísticas detalladas de un algoritmo
    """
    print(f"\n{'='*70}")
    print(f"📊 {algorithm_name} - ANÁLISIS DETALLADO")
    print(f"{'='*70}")
    
    runs = data['runs']
    stats = data['statistics']
    
    # Información general
    print(f"\n🔢 Información General:")
    print(f"   Total de runs: {len(runs)}")
    print(f"   Tiempo promedio por run: {stats['execution_time']['mean']:.2f}s")
    print(f"   Tiempo total: {stats['execution_time']['total']/60:.2f} min")
    
    # Estadísticas de fitness
    print(f"\n📈 Fitness:")
    fs = stats['fitness']
    print(f"   Mean ± Std:  {fs['mean']:.6f} ± {fs['std']:.6f}")
    print(f"   Median:      {fs['median']:.6f}")
    print(f"   Best:        {fs['min']:.6f} (Run {get_best_run(runs)})")
    print(f"   Worst:       {fs['max']:.6f} (Run {get_worst_run(runs)})")
    print(f"   IQR:         {fs['iqr']:.6f}")
    print(f"   CV:          {fs['cv']:.4f}")
    
    # Estadísticas por parámetro
    print(f"\n🎯 Error de Parámetros (%):")
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
    print(f"\n📉 Convergencia:")
    convergences = [run['final_generation'] for run in runs]
    stagnations = [run['stagnation_count'] for run in runs]
    print(f"   Generaciones promedio: {np.mean(convergences):.1f}")
    print(f"   Estancamiento promedio: {np.mean(stagnations):.1f} generaciones")
    
    # Mejor solución
    print(f"\n🏆 Mejor Solución (Run {get_best_run(runs)}):")
    best_run = runs[get_best_run(runs) - 1]
    print(f"   Fitness: {best_run['best_fitness']:.6f}")
    print(f"   Parámetros:")
    for param_name in PARAM_NAMES:
        param_val = best_run['best_params'][param_name]
        error_pct = best_run['param_errors_percent'][param_name]
        print(f"      {param_name:<6} = {param_val:.6f}  (error: {error_pct:>6.2f}%)")


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


def compare_algorithms(results_dir):
    """
    Compara todos los algoritmos
    """
    algorithms = ['GA', 'PSO', 'DE', 'CS']
    
    print(f"\n{'='*70}")
    print(f"🔬 COMPARACIÓN DE ALGORITMOS")
    print(f"{'='*70}\n")
    
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
        print("🏆 Ranking (por fitness promedio):")
        ranked = sorted(all_stats.items(), key=lambda x: x[1]['mean'])
        for i, (algo, stats) in enumerate(ranked, 1):
            print(f"   {i}. {algo:<6} - {stats['mean']:.6f}")
        
        print("\n🎯 Ranking (por mejor solución):")
        ranked_best = sorted(all_stats.items(), key=lambda x: x[1]['min'])
        for i, (algo, stats) in enumerate(ranked_best, 1):
            print(f"   {i}. {algo:<6} - {stats['min']:.6f}")
        
        print("\n📊 Ranking (por robustez - menor CV):")
        ranked_cv = sorted(all_stats.items(), key=lambda x: x[1]['cv'])
        for i, (algo, stats) in enumerate(ranked_cv, 1):
            print(f"   {i}. {algo:<6} - CV: {stats['cv']:.4f}")


def export_convergence_data(results_dir, output_file):
    """
    Exporta datos de convergencia para graficar externamente
    """
    algorithms = ['GA', 'PSO', 'DE', 'CS']
    
    convergence_data = {}
    
    for algo in algorithms:
        data = load_algorithm_results(results_dir, algo)
        if data is None:
            continue
        
        # Extraer historias de convergencia de todos los runs
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
    
    print(f"✓ Datos de convergencia exportados a: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analizar resultados de experimentos')
    parser.add_argument('--phase', type=str, default='phase1', 
                       help='Fase a analizar (phase1, phase2, phase3)')
    parser.add_argument('--algorithm', type=str, default=None,
                       help='Algoritmo específico a analizar (GA, PSO, DE, CS)')
    parser.add_argument('--export-convergence', action='store_true',
                       help='Exportar datos de convergencia')
    parser.add_argument('--plot', action='store_true',
                       help='Mostrar gráficas de simulación con los mejores parámetros')
    args = parser.parse_args()
    
    # Obtener directorio de resultados
    if args.phase not in PHASES:
        print(f"❌ Fase '{args.phase}' no reconocida")
        print(f"   Fases disponibles: {', '.join(PHASES.keys())}")
        return
    
    results_dir = PHASES[args.phase]['output_dir']
    
    if not os.path.exists(results_dir):
        print(f"❌ No se encontró el directorio: {results_dir}")
        print(f"   Ejecuta primero: python main_{args.phase}.py")
        return
    
    print(f"\n{'='*70}")
    print(f"📂 Analizando: {results_dir}")
    print(f"{'='*70}")
    
    # Analizar algoritmo específico o todos
    if args.algorithm:
        data = load_algorithm_results(results_dir, args.algorithm)
        if data:
            print_detailed_statistics(args.algorithm, data)
    else:
        compare_algorithms(results_dir)
        
        # Mostrar detalles de cada algoritmo
        for algo in ['GA', 'PSO', 'DE', 'CS']:
            data = load_algorithm_results(results_dir, algo)
            if data:
                print_detailed_statistics(algo, data)
    
    # Exportar convergencia si se solicita
    if args.export_convergence:
        output_file = os.path.join(results_dir, 'convergence_data.json')
        export_convergence_data(results_dir, output_file)
    
        # Exportar convergencia si se solicita
    if args.export_convergence:
        output_file = os.path.join(results_dir, 'convergence_data.json')
        export_convergence_data(results_dir, output_file)
    
    # Graficar simulaciones
    if getattr(args, "plot", False):
        for algo in ['GA', 'PSO', 'DE', 'CS']:
            data = load_algorithm_results(results_dir, algo)
            if data:
                plot_best_parameters_simulation(results_dir, algo)

    print(f"\n{'='*70}\n")


def plot_best_parameters_simulation(results_dir, algorithm_name):
    """
    Simula y grafica las señales del motor con los mejores parámetros encontrados
    frente a las mediciones originales.
    """
    data = load_algorithm_results(results_dir, algorithm_name)
    if data is None:
        print(f"❌ No se encontraron resultados para {algorithm_name}")
        return

    best_run = data['runs'][get_best_run(data['runs']) - 1]
    best_params = th.tensor(
        [best_run['best_params'][name] for name in PARAM_NAMES],
        dtype=th.float32
    ).unsqueeze(0)

    print(f"\n🔍 Simulando con los mejores parámetros del {algorithm_name}:")
    for name, val in zip(PARAM_NAMES, best_params.squeeze().tolist()):
        print(f"   {name:<8} = {val:.6f}")

    # Configurar PyTorch y modelo
    device = setup_pytorch(PYTORCH_CONFIG)
    best_params = best_params.to(device)
    model = InductionMotorModelBatch(vqs=MOTOR_VOLTAGE['vqs'], vds=MOTOR_VOLTAGE['vds']).to(device)
    model.update_params_batch(best_params)

    # Cargar mediciones
    current_measured, rpm_measured, torque_measured = load_measurement_data(DATA_FILES, device)

    # Simular con los mejores parámetros
    x0 = th.zeros(1, 5, dtype=th.float32, device=device)
    t = th.linspace(0, OPTIMIZATION_CONFIG['time_total'], OPTIMIZATION_CONFIG['time_steps'], device=device)
    sol = odeint(model, x0, t, method=OPTIMIZATION_CONFIG['ode_method'],
                 rtol=OPTIMIZATION_CONFIG['rtol'], atol=OPTIMIZATION_CONFIG['atol'])
    sol = sol.permute(1, 0, 2)

    current_sim = model.calculate_stator_current(sol).squeeze(0).cpu().numpy()
    rpm_sim = model.calculate_rpm(sol).squeeze(0).cpu().numpy()
    torque_sim = model.calculate_torque(sol).squeeze(0).cpu().numpy()

    # === Conversión explícita a CPU y NumPy ===
    t_np = t.detach().cpu().numpy()

    def to_cpu_numpy(x):
        return x.detach().cpu().numpy() if th.is_tensor(x) else x

    current_measured = to_cpu_numpy(current_measured)
    rpm_measured = to_cpu_numpy(rpm_measured)
    torque_measured = to_cpu_numpy(torque_measured)

    current_sim = to_cpu_numpy(current_sim)
    rpm_sim = to_cpu_numpy(rpm_sim)
    torque_sim = to_cpu_numpy(torque_sim)


    # Graficar
    t_np = t.cpu().numpy()
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t_np, current_measured[:len(t_np)], label="Medido", color='gray', linestyle='--')
    plt.plot(t_np, current_sim, label="Simulado", color='blue')
    plt.ylabel("Corriente (A)")
    plt.title(f"Simulación con mejores parámetros - {algorithm_name}")
    plt.legend(); plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t_np, rpm_measured[:len(t_np)], label="Medido", color='gray', linestyle='--')
    plt.plot(t_np, rpm_sim, label="Simulado", color='orange')
    plt.ylabel("RPM")
    plt.legend(); plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(t_np, torque_measured[:len(t_np)], label="Medido", color='gray', linestyle='--')
    plt.plot(t_np, torque_sim, label="Simulado", color='green')
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Torque (N·m)")
    plt.legend(); plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
