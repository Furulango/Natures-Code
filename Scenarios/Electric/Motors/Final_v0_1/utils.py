"""
Funciones auxiliares para el proyecto
"""

import torch as th
import numpy as np
import json
import os
import math
from datetime import datetime

from config import TRUE_PARAMS, PARAM_NAMES, NAMEPLATE, MOTOR_VOLTAGE, BOUNDS, PRIORS_CONFIG



def setup_pytorch(config):
    """
    Configura PyTorch con las optimizaciones necesarias
    """
    th.backends.cudnn.benchmark = config['cudnn_benchmark']
    th.backends.cudnn.deterministic = config['cudnn_deterministic']
    th.set_float32_matmul_precision(config['matmul_precision'])
    
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    
    if th.cuda.is_available():
        print(f"-> GPU detectada: {th.cuda.get_device_name(0)}")
        print(f"-> VRAM disponible: {th.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("->  Usando CPU")
    
    return device


def set_seed(seed):
    """
    Establece semilla para reproducibilidad
    """
    th.manual_seed(seed)
    np.random.seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def load_measurement_data(data_files, device):
    """
    Carga los datos de mediciones
    """
    current = th.tensor(np.loadtxt(data_files['current']), dtype=th.float32, device=device)
    rpm = th.tensor(np.loadtxt(data_files['rpm']), dtype=th.float32, device=device)
    torque = th.tensor(np.loadtxt(data_files['torque']), dtype=th.float32, device=device)
    
    print(f"\n Datos cargados:")
    print(f"   Puntos: {len(current)}")
    print(f"   Corriente: [{current.min():.2f}, {current.max():.2f}] A")
    print(f"   RPM: [{rpm.min():.2f}, {rpm.max():.2f}]")
    print(f"   Torque: [{torque.min():.2f}, {torque.max():.2f}] N·m")
    
    return current, rpm, torque


def calculate_parameter_errors(estimated_params, true_params=None):
    """
    Calcula errores de parámetros respecto a valores verdaderos
    
    Args:
        estimated_params: np.array o list con parámetros estimados
        true_params: dict con valores verdaderos (default: TRUE_PARAMS)
    
    Returns:
        dict con errores absolutos y porcentuales
    """
    if true_params is None:
        true_params = TRUE_PARAMS
    
    if isinstance(estimated_params, th.Tensor):
        estimated_params = estimated_params.cpu().numpy()
    
    errors = {}
    errors_percent = {}
    
    for i, param_name in enumerate(PARAM_NAMES):
        true_val = true_params[param_name]
        est_val = estimated_params[i]
        
        # Error absoluto
        errors[param_name] = float(abs(est_val - true_val))
        
        # Error porcentual
        errors_percent[param_name] = float(abs((est_val - true_val) / true_val) * 100)
    
    return errors, errors_percent


def detect_stagnation(history, window=50, threshold=1e-6):
    """
    Detecta cuántas generaciones lleva sin mejorar significativamente
    """
    if len(history) < window:
        return 0
    
    recent = history[-window:]
    improvement = abs(recent[0] - recent[-1])
    
    if improvement < threshold:
        # Buscar última mejora significativa
        for i in range(len(history)-1, 0, -1):
            if abs(history[i] - history[i-1]) > threshold:
                return len(history) - i
        return len(history)
    
    return 0


def create_output_directory(output_dir):
    """
    Crea directorio de salida si no existe
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"-> Directorio de salida: {output_dir}")


def save_run_data(output_file, run_data, append=True):
    """
    Guarda datos de un run en archivo JSON
    
    Args:
        output_file: Ruta del archivo
        run_data: Dict con datos del run
        append: Si True, agrega al archivo existente
    """
    if append and os.path.exists(output_file):
        # Cargar datos existentes
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
        
        # Agregar nuevo run
        existing_data['runs'].append(run_data)
        
        # Guardar
        with open(output_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
    else:
        # Crear nuevo archivo
        data = {
            'metadata': {
                'algorithm': run_data.get('algorithm', 'Unknown'),
                'created': datetime.now().isoformat(),
                'total_runs': 1
            },
            'runs': [run_data]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)


def load_results(results_file):
    """
    Carga resultados desde archivo JSON
    """
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data


def compute_statistics(values):
    """
    Calcula estadísticas descriptivas de una lista de valores
    
    Returns:
        dict con estadísticas
    """
    values = np.array(values)
    
    stats = {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
        'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
    }
    
    # Coeficiente de variación (std/mean)
    if stats['mean'] != 0:
        stats['cv'] = stats['std'] / abs(stats['mean'])
    else:
        stats['cv'] = float('inf')
    
    return stats


def format_time(seconds):
    """
    Formatea tiempo en formato legible
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.2f}hrs"


def print_progress_bar(iteration, total, prefix='', suffix='', length=40):
    """
    Imprime barra de progreso
    """
    percent = 100 * (iteration / float(total))
    filled = int(length * iteration // total)
    bar = '█' * filled + '-' * (length - filled)
    
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='')
    
    if iteration == total:
        print()


def estimate_remaining_time(elapsed_time, completed, total):
    """
    Estima tiempo restante
    """
    if completed == 0:
        return "Calculando..."
    
    avg_time = elapsed_time / completed
    remaining = (total - completed) * avg_time
    
    return format_time(remaining)


def print_run_summary(run_id, total_runs, best_fitness, execution_time, 
                     param_errors_percent, elapsed_total, algorithm_name):
    """
    Imprime resumen de un run
    """
    avg_error = np.mean(list(param_errors_percent.values()))
    eta = estimate_remaining_time(elapsed_total, run_id, total_runs)
    
    print(f"Run {run_id:2d}/{total_runs}: "
          f"Fitness={best_fitness:.6f}, "
          f"Avg_Error={avg_error:.2f}%, "
          f"Time={format_time(execution_time)}, "
          f"ETA={eta}")


def print_algorithm_statistics(algorithm_name, stats, param_stats):
    """
    Imprime estadísticas de un algoritmo
    """
    print(f"\n{'='*70}")
    print(f" Estadísticas de {algorithm_name}")
    print(f"{'='*70}")
    
    print(f"\nFitness:")
    print(f"  Mean ± Std:  {stats['mean']:.6f} ± {stats['std']:.6f}")
    print(f"  Median:      {stats['median']:.6f}")
    print(f"  Min:         {stats['min']:.6f}")
    print(f"  Max:         {stats['max']:.6f}")
    print(f"  IQR:         {stats['iqr']:.6f}")
    print(f"  CV:          {stats['cv']:.4f}")
    
    print(f"\nErrores de parámetros (%):")
    print(f"  {'Parámetro':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"  {'-'*50}")
    
    for param_name in PARAM_NAMES:
        pstats = param_stats[param_name]
        print(f"  {param_name:<10} "
              f"{pstats['mean']:>9.2f} "
              f"{pstats['std']:>9.2f} "
              f"{pstats['min']:>9.2f} "
              f"{pstats['max']:>9.2f}")


def create_summary_report(all_results, output_file):
    """
    Crea reporte resumen de todos los algoritmos
    """
    summary = {
        'generated_at': datetime.now().isoformat(),
        'algorithms': {}
    }
    
    for algo_name, results in all_results.items():
        # Extraer fitness de todos los runs
        fitness_values = [run['best_fitness'] for run in results['runs']]
        
        # Estadísticas generales
        summary['algorithms'][algo_name] = {
            'total_runs': len(fitness_values),
            'fitness_statistics': compute_statistics(fitness_values),
            'best_overall': {
                'fitness': min(fitness_values),
                'run_id': fitness_values.index(min(fitness_values)) + 1,
                'parameters': results['runs'][fitness_values.index(min(fitness_values))]['best_params']
            }
        }
        
        # Estadísticas por parámetro
        param_stats = {}
        for param_name in PARAM_NAMES:
            param_errors = [run['param_errors_percent'][param_name] 
                          for run in results['runs']]
            param_stats[param_name] = compute_statistics(param_errors)
        
        summary['algorithms'][algo_name]['parameter_errors'] = param_stats
    
    # Guardar
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n-> Resumen guardado en: {output_file}")
    
    return summary

def compute_b_prior_from_nameplate(nameplate: dict) -> float:
    """
    Estima B_prior ≈ P_fw(ω_n) / ω_n^2 con P_fw ≈ friction_frac_of_output * P_out.
    """
    P_out = float(nameplate['rated_power_kw']) * 1000.0
    rpm_n = float(nameplate['rated_speed_rpm'])
    eta   = float(nameplate.get('rated_efficiency', 0.9))
    frac  = float(nameplate.get('friction_frac_of_output', 0.02))
    # Velocidad mecánica nominal
    omega_n = 2.0 * math.pi * rpm_n / 60.0
    # Pérdida mecánica estimada en nominal
    P_fw = max(1e-3, frac * P_out)  # W
    # B_prior en N·m·s/rad (W = B*ω^2)
    B_prior = P_fw / max(1e-6, omega_n**2)
    return B_prior

def compute_lm_prior_from_nameplate(nameplate: dict, motor_voltage: dict,
                                    freq_hz: float = 60.0) -> float:
    """
    Estima un prior para Lm sólo con placa y voltaje.

    Idea:
    1) A partir de Pout, eficiencia y PF, se estima la corriente de línea nominal.
    2) Se asume que la corriente magnetizante Im es una fracción (imag_frac)
       de la corriente nominal (típico 25–40%).
    3) Se usa V_phase_peak ≈ ω_e * Lm * Im_peak  =>  Lm ≈ V / (ω I).
    """
    Pout = float(nameplate.get("rated_power_kw", 2.2)) * 1000.0  # W
    eta  = float(nameplate.get("rated_efficiency", 0.85))
    pf   = float(nameplate.get("power_factor", 0.82))

    V_ll_rms = float(nameplate.get("rated_voltage_ll",
                                   MOTOR_VOLTAGE.get("V_line_rms", 220.0)))

    # Potencia aparente de entrada S = Pout / (eta * pf)
    S_in = Pout / max(eta * pf, 1e-6)
    I_line_rms = S_in / (math.sqrt(3.0) * V_ll_rms + 1e-6)

    imag_frac = float(PRIORS_CONFIG.get("imag_frac", 0.30))
    I_mag_rms = imag_frac * I_line_rms
    I_mag_peak = math.sqrt(2.0) * I_mag_rms

    # V fase pico: ya lo tienes en MOTORVOLTAGE["vqs"]
    V_phase_peak = float(motor_voltage.get("vqs", 179.6))
    omega_e = 2.0 * math.pi * freq_hz

    Lm_prior = V_phase_peak / max(omega_e * I_mag_peak, 1e-12)

    # Asegurar que cae dentro de los límites configurados para Lm
    lm_low, lm_high = BOUNDS[4]  # índice 4 es Lm según PARAMNAMES
    Lm_prior = float(max(lm_low, min(lm_high, Lm_prior)))

    return Lm_prior
