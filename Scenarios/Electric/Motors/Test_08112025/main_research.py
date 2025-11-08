"""
CONFIGURACIÓN ÓPTIMA PARA INVESTIGACIÓN CON MÚLTIPLES RUNS
===========================================================

Objetivo: 30-50 runs × 4 algoritmos × 3 escenarios
Prioridades: 
1. Reproducibilidad
2. Calidad de resultados
3. Tiempo razonable
4. Estadísticas robustas
"""

import torch as th
import numpy as np
import json
from datetime import datetime
import time

# ============================================
# CONFIGURACIÓN RECOMENDADA
# ============================================

# Optimizaciones de PyTorch
th.backends.cudnn.benchmark = True
th.backends.cudnn.deterministic = False  # Más rápido (no determinista)
th.set_float32_matmul_precision('high')  # Tensor Cores

device = 'cuda' if th.cuda.is_available() else 'cpu'

# PARÁMETROS BALANCEADOS (calidad vs velocidad)
RESEARCH_CONFIG = {
    'max_fes': 10000,        # Suficiente para convergencia
    'pop_size': 80,          # Balance entre 50 y 100
    'time_steps': 120,       # Balance entre 100 y 200
    'num_runs': 30,          # Estándar para papers
    'num_scenarios': 3,
    
    # Tolerancias del ODE (ligeramente relajadas)
    'rtol': 1e-3,
    'atol': 1e-4,
    
    # Seeds para reproducibilidad
    'base_seed': 42,
}

# Configuraciones por escenario
SCENARIOS = {
    'scenario_1_nominal': {
        'description': 'Condiciones nominales',
        'noise_level': 0.0,
        'vqs': 220.0,
        'vds': 0.0,
    },
    'scenario_2_noisy': {
        'description': 'Datos con ruido (5%)',
        'noise_level': 0.05,
        'vqs': 220.0,
        'vds': 0.0,
    },
    'scenario_3_undervoltage': {
        'description': 'Bajo voltaje (90%)',
        'noise_level': 0.0,
        'vqs': 198.0,  # 90% de 220V
        'vds': 0.0,
    }
}

# Configuraciones por algoritmo (ajustadas según literatura)
ALGORITHM_CONFIGS = {
    'GA': {
        'pop_size': 80,
        'pc': 0.8,
        'pm': 0.1,
        'tournament_size': 3,
        'expected_time_min': 3.5
    },
    'PSO': {
        'pop_size': 60,
        'w': 0.7298,
        'c1': 1.49618,
        'c2': 1.49618,
        'expected_time_min': 3.0
    },
    'DE': {
        'pop_size': 80,
        'F': 0.5,
        'CR': 0.9,
        'expected_time_min': 3.2
    },
    'CS': {
        'pop_size': 50,
        'pa': 0.25,
        'beta': 1.5,
        'expected_time_min': 3.8
    }
}

# ============================================
# SISTEMA DE GESTIÓN DE EXPERIMENTOS
# ============================================

class ExperimentManager:
    """
    Gestiona múltiples runs con reproducibilidad y logging
    """
    def __init__(self, config, output_dir='results'):
        self.config = config
        self.output_dir = output_dir
        self.results = {
            'metadata': {
                'date': datetime.now().isoformat(),
                'device': str(th.cuda.get_device_name(0)) if th.cuda.is_available() else 'CPU',
                'config': config
            },
            'scenarios': {}
        }
        
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def set_seed(self, seed):
        """Reproducibilidad"""
        th.manual_seed(seed)
        np.random.seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)
    
    def run_experiment(self, algorithm_name, algorithm_func, scenario_name, 
                       scenario_config, motor_fitness_batch, bounds):
        """
        Ejecuta múltiples runs de un algoritmo
        """
        num_runs = self.config['num_runs']
        base_seed = self.config['base_seed']
        
        print(f"\n{'='*70}")
        print(f"📊 {algorithm_name} - {scenario_name}")
        print(f"{'='*70}")
        
        results = {
            'best_fitness': [],
            'best_params': [],
            'convergence_history': [],
            'execution_time': [],
            'final_errors': []
        }
        
        total_start = time.time()
        
        for run in range(num_runs):
            # Seed único pero reproducible
            seed = base_seed + run * 1000
            self.set_seed(seed)
            
            run_start = time.time()
            
            # Ejecutar optimización
            best_params, best_fitness, history = algorithm_func(
                motor_fitness_batch,
                dim=7,
                bounds=bounds,
                max_fes=self.config['max_fes'],
                **ALGORITHM_CONFIGS[algorithm_name]
            )
            
            run_time = time.time() - run_start
            
            # Guardar resultados
            results['best_fitness'].append(best_fitness)
            results['best_params'].append(best_params.cpu().numpy().tolist())
            results['convergence_history'].append(history)
            results['execution_time'].append(run_time)
            
            # Progress
            elapsed = time.time() - total_start
            avg_time = elapsed / (run + 1)
            eta = avg_time * (num_runs - run - 1)
            
            print(f"Run {run+1:2d}/{num_runs}: "
                  f"Fitness={best_fitness:.6f}, "
                  f"Time={run_time:.1f}s, "
                  f"ETA={eta/60:.1f}min")
        
        # Estadísticas
        fitness_array = np.array(results['best_fitness'])
        results['statistics'] = {
            'mean': float(np.mean(fitness_array)),
            'std': float(np.std(fitness_array)),
            'median': float(np.median(fitness_array)),
            'min': float(np.min(fitness_array)),
            'max': float(np.max(fitness_array)),
            'q25': float(np.percentile(fitness_array, 25)),
            'q75': float(np.percentile(fitness_array, 75)),
            'total_time': time.time() - total_start
        }
        
        print(f"\n📈 Estadísticas:")
        print(f"   Mean ± Std: {results['statistics']['mean']:.6f} ± {results['statistics']['std']:.6f}")
        print(f"   Median: {results['statistics']['median']:.6f}")
        print(f"   Min: {results['statistics']['min']:.6f}")
        print(f"   Max: {results['statistics']['max']:.6f}")
        print(f"   Total time: {results['statistics']['total_time']/60:.1f} min")
        
        return results
    
    def save_results(self):
        """Guarda resultados en JSON"""
        filename = f"{self.output_dir}/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Resultados guardados en: {filename}")
        
    def generate_summary(self):
        """Genera resumen estadístico para el paper"""
        print("\n" + "="*70)
        print("📊 RESUMEN GENERAL PARA PAPER")
        print("="*70)
        
        for scenario_name, scenario_data in self.results['scenarios'].items():
            print(f"\n{scenario_name}:")
            print(f"{'Algoritmo':<10} {'Mean':<12} {'Std':<12} {'Median':<12} {'Best':<12}")
            print("-"*58)
            
            for algo_name, algo_data in scenario_data.items():
                stats = algo_data['statistics']
                print(f"{algo_name:<10} "
                      f"{stats['mean']:<12.6f} "
                      f"{stats['std']:<12.6f} "
                      f"{stats['median']:<12.6f} "
                      f"{stats['min']:<12.6f}")

# ============================================
# FUNCIÓN FITNESS CON SOPORTE PARA ESCENARIOS
# ============================================

def motor_fitness_batch_scenarios(params_batch, model_batch, 
                                   current_measured, rpm_measured, torque_measured,
                                   scenario_config):
    """
    Fitness function adaptada a diferentes escenarios
    """
    batch_size = params_batch.shape[0]
    
    model_batch.update_params_batch(params_batch)
    
    x0 = th.zeros(batch_size, 5, dtype=th.float32, device=params_batch.device)
    t = th.linspace(0, 0.1, RESEARCH_CONFIG['time_steps'], device=params_batch.device)
    
    sol = odeint(model_batch, x0, t, method='rk4', 
                 rtol=RESEARCH_CONFIG['rtol'], 
                 atol=RESEARCH_CONFIG['atol'])
    sol = sol.permute(1, 0, 2)
    
    # Calcular señales
    current_sim = model_batch.calculate_stator_current(sol)
    rpm_sim = model_batch.calculate_rpm(sol)
    torque_sim = model_batch.calculate_torque(sol)
    
    # Alinear con mediciones
    n_points = current_sim.shape[1]
    idx = th.linspace(0, len(current_measured) - 1, n_points).long()
    
    # Aplicar ruido si es necesario
    noise_level = scenario_config.get('noise_level', 0.0)
    if noise_level > 0:
        cur_meas = current_measured[idx].unsqueeze(0) * (1 + noise_level * th.randn_like(current_sim))
        rpm_meas = rpm_measured[idx].unsqueeze(0) * (1 + noise_level * th.randn_like(rpm_sim))
        torq_meas = torque_measured[idx].unsqueeze(0) * (1 + noise_level * th.randn_like(torque_sim))
    else:
        cur_meas = current_measured[idx].unsqueeze(0)
        rpm_meas = rpm_measured[idx].unsqueeze(0)
        torq_meas = torque_measured[idx].unsqueeze(0)
    
    # Errores normalizados
    error_current = th.mean((current_sim - cur_meas)**2, dim=1) / th.mean(cur_meas**2)
    error_rpm = th.mean((rpm_sim - rpm_meas)**2, dim=1) / th.mean(rpm_meas**2)
    error_torque = th.mean((torque_sim - torq_meas)**2, dim=1) / th.mean(torq_meas**2)
    
    total_error = error_current + error_rpm + error_torque
    
    return total_error

# ============================================
# ESTIMACIÓN DE TIEMPO TOTAL
# ============================================

def estimate_total_time():
    """
    Estima tiempo total de experimentos
    """
    total_runs = (len(ALGORITHM_CONFIGS) * 
                  RESEARCH_CONFIG['num_runs'] * 
                  RESEARCH_CONFIG['num_scenarios'])
    
    avg_time_per_run = 3.5  # minutos promedio
    total_time_min = total_runs * avg_time_per_run
    
    print("\n⏱️  ESTIMACIÓN DE TIEMPO TOTAL:")
    print(f"   Total runs: {total_runs}")
    print(f"   Tiempo por run: ~{avg_time_per_run:.1f} min")
    print(f"   Tiempo total: ~{total_time_min:.0f} min ({total_time_min/60:.1f} horas)")
    print(f"   Con paralelización: ~{total_time_min/2:.0f} min ({total_time_min/120:.1f} horas)")
    
    return total_time_min

# ============================================
# EJEMPLO DE USO COMPLETO
# ============================================

"""
# En tu main_parallel.py:

from research_config import (
    ExperimentManager, 
    RESEARCH_CONFIG, 
    SCENARIOS, 
    ALGORITHM_CONFIGS,
    motor_fitness_batch_scenarios,
    estimate_total_time
)

# Estimar tiempo
estimate_total_time()

# Crear gestor de experimentos
exp_manager = ExperimentManager(RESEARCH_CONFIG)

# Para cada escenario
for scenario_name, scenario_config in SCENARIOS.items():
    exp_manager.results['scenarios'][scenario_name] = {}
    
    # Crear función fitness para este escenario
    def fitness_func(params_batch):
        return motor_fitness_batch_scenarios(
            params_batch, model_batch,
            current_measured, rpm_measured, torque_measured,
            scenario_config
        )
    
    # Ejecutar cada algoritmo
    results_ga = exp_manager.run_experiment(
        'GA', genetic_algorithm_parallel, 
        scenario_name, scenario_config,
        fitness_func, bounds
    )
    exp_manager.results['scenarios'][scenario_name]['GA'] = results_ga
    
    # ... repetir para PSO, DE, CS

# Guardar y generar resumen
exp_manager.save_results()
exp_manager.generate_summary()
"""

print("\n" + "="*70)
print("✅ CONFIGURACIÓN DE INVESTIGACIÓN CARGADA")
print("="*70)
estimate_total_time()