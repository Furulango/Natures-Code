"""
FASE 1: Validación con datos sintéticos

Este script ejecuta 30 runs de cada algoritmo (GA, PSO, DE, CS)
y guarda los resultados en archivos JSON separados.

EJECUCIÓN:
    python main_phase1.py

SALIDA:
    results/phase1_synthetic/
        ├── GA_results.json
        ├── PSO_results.json
        ├── DE_results.json
        ├── CS_results.json
        └── summary_statistics.json
"""

import torch as th
import numpy as np
from torchdiffeq import odeint

# Imports de módulos locales
from config import (
    BOUNDS,
    MOTOR_VOLTAGE,
    OPTIMIZATION_CONFIG,
    PYTORCH_CONFIG,
    DATA_FILES,
    PHASES
)
from utils import (
    setup_pytorch,
    load_measurement_data,
    create_output_directory
)
from experiment_manager import ExperimentManager
from motor_dynamic_batch import InductionMotorModelBatch
from BIA_algorithms_parallel import (
    genetic_algorithm_parallel,
    particle_swarm_optimization_parallel,
    differential_evolution_parallel,
    cuckoo_search_parallel
)


def create_fitness_function(model_batch, current_measured, rpm_measured, 
                           torque_measured, config):
    """
    Crea la función de fitness para el motor
    
    Args:
        model_batch: Modelo del motor en batch
        current_measured: Corriente medida
        rpm_measured: RPM medido
        torque_measured: Torque medido
        config: Configuración de optimización
    
    Returns:
        Función de fitness que recibe (batch_size, 7) parámetros
    """
    device = current_measured.device
    
    def fitness_function(params_batch):
        """
        Evalúa múltiples conjuntos de parámetros simultáneamente
        
        Args:
            params_batch: Tensor (batch_size, 7) con parámetros
            
        Returns:
            fitness: Tensor (batch_size,) con errores
        """
        batch_size = params_batch.shape[0]
        
        # Actualizar parámetros del modelo
        model_batch.update_params_batch(params_batch)
        
        # Condiciones iniciales (batch_size, 5)
        x0 = th.zeros(batch_size, 5, dtype=th.float32, device=device)
        
        # Tiempo
        t = th.linspace(0, 0.1, config['time_steps'], device=device)
        
        # Resolver ODEs para todos los motores simultáneamente
        sol = odeint(
            model_batch, x0, t, 
            method=config['ode_method'],
            rtol=config['rtol'], 
            atol=config['atol']
        )
        
        # Transponer para tener (batch_size, time_steps, 5)
        sol = sol.permute(1, 0, 2)
        
        # Calcular señales
        current_sim = model_batch.calculate_stator_current(sol)
        rpm_sim = model_batch.calculate_rpm(sol)
        torque_sim = model_batch.calculate_torque(sol)
        
        # Alinear con mediciones
        n_points = current_sim.shape[1]
        idx = th.linspace(0, len(current_measured) - 1, n_points).long()
        
        cur_meas = current_measured[idx].unsqueeze(0)
        rpm_meas = rpm_measured[idx].unsqueeze(0)
        torq_meas = torque_measured[idx].unsqueeze(0)
    

        # Calcular errores normalizados
        error_current = th.mean((current_sim - cur_meas)**2, dim=1) / th.mean(cur_meas**2)
        error_rpm = th.mean((rpm_sim - rpm_meas)**2, dim=1) / th.mean(rpm_meas**2)
        error_torque = th.mean((torque_sim - torq_meas)**2, dim=1) / th.mean(torq_meas**2)
        
        # Si no funciona el escalar, cambiar a este
        #total_error = error_current + error_rpm + error_torque
        
        # Escalar basado en magnitudes (MSE absoluto ponderado)
        total_error = 1e-6 * error_current.abs() + 1e-3 * error_rpm.abs() + 0.1 * error_torque.abs()
        return total_error
    
    return fitness_function


def main():
    """
    Función principal - Ejecuta Fase 1
    """
    print("\n" + "="*70)
    print("🔬 FASE 1: VALIDACIÓN CON DATOS SINTÉTICOS")
    print("="*70)
    
    # 1. Configurar PyTorch
    device = setup_pytorch(PYTORCH_CONFIG)
    
    # 2. Cargar datos de mediciones
    current_measured, rpm_measured, torque_measured = load_measurement_data(
        DATA_FILES, device
    )
    
    # 3. Crear modelo del motor (batch)
    print("\n⚙️  Inicializando modelo del motor...")
    model_batch = InductionMotorModelBatch(
        vqs=MOTOR_VOLTAGE['vqs'],
        vds=MOTOR_VOLTAGE['vds']
    ).to(device)
    print("✓ Modelo creado y movido a GPU")
    
    # 4. Crear función de fitness
    print("✓ Creando función de fitness...")
    fitness_func = create_fitness_function(
        model_batch,
        current_measured,
        rpm_measured,
        torque_measured,
        OPTIMIZATION_CONFIG
    )
    
    # 5. Configurar gestor de experimentos
    phase_config = PHASES['phase1']
    exp_manager = ExperimentManager(
        output_dir=phase_config['output_dir'],
        phase_name=phase_config['name']
    )
    
    # 6. Definir algoritmos a ejecutar
    algorithms = {
        'GA': genetic_algorithm_parallel,
        'PSO': particle_swarm_optimization_parallel,
        'DE': differential_evolution_parallel,
        'CS': cuckoo_search_parallel
    }
    
    # 7. Ejecutar todos los experimentos
    print("\n🚀 Iniciando experimentos...\n")
    
    result_files = exp_manager.run_all_algorithms(
        algorithms,
        fitness_func,
        BOUNDS,
        dim=7
    )
    
    # 8. Resumen final
    print("\n" + "="*70)
    print("✅ FASE 1 COMPLETADA")
    print("="*70)
    print("\nArchivos generados:")
    for algo, filepath in result_files.items():
        print(f"  • {filepath}")
    
    print(f"\n📁 Todos los resultados en: {phase_config['output_dir']}")
    print("\n💡 Siguiente paso:")
    print("   python analyze_results.py --phase phase1")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
