"""
FASE 1: Validación con datos sintéticos (con etapa mecánica previa)
"""
import copy
import torch as th
import numpy as np
from torchdiffeq import odeint

from config import (
    BOUNDS, MOTOR_VOLTAGE, OPTIMIZATION_CONFIG, PYTORCH_CONFIG,
    DATA_FILES, PHASES, TRUE_PARAMS, PARAM_NAMES,
    NAMEPLATE, PRIORS_CONFIG
)
from utils import (
    setup_pytorch, load_measurement_data, create_output_directory,
    compute_b_prior_from_nameplate
)
from experiment_manager import ExperimentManager
from motor_dynamic_batch import InductionMotorModelBatch
from BIA_algorithms_parallel import (
    genetic_algorithm_parallel,
    particle_swarm_optimization_parallel,
    differential_evolution_parallel,
    cuckoo_search_parallel
)

def _linear_interp_1d_torch(x_new: th.Tensor, x_old: th.Tensor, y_old: th.Tensor) -> th.Tensor:
    x_new = x_new.to(dtype=th.float32)
    x_old = x_old.to(dtype=th.float32)
    y_old = y_old.to(dtype=th.float32)
    x_new = th.clamp(x_new, x_old[0], x_old[-1])
    idx_hi = th.searchsorted(x_old, x_new, right=True).clamp(min=1, max=x_old.numel()-1)
    idx_lo = idx_hi - 1
    x0 = x_old[idx_lo]; x1 = x_old[idx_hi]
    w = (x_new - x0) / (x1 - x0 + 1e-12)
    if y_old.dim() == 1:
        y0 = y_old[idx_lo]; y1 = y_old[idx_hi]
        return y0 + (y1 - y0) * w
    B = y_old.shape[0]
    idx_lo_exp = idx_lo.unsqueeze(0).expand(B, -1)
    idx_hi_exp = idx_hi.unsqueeze(0).expand(B, -1)
    y0 = th.gather(y_old, 1, idx_lo_exp)
    y1 = th.gather(y_old, 1, idx_hi_exp)
    return y0 + (y1 - y0) * w.unsqueeze(0)

def create_fitness_function(model_batch, current_measured, rpm_measured,
                            torque_measured, config, b_prior=None, b_prior_weight=0.0):
    # Pesos de señales (recomendado para hacer visible la mecánica)
    w_current = 1e-5
    w_rpm = 0.7
    w_torque = 0.3

    device = current_measured.device
    bounds = config.get('bounds_tensor', None)
    l2_weight = config.get('l2_weight', 5e-3)
    l2_param_weights = th.tensor([1,1,1,1,1,0.5,2.0], device=device, dtype=th.float32)

    t_sim  = th.linspace(0.0, float(config['time_total']), int(config['time_steps']), device=device)
    t_meas = th.linspace(0.0, float(config.get('meas_time_total', config['time_total'])),
                         int(len(current_measured)), device=device)

    cur_meas_1 = current_measured.flatten().unsqueeze(0)
    rpm_meas_1 = rpm_measured.flatten().unsqueeze(0)
    torq_meas_1= torque_measured.flatten().unsqueeze(0)

    def fitness_function(params_batch: th.Tensor) -> th.Tensor:
        if bounds is not None:
            min_b = bounds[:, 0].unsqueeze(0); max_b = bounds[:, 1].unsqueeze(0)
            params_batch = th.clamp(params_batch, min=min_b, max=max_b)

        model_batch.update_params_batch(params_batch)
        x0 = th.zeros(params_batch.shape[0], 5, dtype=th.float32, device=device)
        sol = odeint(model_batch, x0, t_sim,
                     method=config['ode_method'], rtol=config['rtol'], atol=config['atol']).permute(1,0,2)

        current_sim = model_batch.calculate_stator_current(sol)
        rpm_sim     = model_batch.calculate_rpm(sol)
        torque_sim  = model_batch.calculate_torque(sol)

        cur_meas_t  = _linear_interp_1d_torch(t_sim, t_meas, cur_meas_1.to(device))
        rpm_meas_t  = _linear_interp_1d_torch(t_sim, t_meas, rpm_meas_1.to(device))
        torq_meas_t = _linear_interp_1d_torch(t_sim, t_meas, torq_meas_1.to(device))

        eps=1e-12
        cur_den  = th.mean(cur_meas_t**2)+eps
        rpm_den  = th.mean(rpm_meas_t**2)+eps
        torq_den = th.mean(torq_meas_t**2)+eps

        err_cur = th.mean((current_sim-cur_meas_t)**2, dim=1)/cur_den
        err_rpm = th.mean((rpm_sim   -rpm_meas_t)**2, dim=1)/rpm_den
        err_tor = th.mean((torque_sim-torq_meas_t)**2, dim=1)/torq_den

        total = w_current*err_cur + w_rpm*err_rpm + w_torque*err_tor

        # Regularización L2 selectiva
        rms = th.sqrt(th.mean(params_batch**2, dim=1, keepdim=True))+1e-8
        norm_params = params_batch / rms
        l2_penalty = l2_weight * th.mean(l2_param_weights.unsqueeze(0)*(norm_params**2), dim=1)
        total = total + l2_penalty

        # Regularización del prior de B (si se suministra)
        if b_prior is not None and b_prior_weight > 0.0:
            B = params_batch[:, 6]
            prior_pen = ((B - b_prior)/ (b_prior + 1e-12))**2
            total = total + b_prior_weight * prior_pen

        return total

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