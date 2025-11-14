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
    setup_pytorch,
    load_measurement_data,
    create_output_directory,
    compute_b_prior_from_nameplate,
    compute_lm_prior_from_nameplate  
)

from experiment_manager import ExperimentManager
from motor_dynamic_batch import InductionMotorModelBatch
from BIA_algorithms_parallel import (
    genetic_algorithm_parallel,
    particle_swarm_optimization_parallel,
    cma_es_parallel,
    hybrid_pso_lbfgs_parallel,
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
                            torque_measured, config,
                            b_prior=None, b_prior_weight=0.0,
                            lm_prior=None, lm_prior_weight=0.0):
    # Pesos de señales
    w_current = 0.2
    w_rpm     = 0.5
    w_torque  = 0.1

    device = current_measured.device
    bounds = config.get('bounds_tensor', None)
    l2_weight = config.get('l2_weight', 5e-3)

    # Si quieres que B NO entre en la L2, pon el último peso a 0.0
    l2_param_weights = th.tensor(
        [1., 1., 1., 1., 1., 1., 0.],
        device=device, dtype=th.float32
    )

    # Valores nominales para normalizar (TRUE_PARAMS en orden PARAM_NAMES)
    nominal_params = th.tensor(
        [TRUE_PARAMS[name] for name in PARAM_NAMES],
        device=device, dtype=th.float32
    )

    t_sim  = th.linspace(0.0, float(config['time_total']),
                         int(config['time_steps']), device=device)
    t_meas = th.linspace(0.0,
                         float(config.get('meas_time_total',
                                          config['time_total'])),
                         int(len(current_measured)), device=device)

    cur_meas_1  = current_measured.flatten().unsqueeze(0)
    rpm_meas_1  = rpm_measured.flatten().unsqueeze(0)
    torq_meas_1 = torque_measured.flatten().unsqueeze(0)

    # Si quieres fijar B al prior duro, déjalo así; si no, pon B_fixed = None
    B_fixed = b_prior

    def fitness_function(params_batch: th.Tensor) -> th.Tensor:
        # 1) Clamping
        if bounds is not None:
            min_b = bounds[:, 0].unsqueeze(0)
            max_b = bounds[:, 1].unsqueeze(0)
            params_batch = th.clamp(params_batch, min=min_b, max=max_b)

        # 2) Congelar B ANTES de actualizar el modelo y de integrar
        if B_fixed is not None:
            params_batch = params_batch.clone()
            params_batch[:, 6] = B_fixed

        # 3) Integrar
        model_batch.update_params_batch(params_batch)
        x0 = th.zeros(params_batch.shape[0], 5, dtype=th.float32, device=device)
        sol = odeint(model_batch, x0, t_sim,
                     method=config['ode_method'],
                     rtol=config['rtol'],
                     atol=config['atol']).permute(1, 0, 2)

        current_sim = model_batch.calculate_stator_current(sol)
        rpm_sim     = model_batch.calculate_rpm(sol)
        torque_sim  = model_batch.calculate_torque(sol)

        cur_meas_t  = _linear_interp_1d_torch(t_sim, t_meas, cur_meas_1.to(device))
        rpm_meas_t  = _linear_interp_1d_torch(t_sim, t_meas, rpm_meas_1.to(device))
        torq_meas_t = _linear_interp_1d_torch(t_sim, t_meas, torq_meas_1.to(device))

        eps = 1e-12
        cur_den  = th.mean(cur_meas_t**2)  + eps
        rpm_den  = th.mean(rpm_meas_t**2)  + eps
        torq_den = th.mean(torq_meas_t**2) + eps

        err_cur = th.mean((current_sim - cur_meas_t)**2, dim=1) / cur_den
        err_rpm = th.mean((rpm_sim   - rpm_meas_t)**2, dim=1) / rpm_den
        err_tor = th.mean((torque_sim- torq_meas_t)**2, dim=1) / torq_den

        total = w_current*err_cur + w_rpm*err_rpm + w_torque*err_tor

        # --- ÚNICA L2: desviación relativa respecto a nominal ---
        if l2_weight > 0.0:
            rel = params_batch / nominal_params.unsqueeze(0)   # p / p_nom
            deviation = rel - 1.0                              # 0 si coincide
            l2_penalty = l2_weight * th.mean(
                l2_param_weights.unsqueeze(0) * deviation**2,
                dim=1
            )
            total = total + l2_penalty

        # --- Prior sobre B (si además quieres prior suave) ---
        if b_prior is not None and b_prior_weight > 0.0:
            B_vals = params_batch[:, 6]
            prior_pen = (B_vals - b_prior) / (b_prior + 1e-12)
            prior_pen = prior_pen**2
            total = total + b_prior_weight * prior_pen

        # --- Prior explícito sobre Lm ---
        if lm_prior is not None and lm_prior_weight > 0.0:
            Lm_vals = params_batch[:, 4]   # índice de Lm según PARAM_NAMES
            lm_pen = (Lm_vals - lm_prior) / (lm_prior + 1e-12)
            lm_pen = lm_pen**2
            total = total + lm_prior_weight * lm_pen

        return total

    return fitness_function


def main():
    """
    Función principal - Ejecuta Fase 1
    """
    print("\n" + "="*70)
    print("FASE 1: VALIDACIÓN CON DATOS SINTÉTICOS")
    print("="*70)
    
    # 1. Configurar PyTorch
    device = setup_pytorch(PYTORCH_CONFIG)
    
    # 2. Cargar datos de mediciones
    current_measured, rpm_measured, torque_measured = load_measurement_data(
        DATA_FILES, device
    )
    
    # 3. Crear modelo del motor (batch)
    print("\nInicializando modelo del motor...")
    model_batch = InductionMotorModelBatch(
        vqs=MOTOR_VOLTAGE['vqs'],
        vds=MOTOR_VOLTAGE['vds']
    ).to(device)
    print("-> Modelo creado y movido a GPU")

    #*********
    # 4.1 Prior de B y límites locales
    b_prior = compute_b_prior_from_nameplate(NAMEPLATE)  # prior físico a partir de placa
    lm_prior = compute_lm_prior_from_nameplate(NAMEPLATE, MOTOR_VOLTAGE)

    bounds_local = copy.deepcopy(BOUNDS)
    if PRIORS_CONFIG['use_b_prior']:
        b_lo = PRIORS_CONFIG['b_bounds_low']  * b_prior
        b_hi = PRIORS_CONFIG['b_bounds_high'] * b_prior
        bounds_local[6] = [float(b_lo), float(b_hi)]

    # 4.2 Config local con clamping interno y ODE más sensible
    config_local = copy.deepcopy(OPTIMIZATION_CONFIG)
    config_local['bounds_tensor'] = th.tensor(bounds_local, dtype=th.float32, device=device)
    config_local['ode_method'] = 'dopri5'   # más estable para efectos sutiles
    config_local['rtol'] = 1e-4             # antes 1e-3
    config_local['atol'] = 1e-5             # antes 1e-4
    config_local['l2_weight'] = 3e-4        # reduce sesgo L2 contra B pequeño
    #*********

    # 4. Crear función de fitness (con prior de B)
    print("-> Creando función de fitness...")
    fitness_func = create_fitness_function(
        model_batch,
        current_measured,
        rpm_measured,
        torque_measured,
        config_local,
        b_prior=b_prior,
        b_prior_weight=PRIORS_CONFIG["b_prior_weight"],
        lm_prior=lm_prior if PRIORS_CONFIG.get("use_lm_prior", False) else None,
        lm_prior_weight=PRIORS_CONFIG.get("lm_prior_weight", 0.0),
    )

    
    # 5. Configurar gestor de experimentos
    phase_config = PHASES['phase1']
    exp_manager = ExperimentManager(
        output_dir=phase_config['output_dir'],
        phase_name=phase_config['name']
    )
    
    # 6. Definir algoritmos a ejecutar
    algorithms = {
        #"GA": genetic_algorithm_parallel,
        #"PSO": particle_swarm_optimization_parallel,
        #"CMAES": cma_es_parallel,
        "HYBRID_PSO_LBFGS": hybrid_pso_lbfgs_parallel,
    }
    
    # 7. Ejecutar todos los experimentos
    print("Iniciando experimentos...")
    result_files = exp_manager.run_all_algorithms(
        algorithms,
        fitness_func,
        bounds_local,  # usar los límites locales (B estrechado)
        dim=7
    )

    
    # 8. Resumen final
    print("\n" + "="*70)
    print("-> FASE 1 COMPLETADA")
    print("="*70)
    print("\nArchivos generados:")
    for algo, filepath in result_files.items():
        print(f"  • {filepath}")
    
    print(f"\n-> Todos los resultados en: {phase_config['output_dir']}")
    print("\n-> Analizar:")
    print("   python analyze_results.py --phase phase1")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()