import numpy as np
import torch as th
from torchdiffeq import odeint
from motor_dynamic import InductionMotorModelBatch
from BIA_algorithms import (
    genetic_algorithm_parallel,
    particle_swarm_optimization_parallel,
    differential_evolution_parallel,
    cuckoo_search_parallel,
)

# CONFIGURACIÓN GENERAL
device = 'cuda' if th.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

# Cargar datos
current_measured = th.tensor(np.loadtxt('current_measured.txt'), dtype=th.float32, device=device)
rpm_measured = th.tensor(np.loadtxt('rpm_measured.txt'), dtype=th.float32, device=device)
torque_measured = th.tensor(np.loadtxt('torque_measured.txt'), dtype=th.float32, device=device)

print(f"Datos cargados: {len(current_measured)} puntos")
print(f"Rango corriente: [{current_measured.min():.2f}, {current_measured.max():.2f}] A")
print(f"Rango RPM: [{rpm_measured.min():.2f}, {rpm_measured.max():.2f}]")
print(f"Rango torque: [{torque_measured.min():.2f}, {torque_measured.max():.2f}] N·m")

# ============================
# FUNCIÓN FITNESS EN BATCH
# ============================

bounds = [
    [0.05, 2.0],     # rs 
    [0.1, 2.0],      # rr 
    [0.0001, 0.01],  # Lls 
    [0.0001, 0.01],  # Llr 
    [0.01, 0.20],    # Lm 
    [0.01, 0.20],    # J 
    [0.00001, 0.001] # B
]

max_fes = 5000
pop_size = 50

# Crear modelo batch (una sola vez)
model_batch = InductionMotorModelBatch(vqs=220.0, vds=0.0).to(device)

def motor_fitness_batch(params_batch):
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
    t = th.linspace(0, 0.1, 200, device=device)
    
    # Resolver ODEs para todos los motores simultáneamente
    # sol shape: (batch_size, time_steps, 5)
    sol = odeint(model_batch, x0, t, method='rk4', rtol=1e-3, atol=1e-4)
    
    # Transponer para tener (batch_size, time_steps, 5)
    sol = sol.permute(1, 0, 2)
    
    # Calcular señales
    current_sim = model_batch.calculate_stator_current(sol)  # (batch_size, time_steps)
    rpm_sim = model_batch.calculate_rpm(sol)
    torque_sim = model_batch.calculate_torque(sol)
    
    # Alinear con mediciones
    n_points = current_sim.shape[1]
    idx = th.linspace(0, len(current_measured) - 1, n_points).long()
    
    cur_meas = current_measured[idx].unsqueeze(0)  # (1, n_points)
    rpm_meas = rpm_measured[idx].unsqueeze(0)
    torq_meas = torque_measured[idx].unsqueeze(0)
    
    # Calcular errores normalizados para cada motor
    error_current = th.mean((current_sim - cur_meas)**2, dim=1) / th.mean(cur_meas**2)
    error_rpm = th.mean((rpm_sim - rpm_meas)**2, dim=1) / th.mean(rpm_meas**2)
    error_torque = th.mean((torque_sim - torq_meas)**2, dim=1) / th.mean(torq_meas**2)
    
    total_error = error_current + error_rpm + error_torque
    
    return total_error

# ============================
# OPTIMIZACIÓN
# ============================

print("\n=== Ejecutando optimización PARALELA ===")

print("\n[1/4] Genetic Algorithm (Parallel)...")
best_ga, fit_ga, hist_ga = genetic_algorithm_parallel(
    motor_fitness_batch, dim=7, bounds=bounds,
    max_fes=max_fes, pop_size=pop_size
)

print("\n[2/4] Particle Swarm Optimization (Parallel)...")
best_pso, fit_pso, hist_pso = particle_swarm_optimization_parallel(
    motor_fitness_batch, dim=7, bounds=bounds,
    max_fes=max_fes, pop_size=pop_size
)

print("\n[3/4] Differential Evolution (Parallel)...")
best_de, fit_de, hist_de = differential_evolution_parallel(
    motor_fitness_batch, dim=7, bounds=bounds,
    max_fes=max_fes, pop_size=pop_size
)

print("\n[4/4] Cuckoo Search (Parallel)...")
best_cs, fit_cs, hist_cs = cuckoo_search_parallel(
    motor_fitness_batch, dim=7, bounds=bounds,
    max_fes=max_fes, pop_size=pop_size
)

# ============================
# RESULTADOS
# ============================

true_params = [0.435, 0.816, 0.002, 0.002, 0.0693, 0.089, 0.0001]

print("\n" + "="*60)
print("RESULTADOS DE OPTIMIZACIÓN")
print("="*60)

param_names = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']

print(f"\n{'Algoritmo':<25} {'Error Final':<15}")
print("-"*40)
print(f"{'Genetic Algorithm':<25} {fit_ga:.8f}")
print(f"{'PSO':<25} {fit_pso:.8f}")
print(f"{'Differential Evolution':<25} {fit_de:.8f}")
print(f"{'Cuckoo Search':<25} {fit_cs:.8f}")

print(f"\n{'Parámetro':<10} {'GA':<12} {'PSO':<12} {'DE':<12} {'CS':<12} {'Verdadero':<12}")
print("-"*72)
for i, name in enumerate(param_names):
    print(f"{name:<10} {best_ga[i]:.6f}   {best_pso[i]:.6f}   {best_de[i]:.6f}   {best_cs[i]:.6f}   {true_params[i]:.6f}")

print("\n" + "-"*72)

# Porcentaje del error
print(f"\n{'Parámetro':<10} {'GA (%)':<12} {'PSO (%)':<12} {'DE (%)':<12} {'CS (%)':<12}")
print("-"*60)
for i, name in enumerate(param_names):
    err_ga = abs((best_ga[i] - true_params[i]) / true_params[i]) * 100
    err_pso = abs((best_pso[i] - true_params[i]) / true_params[i]) * 100
    err_de = abs((best_de[i] - true_params[i]) / true_params[i]) * 100
    err_cs = abs((best_cs[i] - true_params[i]) / true_params[i]) * 100
    print(f"{name:<10} {err_ga:.4f}     {err_pso:.4f}     {err_de:.4f}     {err_cs:.4f}")

print("\n" + "="*60)