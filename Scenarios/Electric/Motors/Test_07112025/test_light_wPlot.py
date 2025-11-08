# ============================================================================
# PIPELINE COMPLETO: Optimización y Visualización
# ============================================================================

import numpy as np
import torch as th
from torchdiffeq import odeint
import motor_dynamic
from BIA_algorithms import (genetic_algorithm, particle_swarm_optimization,
                               differential_evolution, cuckoo_search)
from plot_optimization_results import generate_all_plots

device = 'cuda' if th.cuda.is_available() else 'cpu'

# ============================================================================
# 1. CARGAR DATOS MEDIDOS
# ============================================================================
print("-"*60)
print("CARGANDO DATOS")
print("-"*60)

current_measured = th.tensor(np.loadtxt('current_measured.txt'), dtype=th.float32).to(device)
rpm_measured = th.tensor(np.loadtxt('rpm_measured.txt'), dtype=th.float32).to(device)
torque_measured = th.tensor(np.loadtxt('torque_measured.txt'), dtype=th.float32).to(device)

print(f" Datos cargados: {len(current_measured)} puntos")
print(f"  Corriente: [{current_measured.min():.2f}, {current_measured.max():.2f}] A")
print(f"  RPM: [{rpm_measured.min():.2f}, {rpm_measured.max():.2f}]")
print(f"  Torque: [{torque_measured.min():.2f}, {torque_measured.max():.2f}] N·m")

# ============================================================================
# 2. DEFINIR FUNCIÓN OBJETIVO
# ============================================================================

def motor_fitness(params):
    """
    Función objetivo: minimizar error entre simulación y mediciones
    """
    params_list = params.cpu().tolist()
    
    # Crear modelo con parámetros candidatos
    model = motor_dynamic.InductionMotorModel(params_list, vqs=220.0, vds=0.0).to(device)
    
    # Condiciones iniciales
    x0 = th.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=th.float64, device=device)
    t = th.linspace(0, 1, 1000, device=device)
    
    # Resolver ODE
    sol = odeint(model, x0, t)
    
    # Calcular variables simuladas
    current_sim = model.calculate_stator_current(sol).float()
    rpm_sim = model.calculate_rpm(sol).float()
    torque_sim = model.calculate_torque(sol).float()
    
    # Normalizar errores (para que tengan peso similar)
    error_current = th.mean((current_sim - current_measured)**2) / (th.mean(current_measured**2) + 1e-8)
    error_rpm = th.mean((rpm_sim - rpm_measured)**2) / (th.mean(rpm_measured**2) + 1e-8)
    error_torque = th.mean((torque_sim - torque_measured)**2) / (th.mean(torque_measured**2) + 1e-8)
    
    # Pesos (ajusta según tu criterio)
    w1, w2, w3 = 1.0, 1.0, 1.0
    
    total_error = w1*error_current + w2*error_rpm + w3*error_torque
    
    return total_error

# 3. CONFIGURAR OPTIMIZACIÓN

# Bounds de parámetros [rs, rr, Lls, Llr, Lm, J, B]
bounds = [
    [0.1, 1.0],      # rs (resistencia estator)
    [0.3, 1.5],      # rr (resistencia rotor)
    [0.0005, 0.005], # Lls (inductancia fuga estator)
    [0.0005, 0.005], # Llr (inductancia fuga rotor)
    [0.03, 0.15],    # Lm (inductancia magnetización)
    [0.03, 0.20],    # J (inercia)
    [0.00001, 0.001] # B (fricción)
]

max_fes = 50  # Total de evaluaciones 
pop_size = 10  # Tamaño de población

# EJECUTAR ALGORITMOS DE OPTIMIZACIÓN

print("\n" + "-"*60)
print("EJECUTANDO ALGORITMOS ")
print("-"*60)
print(f"Evaluaciones por algoritmo: {max_fes}")

print("\n[1/4] Genetic Algorithm:")
best_ga, fit_ga, hist_ga = genetic_algorithm(
    motor_fitness, dim=7, bounds=bounds, max_fes=max_fes, pop_size=pop_size
)

print("[2/4] Particle Swarm Optimization:")
best_pso, fit_pso, hist_pso = particle_swarm_optimization(
    motor_fitness, dim=7, bounds=bounds, max_fes=max_fes, pop_size=pop_size
)

print("[3/4] Differential Evolution:")
best_de, fit_de, hist_de = differential_evolution(
    motor_fitness, dim=7, bounds=bounds, max_fes=max_fes, pop_size=pop_size
)

print("[4/4] Cuckoo Search:")
best_cs, fit_cs, hist_cs = cuckoo_search(
    motor_fitness, dim=7, bounds=bounds, max_fes=max_fes, pop_size=pop_size
)

# MOSTRAR RESULTADOS NUMÉRICOS

print("\n" + "-"*60)
print("RESULTADOS")
print("-"*60)

param_names = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
true_params = [0.435, 0.816, 0.002, 0.002, 0.0693, 0.089, 0.0001]

print(f"\n{'Algoritmo':<30} {'Error Final':<15}")
print("-"*45)
print(f"{'Genetic Algorithm':<30} {fit_ga:.8f}")
print(f"{'Particle Swarm Optimization':<30} {fit_pso:.8f}")
print(f"{'Differential Evolution':<30} {fit_de:.8f}")
print(f"{'Cuckoo Search':<30} {fit_cs:.8f}")


print(f"\n{'Parámetro':<10} {'GA':<12} {'PSO':<12} {'DE':<12} {'CS':<12} {'Verdadero':<12}")
print("-"*70)
for i, name in enumerate(param_names):
    print(f"{name:<10} {best_ga[i]:.6f}   {best_pso[i]:.6f}   {best_de[i]:.6f}   {best_cs[i]:.6f}   {true_params[i]:.6f}")

print("\n" + "-"*60)

# Mostrar porcentaje de error contra valores verdaderos
print("PORCENTAJE DE ERROR RESPECTO A VALORES VERDADEROS")
print("-"*60)
for i, name in enumerate(param_names):
    err_ga = abs((best_ga[i] - true_params[i]) / true_params[i]) * 100
    err_pso = abs((best_pso[i] - true_params[i]) / true_params[i]) * 100
    err_de = abs((best_de[i] - true_params[i]) / true_params[i]) * 100
    err_cs = abs((best_cs[i] - true_params[i]) / true_params[i]) * 100
    print(f"{name:<10} GA: {err_ga:.2f}% | PSO: {err_pso:.2f}% | DE: {err_de:.2f}% | CS: {err_cs:.2f}%")
print("\n" + "="*60)


print(f"\n{'Parámetro':<10} {'Verdadero':<12}")
print("-"*22)
for i, name in enumerate(param_names):
    print(f"{name:<10} {true_params[i]:.6f}")

# GENERAR TODOS LOS GRÁFICOS

generate_all_plots(
    hist_ga, hist_pso, hist_de, hist_cs,
    best_ga, best_pso, best_de, best_cs,
    fit_ga, fit_pso, fit_de, fit_cs,
    current_measured, rpm_measured, torque_measured,
    true_params=true_params
)

print("\n" + "-"*60)
print("PIPELINE COMPLETADO")
print("-"*60)