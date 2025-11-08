
# Cargar datos medidos desde archivos

import numpy as np
import torch as th

# Cargar los datos (saltando las líneas de comentario)
current_measured = th.tensor(np.loadtxt('current_measured.txt'), dtype=th.float32)
rpm_measured = th.tensor(np.loadtxt('rpm_measured.txt'), dtype=th.float32)
torque_measured = th.tensor(np.loadtxt('torque_measured.txt'), dtype=th.float32)

print(f"Datos cargados: {len(current_measured)} puntos")
print(f"Rango corriente: [{current_measured.min():.2f}, {current_measured.max():.2f}] A")
print(f"Rango RPM: [{rpm_measured.min():.2f}, {rpm_measured.max():.2f}]")
print(f"Rango torque: [{torque_measured.min():.2f}, {torque_measured.max():.2f}] N·m")


# Función objetivo con datos reales

from torchdiffeq import odeint
import motor_dynamic

device = 'cuda' if th.cuda.is_available() else 'cpu'

# Mover datos a GPU si está disponible
current_measured = current_measured.to(device)
rpm_measured = rpm_measured.to(device)
torque_measured = torque_measured.to(device)

def motor_fitness(params):
    """
    Función objetivo para optimizar parámetros del motor
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
    
    # Normalizar errores para que tengan peso similar
    error_current = th.mean((current_sim - current_measured)**2) / th.mean(current_measured**2)
    error_rpm = th.mean((rpm_sim - rpm_measured)**2) / th.mean(rpm_measured**2)
    error_torque = th.mean((torque_sim - torque_measured)**2) / th.mean(torque_measured**2)
    
    # Pesos (ajusta según tu criterio)
    w1, w2, w3 = 1.0, 1.0, 1.0
    
    total_error = w1*error_current + w2*error_rpm + w3*error_torque
    
    return total_error


# Ejecutar optimización

from BIA_algorithms import genetic_algorithm, particle_swarm_optimization
from BIA_algorithms import differential_evolution, cuckoo_search

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

max_fes = 5000  # Evaluaciones de función

print("\n=== Ejecutando optimización ===")

print("\n[1/4] Genetic Algorithm...")
best_ga, fit_ga, hist_ga = genetic_algorithm(motor_fitness, dim=7, bounds=bounds, max_fes=max_fes, pop_size=50)

print("[2/4] Particle Swarm Optimization...")
best_pso, fit_pso, hist_pso = particle_swarm_optimization(motor_fitness, dim=7, bounds=bounds, max_fes=max_fes, pop_size=30)

print("[3/4] Differential Evolution...")
best_de, fit_de, hist_de = differential_evolution(motor_fitness, dim=7, bounds=bounds, max_fes=max_fes, pop_size=50)

print("[4/4] Cuckoo Search...")
best_cs, fit_cs, hist_cs = cuckoo_search(motor_fitness, dim=7, bounds=bounds, max_fes=max_fes, pop_size=25)


# Resultados

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

print(f"\n{'Parámetro':<10} {'GA':<12} {'PSO':<12} {'DE':<12} {'CS':<12}")
print("-"*58)
for i, name in enumerate(param_names):
    print(f"{name:<10} {best_ga[i]:.6f}   {best_pso[i]:.6f}   {best_de[i]:.6f}   {best_cs[i]:.6f}")