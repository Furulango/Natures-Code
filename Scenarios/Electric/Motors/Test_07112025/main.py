import torch as th
from torchdiffeq import odeint
import motor_dynamic

# 1. Define tu función objetivo
def motor_fitness(params):
    """
    params: tensor con [rs, rr, Lls, Llr, Lm, J, B]
    Retorna: error escalar (suma ponderada de errores)
    """
    # Convertir a lista para el modelo
    params_list = params.tolist()
    
    # Crear modelo con nuevos parámetros
    model = motor_dynamic.InductionMotorModel(params_list, vqs=220.0, vds=0.0)
    
    # Condición inicial y tiempo
    x0 = th.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=th.float64)
    t = th.linspace(0, 1, 1000)
    
    # Resolver ODE
    sol = odeint(model, x0, t)
    
    # Calcular variables
    current_sim = model.calculate_stator_current(sol)
    rpm_sim = model.calculate_rpm(sol)
    torque_sim = model.calculate_torque(sol)
    
    # Error con datos medidos (MSE ponderado)
    w1, w2, w3 = 1.0, 1.0, 1.0  # Pesos
    error_current = th.mean((current_sim - current_measured)**2)
    error_rpm = th.mean((rpm_sim - rpm_measured)**2)
    error_torque = th.mean((torque_sim - torque_measured)**2)
    
    total_error = w1*error_current + w2*error_rpm + w3*error_torque
    
    return total_error

# 2. Define los bounds de tus parámetros
bounds = [
    [0.1, 1.0],   # rs
    [0.3, 1.5],   # rr
    [0.001, 0.01],  # Lls
    [0.001, 0.01],  # Llr
    [0.02, 0.15],   # Lm
    [0.01, 0.2],    # J
    [0.00001, 0.001]  # B
]

# 3. Ejecuta los algoritmos (comparación justa con mismo FES)
max_fes = 5000  # Total de evaluaciones

best_ga, fit_ga, hist_ga = genetic_algorithm(motor_fitness, dim=7, bounds=bounds, max_fes=max_fes)
best_pso, fit_pso, hist_pso = particle_swarm_optimization(motor_fitness, dim=7, bounds=bounds, max_fes=max_fes)
best_de, fit_de, hist_de = differential_evolution(motor_fitness, dim=7, bounds=bounds, max_fes=max_fes)
best_cs, fit_cs, hist_cs = cuckoo_search(motor_fitness, dim=7, bounds=bounds, max_fes=max_fes)

print(f"GA  - Fitness: {fit_ga:.6f}, Params: {best_ga}")
print(f"PSO - Fitness: {fit_pso:.6f}, Params: {best_pso}")
print(f"DE  - Fitness: {fit_de:.6f}, Params: {best_de}")
print(f"CS  - Fitness: {fit_cs:.6f}, Params: {best_cs}")