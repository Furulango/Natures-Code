import numpy as np
import torch as th
from torchdiffeq import odeint
import motor_dynamic

device = 'cuda' if th.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

# ============================================================================
# CARGAR DATOS
# ============================================================================
print("-"*60)
print("CARGANDO DATOS")
print("-"*60)

current_measured = th.tensor(np.loadtxt('current_measured.txt'), dtype=th.float32, device=device)
rpm_measured = th.tensor(np.loadtxt('rpm_measured.txt'), dtype=th.float32, device=device)
torque_measured = th.tensor(np.loadtxt('torque_measured.txt'), dtype=th.float32, device=device)

print(f"Datos cargados: {len(current_measured)} puntos")
print(f"Corriente: [{current_measured.min():.2f}, {current_measured.max():.2f}] A")
print(f"RPM: [{rpm_measured.min():.2f}, {rpm_measured.max():.2f}]")
print(f"Torque: [{torque_measured.min():.2f}, {torque_measured.max():.2f}] N·m")

# ============================================================================
# FUNCIÓN OBJETIVO (Batch version)
# ============================================================================

def motor_fitness_batch(params_batch):
    """
    Evalúa la población completa en GPU en paralelo.
    params_batch: tensor [N, 7]
    """
    N = params_batch.shape[0]
    t = th.linspace(0, 1, len(current_measured), device=device)
    x0 = th.zeros(N, 5, device=device, dtype=th.float64)

    errors = th.zeros(N, device=device)

    for i in range(N):
        params = params_batch[i].cpu().tolist()
        model = motor_dynamic.InductionMotorModel(params, vqs=220.0, vds=0.0).to(device)
        sol = odeint(model, x0[i], t, method='rk4')
        current_sim = model.calculate_stator_current(sol).float()
        rpm_sim = model.calculate_rpm(sol).float()
        torque_sim = model.calculate_torque(sol).float()

        error_current = th.mean((current_sim - current_measured)**2) / (th.mean(current_measured**2) + 1e-8)
        error_rpm = th.mean((rpm_sim - rpm_measured)**2) / (th.mean(rpm_measured**2) + 1e-8)
        error_torque = th.mean((torque_sim - torque_measured)**2) / (th.mean(torque_measured**2) + 1e-8)

        errors[i] = error_current + error_rpm + error_torque

    return errors


# ============================================================================
# OPTIMIZADOR (versión rápida)
# ============================================================================
def simple_genetic_algorithm_batch(fitness_func, dim, bounds, max_fes=100, pop_size=10, pc=0.8, pm=0.1):
    bounds = th.tensor(bounds, dtype=th.float32, device=device)
    lower, upper = bounds[:, 0], bounds[:, 1]

    population = th.rand(pop_size, dim, device=device) * (upper - lower) + lower
    fitness = fitness_func(population)

    best_idx = th.argmin(fitness)
    best_solution = population[best_idx].clone()
    best_fitness = fitness[best_idx].item()
    history = [best_fitness]

    fes = pop_size
    while fes < max_fes:
        # Selección por torneo simplificada
        idx = th.randint(0, pop_size, (pop_size,), device=device)
        parents = population[idx]

        # Crossover (blend)
        alpha = th.rand(pop_size, dim, device=device)
        offspring = alpha * parents + (1 - alpha) * population

        # Mutación
        mutation_mask = th.rand_like(offspring) < pm
        random_mutation = th.rand_like(offspring) * (upper - lower) + lower
        offspring = th.where(mutation_mask, random_mutation, offspring)
        offspring = th.clamp(offspring, lower, upper)

        # Evaluar hijos
        fitness_off = fitness_func(offspring)
        fes += pop_size

        # Reemplazo elitista
        combined = th.cat([population, offspring], dim=0)
        combined_fit = th.cat([fitness, fitness_off], dim=0)
        sorted_idx = th.argsort(combined_fit)
        population = combined[sorted_idx[:pop_size]]
        fitness = combined_fit[sorted_idx[:pop_size]]

        best_idx = th.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx].item()
            best_solution = population[best_idx].clone()

        history.append(best_fitness)
        print(f"FEs: {fes}, Mejor fitness: {best_fitness:.6f}", end='\r')

    print()
    return best_solution.cpu(), best_fitness, history


# ============================================================================
# CONFIGURACIÓN Y EJECUCIÓN
# ============================================================================
bounds = [
    [0.1, 1.0], [0.3, 1.5], [0.0005, 0.005],
    [0.0005, 0.005], [0.03, 0.15], [0.03, 0.20], [0.00001, 0.001]
]

max_fes = 100
pop_size = 10

print("\nEJECUTANDO GA EN GPU (BATCH)...")
best_sol, best_fit, hist = simple_genetic_algorithm_batch(
    motor_fitness_batch, dim=7, bounds=bounds, max_fes=max_fes, pop_size=pop_size
)

print("\nRESULTADOS:")
param_names = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
for i, name in enumerate(param_names):
    print(f"{name:<8}: {best_sol[i]:.6f}")
print(f"\nFitness final: {best_fit:.6e}")
