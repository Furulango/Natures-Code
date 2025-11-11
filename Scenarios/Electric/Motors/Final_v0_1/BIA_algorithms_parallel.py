import math
import torch as th
import numpy as np
from config import OPTIMIZATION_CONFIG, ALGORITHM_CONFIGS

device = 'cuda' if th.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# -----------------------------
# Genetic Algorithm (GA)
# -----------------------------
def genetic_algorithm_parallel(
    fitness_func_batch, dim, bounds,
    max_fes=None, pop_size=None, pc=None, pm=None, tournament_size=None,
    eta_c=None, eta_m=None, immigrants_frac=None, stagnation_gens=None
):
    ga_cfg = ALGORITHM_CONFIGS.get('GA', {})
    max_fes = max_fes if max_fes is not None else OPTIMIZATION_CONFIG.get('max_fes', 10000)
    pop_size = pop_size if pop_size is not None else ga_cfg.get('pop_size', 80)
    pc = pc if pc is not None else ga_cfg.get('pc', 0.8)
    pm = pm if pm is not None else ga_cfg.get('pm', 0.2)  # ↑ mutación por mayor diversidad
    tournament_size = tournament_size if tournament_size is not None else ga_cfg.get('tournament_size', 2)
    eta_c = eta_c if eta_c is not None else ga_cfg.get('eta_c', 20)
    eta_m = eta_m if eta_m is not None else ga_cfg.get('eta_m', 20)
    immigrants_frac = immigrants_frac if immigrants_frac is not None else ga_cfg.get('immigrants_frac', 0.1)
    stagnation_gens = stagnation_gens if stagnation_gens is not None else ga_cfg.get('stagnation_gens', 6)

    bounds = th.tensor(bounds, dtype=th.float32, device=device)
    lower, upper = bounds[:, 0], bounds[:, 1]

    # Inicialización
    population = th.rand(pop_size, dim, device=device) * (upper - lower) + lower
    fitness = fitness_func_batch(population)
    fes = pop_size

    best_idx = th.argmin(fitness)
    best_solution = population[best_idx].clone()
    best_fitness = fitness[best_idx].item()
    history = [best_fitness]
    no_improve = 0

    def tournament_selection(pop, fit, k):
        idx = th.randint(0, len(pop), (k,), device=device)
        winner = idx[th.argmin(fit[idx])]
        return pop[winner].clone()

    def sbx(parent1, parent2):
        u = th.rand_like(parent1)
        beta = th.where(u <= 0.5, (2*u)**(1/(eta_c+1)), (1/(2*(1-u)))**(1/(eta_c+1)))
        child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
        child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
        return th.clamp(child1, lower, upper), th.clamp(child2, lower, upper)

    def poly_mut(individual):
        mutated = individual.clone()
        for i in range(dim):
            if th.rand(1).item() < pm:
                u = th.rand(1).item()
                if u < 0.5:
                    delta = (2*u)**(1/(eta_m+1)) - 1
                else:
                    delta = 1 - (2*(1-u))**(1/(eta_m+1))
                mutated[i] = mutated[i] + delta * (upper[i] - lower[i])
                # reflexión en límites
                if mutated[i] < lower[i]:
                    over = lower[i] - mutated[i]
                    mutated[i] = lower[i] + over
                if mutated[i] > upper[i]:
                    over = mutated[i] - upper[i]
                    mutated[i] = upper[i] - over
        return th.clamp(mutated, lower, upper)

    while fes < max_fes:
        # Genera hijos
        offsprings = th.empty_like(population)
        for i in range(0, pop_size, 2):
            p1 = tournament_selection(population, fitness, tournament_size)
            p2 = tournament_selection(population, fitness, tournament_size)
            if th.rand(1).item() < pc:
                c1, c2 = sbx(p1, p2)
            else:
                c1, c2 = p1.clone(), p2.clone()
            offsprings[i] = poly_mut(c1)
            if i + 1 < pop_size:
                offsprings[i+1] = poly_mut(c2)

        # Evalúa hijos con presupuesto
        if fes + pop_size <= max_fes:
            off_fitness = fitness_func_batch(offsprings)
            fes += pop_size
        else:
            remaining = max_fes - fes
            off_fitness = fitness_func_batch(offsprings[:remaining])
            # Relleno para mantener shape
            pad = fitness[:pop_size-remaining]
            off_fitness = th.cat([off_fitness, pad])
            fes = max_fes

        # μ+λ: selección de los mejores
        combined = th.cat([population, offsprings], dim=0)
        combined_fit = th.cat([fitness, off_fitness], dim=0)
        best_indices = th.argsort(combined_fit)[:pop_size]
        population = combined[best_indices]
        fitness = combined_fit[best_indices]

        # Actualiza mejor global
        cur_idx = th.argmin(fitness)
        cur_fit = fitness[cur_idx].item()
        if cur_fit + 1e-12 < best_fitness:
            best_fitness = cur_fit
            best_solution = population[cur_idx].clone()
            no_improve = 0
        else:
            no_improve += 1

        # Elitismo sincronizado
        worst_idx = th.argmax(fitness)
        population[worst_idx] = best_solution
        fitness[worst_idx] = th.tensor(best_fitness, device=device)

        # Inmigración anti-estancamiento
        if immigrants_frac > 0 and no_improve >= stagnation_gens:
            k = max(1, int(pop_size * immigrants_frac))
            worst_idxs = th.argsort(fitness, descending=True)[:k]
            population[worst_idxs] = th.rand(k, dim, device=device) * (upper - lower) + lower
            fitness[worst_idxs] = fitness_func_batch(population[worst_idxs])
            fes = min(max_fes, fes + k)
            no_improve = 0

        history.append(best_fitness)
        print(f"FEs: {fes}, Best Fitness: {best_fitness:.6f}", end='\r')

    return best_solution.detach().cpu(), best_fitness, history

# -----------------------------
# Particle Swarm Optimization (PSO)
# -----------------------------
def particle_swarm_optimization_parallel(
    fitness_func_batch, dim, bounds,
    max_fes=None, pop_size=None, w=None, c1=None, c2=None,
    w_start=None, w_end=None, vmax_frac=None
):
    pso_cfg = ALGORITHM_CONFIGS.get('PSO', {})
    max_fes = max_fes if max_fes is not None else OPTIMIZATION_CONFIG.get('max_fes', 10000)
    pop_size = pop_size if pop_size is not None else pso_cfg.get('pop_size', 60)
    # parámetros clásicos
    w = w if w is not None else pso_cfg.get('w', 0.7298)
    c1 = c1 if c1 is not None else pso_cfg.get('c1', 1.49618)
    c2 = c2 if c2 is not None else pso_cfg.get('c2', 1.49618)
    # mejoras: inercia decreciente + clamp de velocidad
    w_start = w_start if w_start is not None else pso_cfg.get('w_start', 0.8)
    w_end = w_end if w_end is not None else pso_cfg.get('w_end', 0.5)
    vmax_frac = vmax_frac if vmax_frac is not None else pso_cfg.get('vmax_frac', 0.1)

    bounds = th.tensor(bounds, dtype=th.float32, device=device)
    lower, upper = bounds[:, 0], bounds[:, 1]
    span = (upper - lower)

    particles = th.rand(pop_size, dim, device=device) * span + lower
    velocities = th.randn(pop_size, dim, device=device) * (span * 0.05)

    pbest = particles.clone()
    pbest_fitness = fitness_func_batch(particles)
    fes = pop_size

    gbest_idx = th.argmin(pbest_fitness)
    gbest = pbest[gbest_idx].clone()
    gbest_fitness = pbest_fitness[gbest_idx].item()
    history = [gbest_fitness]

    vmax = vmax_frac * span

    while fes < max_fes:
        # Inercia decreciente según progreso de presupuesto
        progress = float(fes) / float(max_fes)
        w_eff = w_start - (w_start - w_end) * progress

        r1 = th.rand(pop_size, dim, device=device)
        r2 = th.rand(pop_size, dim, device=device)
        velocities = (w_eff * velocities +
                      c1 * r1 * (pbest - particles) +
                      c2 * r2 * (gbest.unsqueeze(0) - particles))
        # Clamp de velocidad
        velocities = th.clamp(velocities, -vmax, vmax)
        # Actualización de posiciones con clamp en límites
        particles = th.clamp(particles + velocities, lower, upper)

        # Evaluación respetando presupuesto
        if fes + pop_size <= max_fes:
            fitness = fitness_func_batch(particles)
            fes += pop_size
        else:
            remaining = max_fes - fes
            fitness = fitness_func_batch(particles[:remaining])
            fes = max_fes

        # Actualizar pbest y gbest
        n_eval = fitness.shape[0]
        improved = fitness < pbest_fitness[:n_eval]
        pbest_fitness[:n_eval] = th.where(improved, fitness, pbest_fitness[:n_eval])
        pbest[:n_eval] = th.where(improved.unsqueeze(1), particles[:n_eval], pbest[:n_eval])

        current_best_idx = th.argmin(pbest_fitness)
        if pbest_fitness[current_best_idx] < gbest_fitness:
            gbest_fitness = pbest_fitness[current_best_idx].item()
            gbest = pbest[current_best_idx].clone()

        history.append(gbest_fitness)
        print(f"FEs: {fes}, Best Fitness: {gbest_fitness:.6f}", end='\r')

    return gbest.cpu(), gbest_fitness, history

# -----------------------------
# Differential Evolution (DE)
# -----------------------------
def differential_evolution_parallel(
    fitness_func_batch, dim, bounds,
    max_fes=None, pop_size=None, F=None, CR=None,
    jitter=None
):
    de_cfg = ALGORITHM_CONFIGS.get('DE', {})
    max_fes = max_fes if max_fes is not None else OPTIMIZATION_CONFIG.get('max_fes', 10000)
    pop_size = pop_size if pop_size is not None else de_cfg.get('pop_size', 80)
    F = F if F is not None else de_cfg.get('F', 0.5)
    CR = CR if CR is not None else de_cfg.get('CR', 0.9)
    jitter = jitter if jitter is not None else de_cfg.get('jitter', 0.1)  # factor de jitter en F

    bounds = th.tensor(bounds, dtype=th.float32, device=device)
    lower, upper = bounds[:, 0], bounds[:, 1]

    population = th.rand(pop_size, dim, device=device) * (upper - lower) + lower
    fitness = fitness_func_batch(population)
    fes = pop_size

    best_idx = th.argmin(fitness)
    best_solution = population[best_idx].clone()
    best_fitness = fitness[best_idx].item()
    prev_best = best_fitness
    history = [best_fitness]

    while fes < max_fes:
        trials = th.zeros_like(population)
        # Jitter adaptativo de F por iteración para romper simetrías
        F_eff = F * (1.0 + jitter * th.randn(1, device=device).item())

        for i in range(pop_size):
            indices = th.randperm(pop_size, device=device)
            indices = indices[indices != i][:3]
            r1, r2, r3 = indices
            mutant = th.clamp(population[r1] + F_eff * (population[r2] - population[r3]), lower, upper)
            trial = population[i].clone()
            cross_points = th.rand(dim, device=device) < CR
            if not cross_points.any():
                cross_points[th.randint(0, dim, (1,), device=device)] = True
            trial[cross_points] = mutant[cross_points]
            trials[i] = trial

        if fes + pop_size <= max_fes:
            trial_fitness = fitness_func_batch(trials)
            fes += pop_size
        else:
            remaining = max_fes - fes
            trial_fitness = fitness_func_batch(trials[:remaining])
            trial_fitness = th.cat([trial_fitness, fitness[remaining:]])
            fes = max_fes

        improved = trial_fitness < fitness
        population = th.where(improved.unsqueeze(1), trials, population)
        fitness = th.where(improved, trial_fitness, fitness)

        current_best_idx = th.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx].item()
            best_solution = population[current_best_idx].clone()

        # Si no mejora respecto al mejor global previo, aumentar ligeramente F para explorar
        if best_fitness >= prev_best - 1e-12:
            F = min(0.9, F * 1.02)
        else:
            F = max(0.3, F * 0.99)
        prev_best = best_fitness

        history.append(best_fitness)
        print(f"FEs: {fes}, Best Fitness: {best_fitness:.6f}", end='\r')

    return best_solution.cpu(), best_fitness, history

# -----------------------------
# Cuckoo Search (CS)
# -----------------------------
def cuckoo_search_parallel(
    fitness_func_batch, dim, bounds,
    max_fes=None, pop_size=None, pa=None, beta=None,
    step_scale=None
):
    cs_cfg = ALGORITHM_CONFIGS.get('CS', {})
    max_fes = max_fes if max_fes is not None else OPTIMIZATION_CONFIG.get('max_fes', 10000)
    pop_size = pop_size if pop_size is not None else cs_cfg.get('pop_size', 50)
    pa = pa if pa is not None else cs_cfg.get('pa', 0.25)
    beta = beta if beta is not None else cs_cfg.get('beta', 1.5)
    step_scale = step_scale if step_scale is not None else cs_cfg.get('step_scale', 0.02)  # ↑ paso base

    bounds = th.tensor(bounds, dtype=th.float32, device=device)
    lower, upper = bounds[:, 0], bounds[:, 1]

    nests = th.rand(pop_size, dim, device=device) * (upper - lower) + lower
    fitness = fitness_func_batch(nests)
    fes = pop_size

    best_idx = th.argmin(fitness)
    best_nest = nests[best_idx].clone()
    best_fitness = fitness[best_idx].item()
    history = [best_fitness]

    while fes < max_fes:
        # Programación suave del abandono para explorar más al inicio
        progress = float(fes) / float(max_fes)
        pa_eff = min(0.6, max(0.15, pa + 0.2 * (0.5 - progress)))  # más abandono al inicio

        steps = levy_flight_batch(pop_size, dim, beta, device)
        step_sizes = step_scale * steps * (nests - best_nest.unsqueeze(0))
        new_nests = th.clamp(nests + step_sizes * th.randn(pop_size, dim, device=device), lower, upper)

        if fes + pop_size <= max_fes:
            new_fitness = fitness_func_batch(new_nests)
            fes += pop_size
        else:
            remaining = max_fes - fes
            new_fitness = fitness_func_batch(new_nests[:remaining])
            new_fitness = th.cat([new_fitness, fitness[remaining:]])
            fes = max_fes

        improved = new_fitness < fitness
        nests = th.where(improved.unsqueeze(1), new_nests, nests)
        fitness = th.where(improved, new_fitness, fitness)

        current_best_idx = th.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx].item()
            best_nest = nests[current_best_idx].clone()

        # Abandono de peores nidos
        num_abandon = int(pa_eff * pop_size)
        if num_abandon > 0:
            if fes + num_abandon <= max_fes:
                worst_indices = th.argsort(fitness, descending=True)[:num_abandon]
                nests[worst_indices] = th.rand(num_abandon, dim, device=device) * (upper - lower) + lower
                fitness[worst_indices] = fitness_func_batch(nests[worst_indices])
                fes += num_abandon
            # actualizar mejor
            current_best_idx = th.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx].item()
                best_nest = nests[current_best_idx].clone()

        history.append(best_fitness)
        print(f"FEs: {fes}, Best Fitness: {best_fitness:.6f}", end='\r')

    return best_nest.cpu(), best_fitness, history

# Helpers: Lévy flights
def levy_flight_batch(batch_size, dim, beta, device):
    # Estimación de sigma_u para Lévy (Mantegna)
    sigma_u = (th.exp(th.lgamma(th.tensor(1+beta, device=device)) -
                      th.lgamma(th.tensor((1+beta)/2, device=device))) *
               th.sin(th.tensor(np.pi*beta/2, device=device)) /
               (((beta-1)/2) * beta * 2**((beta-1)/2)))**(1/beta)
    u = th.randn(batch_size, dim, device=device) * sigma_u
    v = th.randn(batch_size, dim, device=device)
    step = u / th.abs(v)**(1/beta)
    return step
