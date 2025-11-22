import math
import torch as th
import numpy as np
from config import OPTIMIZATION_CONFIG, ALGORITHM_CONFIGS


from typing import Optional 


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
# CMA-ES (3er algoritmo)
# -----------------------------

def bee_memetic_parallel(
    fitness_func_batch,
    dim,
    bounds,
    max_fes=None,
    pop_size=None, 
    **kwargs,      
):
    """
    Algoritmo bioinspirado tipo 'abejas memético' en 3 niveles:
    1) Exploración global por abejas exploradoras.
    2) Exploración intermedia en sitios prometedores (vecindarios).
    3) Refinamiento local con L-BFGS en los mejores sitios.
    """

    bee_cfg = ALGORITHM_CONFIGS.get("BEE_MEMETIC", {})
    max_fes = max_fes if max_fes is not None else OPTIMIZATION_CONFIG.get("max_fes", 10000)

    # Config por defecto (puedes afinarlos en config.py)
    global_frac = bee_cfg.get("global_frac", 0.3)      # FEs para fase global
    neigh_frac  = bee_cfg.get("neigh_frac", 0.4)       # FEs para fase intermedia
    local_frac  = 1.0 - global_frac - neigh_frac       # resto para L-BFGS

    n_scouts    = bee_cfg.get("n_scouts", 200)         # muestras globales
    n_sites     = bee_cfg.get("n_sites", 10)           # sitios a explorar
    n_elite     = bee_cfg.get("n_elite", 3)            # sitios "élite"
    neigh_elite = bee_cfg.get("neigh_elite", 20)       # vecinos por sitio élite
    neigh_other = bee_cfg.get("neigh_other", 10)       # vecinos por sitios normales

    max_local_iters = bee_cfg.get("max_local_iters", 10)
    lbfgs_lr        = bee_cfg.get("lbfgs_lr", 1.0)

    bounds_t = th.tensor(bounds, dtype=th.float32, device=device)
    lower, upper = bounds_t[:, 0], bounds_t[:, 1]
    span = upper - lower

    history = []

    # -------------------------
    # 1) Fase global: exploradoras
    # -------------------------
    max_fes_global = int(max_fes * global_frac)
    max_fes_global = max(n_scouts, max_fes_global)

    # número real de scouts limitado por presupuesto
    n_scouts_eff = min(n_scouts, max_fes_global)
    scouts = th.rand(n_scouts_eff, dim, device=device) * span + lower
    
    fitness_scouts = fitness_func_batch(scouts)
    fes = n_scouts_eff

    # ordenar mejores sitios
    sorted_idx = th.argsort(fitness_scouts)
    best_idx_sites = sorted_idx[:min(n_sites, n_scouts_eff)]
    sites = scouts[best_idx_sites]
    sites_f = fitness_scouts[best_idx_sites]

    best_solution = sites[0].clone()
    best_fitness = sites_f[0].item()
    history.append(best_fitness)

    print(f"[BEE] Global FEs: {fes}, Best Fitness: {best_fitness:.6f}", end="\r")

    # -------------------------
    # 2) Fase intermedia: sitios y vecindarios
    # -------------------------
    max_fes_neigh = int(max_fes * neigh_frac)
    max_fes_neigh = max(0, max_fes_neigh)
    fes_neigh_used = 0

    # radio inicial de vecindario (por ejemplo 10% del rango)
    neigh_radius_init = 0.1 * span
    neigh_radius = neigh_radius_init.clone()

    # índice de sitios élite y normales
    n_sites_eff = sites.shape[0]
    n_elite_eff = min(n_elite, n_sites_eff)
    elite_sites = sites[:n_elite_eff]
    elite_f     = sites_f[:n_elite_eff]
    other_sites = sites[n_elite_eff:]
    other_f     = sites_f[n_elite_eff:]

    all_neigh_solutions = []
    all_neigh_fitness   = []

    while fes < max_fes and fes_neigh_used < max_fes_neigh and n_sites_eff > 0:
        # generar vecinos para sitios élite
        if elite_sites.shape[0] > 0:
            n_elite_sites = elite_sites.shape[0]
            n_neigh_elite = min(neigh_elite, (max_fes_neigh - fes_neigh_used) // max(1, n_elite_sites))
            if n_neigh_elite > 0:
                noise_elite = (th.rand(n_elite_sites, n_neigh_elite, dim, device=device) - 0.5) * 2.0
                neigh_elite_pop = elite_sites.unsqueeze(1) + noise_elite * neigh_radius.unsqueeze(0).unsqueeze(1)
                neigh_elite_pop = th.clamp(neigh_elite_pop, lower, upper)
                neigh_elite_pop = neigh_elite_pop.reshape(-1, dim)

                f_elite = fitness_func_batch(neigh_elite_pop)
                fes += neigh_elite_pop.shape[0]
                fes_neigh_used += neigh_elite_pop.shape[0]

                all_neigh_solutions.append(neigh_elite_pop)
                all_neigh_fitness.append(f_elite)

        # generar vecinos para sitios no élite
        if other_sites.shape[0] > 0 and fes_neigh_used < max_fes_neigh:
            n_other_sites = other_sites.shape[0]
            n_neigh_other = min(neigh_other, (max_fes_neigh - fes_neigh_used) // max(1, n_other_sites))
            if n_neigh_other > 0:
                noise_other = (th.rand(n_other_sites, n_neigh_other, dim, device=device) - 0.5) * 2.0
                neigh_other_pop = other_sites.unsqueeze(1) + noise_other * neigh_radius.unsqueeze(0).unsqueeze(1)
                neigh_other_pop = th.clamp(neigh_other_pop, lower, upper)
                neigh_other_pop = neigh_other_pop.reshape(-1, dim)

                f_other = fitness_func_batch(neigh_other_pop)
                fes += neigh_other_pop.shape[0]
                fes_neigh_used += neigh_other_pop.shape[0]

                all_neigh_solutions.append(neigh_other_pop)
                all_neigh_fitness.append(f_other)

        if len(all_neigh_solutions) == 0:
            break

        # actualizar mejor global con todos los vecinos generados hasta ahora
        neigh_solutions = th.cat(all_neigh_solutions, dim=0)
        neigh_fitness = th.cat(all_neigh_fitness, dim=0)
        cur_idx = th.argmin(neigh_fitness)
        cur_fit = neigh_fitness[cur_idx].item()
        if cur_fit < best_fitness:
            best_fitness = cur_fit
            best_solution = neigh_solutions[cur_idx].clone()

        history.append(best_fitness)
        print(f"[BEE] Neigh FEs: {fes}, Best Fitness: {best_fitness:.6f}", end="\r")

        # opcional: reducir radio de vecindario para exploración más fina
        neigh_radius = neigh_radius * 0.7

        # re-seleccionar sitios a partir de todos los vecinos
        all_sites = th.cat([sites, neigh_solutions], dim=0)
        all_sites_f = th.cat([sites_f, neigh_fitness], dim=0)
        sorted_idx = th.argsort(all_sites_f)
        best_idx_sites = sorted_idx[:min(n_sites, all_sites.shape[0])]
        sites = all_sites[best_idx_sites]
        sites_f = all_sites_f[best_idx_sites]

        n_sites_eff = sites.shape[0]
        if n_sites_eff == 0:
            break

        n_elite_eff = min(n_elite, n_sites_eff)
        elite_sites = sites[:n_elite_eff]
        elite_f     = sites_f[:n_elite_eff]
        other_sites = sites[n_elite_eff:]
        other_f     = sites_f[n_elite_eff:]

    # -------------------------
    # 3) Fase local: L-BFGS en mejores sitios
    # -------------------------
    max_fes_local_total = max_fes - fes
    if max_fes_local_total <= 0 or sites.shape[0] == 0:
        return best_solution.detach().cpu(), best_fitness, history

    n_local_starts = bee_cfg.get("n_local_starts", 3)
    n_local_starts = min(n_local_starts, sites.shape[0])

    max_fes_per_start = max_fes_local_total // n_local_starts

    for i in range(n_local_starts):
        x0 = sites[i]
        x_opt, f_opt, local_hist = _lbfgs_local_refinement(
            fitness_func_batch,
            x0,
            lower,
            upper,
            max_iters=max_local_iters,
            lr=lbfgs_lr,
            max_local_fes=max_fes_per_start,
            fes_start=fes,
        )
        # aprox: sumamos el número de evaluaciones locales
        fes += len(local_hist)
        if f_opt < best_fitness:
            best_fitness = f_opt
            best_solution = x_opt.clone()
        history.extend(local_hist)
        print(f"[BEE] Local FEs: {fes}, Best Fitness: {best_fitness:.6f}", end="\r")

    return best_solution.detach().cpu(), float(best_fitness), history


# -----------------------------
# Hybrid PSO + L-BFGS (4º algoritmo)
# -----------------------------
def hybrid_pso_lbfgs_parallel(
    fitness_func_batch, dim, bounds,
    max_fes=None, pop_size=None,
    w=None, c1=None, c2=None,
    w_start=None, w_end=None, vmax_frac=None,
    global_frac=None, max_local_iters=None,
    lbfgs_lr=None
):
    """
    Fase 1: PSO global en batch.
    Fase 2: refinamiento local con L-BFGS sobre la mejor solución encontrada.
    """

    hyb_cfg = ALGORITHM_CONFIGS.get("HYBRID_PSO_LBFGS", {})
    pso_cfg = ALGORITHM_CONFIGS.get("PSO", {})

    max_fes = max_fes if max_fes is not None else OPTIMIZATION_CONFIG.get("max_fes", 10000)
    pop_size = pop_size if pop_size is not None else pso_cfg.get("pop_size", 60)

    # Parámetros PSO
    w = w if w is not None else pso_cfg.get("w", 0.7298)
    c1 = c1 if c1 is not None else pso_cfg.get("c1", 1.49618)
    c2 = c2 if c2 is not None else pso_cfg.get("c2", 1.49618)
    w_start = w_start if w_start is not None else pso_cfg.get("w_start", 0.8)
    w_end = w_end if w_end is not None else pso_cfg.get("w_end", 0.5)
    vmax_frac = vmax_frac if vmax_frac is not None else pso_cfg.get("vmax_frac", 0.1)

    # Parámetros de hibridación
    global_frac = global_frac if global_frac is not None else hyb_cfg.get("global_frac", 0.8)
    max_local_iters = max_local_iters if max_local_iters is not None else hyb_cfg.get("max_local_iters", 30)
    lbfgs_lr = lbfgs_lr if lbfgs_lr is not None else hyb_cfg.get("lbfgs_lr", 1.0)

    max_fes_global = int(max_fes * global_frac)
    max_fes_global = max(pop_size, max_fes_global)
    max_fes_local = max_fes - max_fes_global

    # -------------------------
    # Fase global: PSO
    # -------------------------
    print("\n[HYBRID] Fase global PSO...")
    gbest, gbest_fitness, pso_history = particle_swarm_optimization_parallel(
        fitness_func_batch, dim, bounds,
        max_fes=max_fes_global,
        pop_size=pop_size,
        w=w, c1=c1, c2=c2,
        w_start=w_start, w_end=w_end,
        vmax_frac=vmax_frac
    )

    # -------------------------
    # Fase local: L-BFGS
    # -------------------------
    print("\n[HYBRID] Fase local L-BFGS...")
    bounds_t = th.tensor(bounds, dtype=th.float32, device=device)
    lower, upper = bounds_t[:, 0], bounds_t[:, 1]

    x0 = gbest.to(device=device, dtype=th.float32)

    x_opt, f_opt, local_history = _lbfgs_local_refinement(
        fitness_func_batch,
        x0,
        lower,
        upper,
        max_iters=max_local_iters,
        lr=lbfgs_lr,
        max_local_fes=max_fes_local if max_fes_local > 0 else None,
        fes_start=max_fes_global,
    )


    history = pso_history + local_history

    return x_opt.detach().cpu(), float(f_opt), history


def _lbfgs_local_refinement(
    fitness_func_batch,
    x0: th.Tensor,
    lower: th.Tensor,
    upper: th.Tensor,
    max_iters=10,
    lr=1.0,
    max_local_fes=None,
    fes_start=0,
):
    """
    Refinamiento local sobre un solo vector de parámetros usando L-BFGS de PyTorch.
    """

    x = x0.clone().detach().to(device)
    x.requires_grad_(True)

    optimizer = th.optim.LBFGS(
        [x],
        lr=lr,
        max_iter=max_iters,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    history = []
    eval_count = 0
    best_local = float("inf")  # <-- definida en el scope externo

    def closure():
        nonlocal eval_count, best_local  # ahora sí existe en el scope externo
        optimizer.zero_grad()
        x_clamped = th.clamp(x, lower, upper)
        f = fitness_func_batch(x_clamped.unsqueeze(0))[0]
        f.backward()
        eval_count += 1

        if f.item() < best_local:
            best_local = f.item()

        total_fes = fes_start + eval_count
        history.append(f.item())

        print(f"FEs: {total_fes}, Best Fitness: {best_local:.6e}", end="\r")
        return f

    for it in range(max_iters):
        if max_local_fes is not None and eval_count >= max_local_fes:
            print(f"\n[L-BFGS] Cortando por max_local_fes={max_local_fes}")
            break
        optimizer.step(closure)

    x_clamped = th.clamp(x.detach(), lower, upper)
    f_final = fitness_func_batch(x_clamped.unsqueeze(0))[0].item()
    history.append(f_final)
    print(f"\n[HYBRID L-BFGS] Final FEs: {fes_start + eval_count}, Best Fitness: {best_local:.6e}")
    return x_clamped, f_final, history

