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
def cma_es_parallel(
    fitness_func_batch,
    dim,
    bounds,
    max_fes=None,
    pop_size=None,
    mu=None,         
    sigma0=None,
    **kwargs,        
):
    """
    CMA-ES continuo, versión vectorizada en PyTorch.
    Se adapta muy bien a problemas de baja dimensión como este.
    """

    cma_cfg = ALGORITHM_CONFIGS.get("CMAES", {})
    max_fes = max_fes if max_fes is not None else OPTIMIZATION_CONFIG.get("max_fes", 10000)

    bounds_t = th.tensor(bounds, dtype=th.float32, device=device)
    lower, upper = bounds_t[:, 0], bounds_t[:, 1]
    span = upper - lower

    # Tamaño de población (lambda) típico: 4 + int(3 * log(dim))
    if pop_size is None:
        pop_size = cma_cfg.get("pop_size", 4 + int(3 * math.log(dim + 1)))
    lam = pop_size

    # Parámetros de pesos (μ y pesos recombinación)
    mu = cma_cfg.get("mu", lam // 2)
    weights = th.tensor(
        [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)],
        dtype=th.float32,
        device=device
    )
    weights = weights / weights.sum()
    mu_eff = (weights.sum() ** 2) / (weights ** 2).sum()

    # Parámetros de adaptación estándar CMA-ES
    c_sigma = cma_cfg.get("c_sigma", (mu_eff + 2) / (dim + mu_eff + 5))
    d_sigma = cma_cfg.get("d_sigma", 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma)
    c_c = cma_cfg.get("c_c", (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim))
    c1 = cma_cfg.get("c1", 2 / ((dim + 1.3) ** 2 + mu_eff))
    c_mu = cma_cfg.get("c_mu", min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff)))

    # Sigma inicial
    if sigma0 is None:
        sigma0 = cma_cfg.get("sigma0", 0.3)
    sigma = sigma0 * span.mean().item()

    # Inicializar media en el centro del hipercubo
    m = (lower + upper) / 2.0

    # Matriz de covarianza inicial identidad
    C = th.eye(dim, device=device)
    p_sigma = th.zeros(dim, device=device)
    p_c = th.zeros(dim, device=device)

    history = []
    fes = 0

    # Factor de expectativa de norma de N(0, I)
    expected_norm = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim * dim))

    best_solution = None
    best_fitness = float("inf")

    while fes < max_fes:
        # -----------------------
        # 1) Muestreo de población
        # -----------------------
        # Descomponer C = B D^2 B^T
        eigvals, eigvecs = th.linalg.eigh(C)
        eigvals = th.clamp(eigvals, min=1e-12)
        D = th.sqrt(eigvals)
        B = eigvecs
        BD = B * D.unsqueeze(0)  # columnas escaladas

        # Generar lam muestras: x_k = m + sigma * BD * z_k
        z = th.randn(lam, dim, device=device)
        y = z @ BD.T
        xs = m.unsqueeze(0) + sigma * y
        xs = th.clamp(xs, lower, upper)

        # -----------------------
        # 2) Evaluación en batch
        # -----------------------
        if fes + lam <= max_fes:
            fit = fitness_func_batch(xs)
            fes += lam
        else:
            remaining = max_fes - fes
            fit_part = fitness_func_batch(xs[:remaining])
            pad = th.full((lam - remaining,), float("inf"), device=device)
            fit = th.cat([fit_part, pad])
            fes = max_fes

        # Actualizar mejor global
        cur_best_idx = th.argmin(fit)
        cur_best_fit = fit[cur_best_idx].item()
        if cur_best_fit < best_fitness:
            best_fitness = cur_best_fit
            best_solution = xs[cur_best_idx].clone()

        history.append(best_fitness)

        # -----------------------
        # 3) Ordenar y actualizar distribución
        # -----------------------
        sorted_idx = th.argsort(fit)
        elites = xs[sorted_idx[:mu]]
        z_elites = z[sorted_idx[:mu]]

        # Nueva media
        m_old = m.clone()
        m = (weights.unsqueeze(1) * elites).sum(dim=0)

        # Actualización de caminos
        y_w = (weights.unsqueeze(1) * z_elites).sum(dim=0)

        # Camino de sigma
        C_inv_sqrt = B @ th.diag(1.0 / D) @ B.T
        p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (m - m_old) / sigma)

        # Actualización de sigma
        norm_p_sigma = p_sigma.norm()
        sigma *= math.exp((c_sigma / d_sigma) * (norm_p_sigma / expected_norm - 1))

        # Camino de covarianza
        h_sigma_cond = norm_p_sigma / math.sqrt(1 - (1 - c_sigma) ** (2 * fes / lam)) < (1.4 + 2 / (dim + 1)) * expected_norm
        h_sigma = 1.0 if h_sigma_cond else 0.0
        p_c = (1 - c_c) * p_c + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * ((m - m_old) / sigma)

        # Actualización de C
        rank_one = p_c.unsqueeze(1) @ p_c.unsqueeze(0)
        rank_mu = th.zeros_like(C)
        for k in range(mu):
            dz = z_elites[k].unsqueeze(1)
            rank_mu += weights[k] * (dz @ dz.T)

        C = (1 - c1 - c_mu) * C + c1 * rank_one + c_mu * rank_mu

        # Forzar simetría numérica
        C = 0.5 * (C + C.T)

        print(f"CMA-ES FEs: {fes}, Best Fitness: {best_fitness:.6f}", end="\r")

    return best_solution.detach().cpu(), best_fitness, history


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
        max_local_fes=max_fes_local if max_fes_local > 0 else None
    )

    history = pso_history + local_history

    return x_opt.detach().cpu(), float(f_opt), history


def _lbfgs_local_refinement(
    fitness_func_batch,
    x0: th.Tensor,
    lower: th.Tensor,
    upper: th.Tensor,
    max_iters: int = 30,
    lr: float = 1.0,
    max_local_fes: Optional[int] = None,   # o simplemente max_local_fes=None
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

    def closure():
        nonlocal eval_count
        optimizer.zero_grad()

        x_clamped = th.clamp(x, lower, upper)
        f = fitness_func_batch(x_clamped.unsqueeze(0))[0]
        f.backward()

        eval_count += 1
        history.append(f.item())

        # Aquí podrías comprobar max_local_fes si quisieras cortar antes
        return f

    optimizer.step(closure)

    x_clamped = th.clamp(x.detach(), lower, upper)
    f_final = fitness_func_batch(x_clamped.unsqueeze(0))[0].item()
    history.append(f_final)

    return x_clamped, f_final, history
