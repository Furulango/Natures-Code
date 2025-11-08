import torch as th
import numpy as np

device = 'cuda' if th.cuda.is_available() else 'cpu'
print(f"Using device: {device}") 


def genetic_algorithm_parallel(fitness_func_batch, dim, bounds, max_fes=10000, pop_size=50, 
                               pc=0.8, pm=0.1, tournament_size=3):
    """
    Genetic Algorithm (GA) - PARALLEL VERSION
    
    Args:
        fitness_func_batch: Function that accepts (pop_size, dim) and returns (pop_size,) fitness
        dim: Problem dimensionality
        bounds: List of [min, max] for each dimension, shape (dim, 2)
        max_fes: Maximum function evaluations
        pop_size: Population size
        pc: Crossover probability
        pm: Mutation probability
        tournament_size: Tournament selection size
    
    Returns:
        best_solution, best_fitness, history
    """
    bounds = th.tensor(bounds, dtype=th.float32, device=device)
    lower, upper = bounds[:, 0], bounds[:, 1]
    
    # Initialize population
    population = th.rand(pop_size, dim, device=device) * (upper - lower) + lower
    
    # PARALLEL EVALUATION - All at once!
    fitness = fitness_func_batch(population)
    fes = pop_size
    
    best_idx = th.argmin(fitness)
    best_solution = population[best_idx].clone()
    best_fitness = fitness[best_idx].item()
    
    history = [best_fitness]
    
    # Main loop
    while fes < max_fes:
        new_population = th.zeros_like(population)
        
        for i in range(0, pop_size, 2):
            # Tournament selection
            parent1 = tournament_selection(population, fitness, tournament_size)
            parent2 = tournament_selection(population, fitness, tournament_size)
            
            # Crossover
            if th.rand(1).item() < pc:
                child1, child2 = sbx_crossover(parent1, parent2, lower, upper)
            else:
                child1, child2 = parent1.clone(), parent2.clone()
            
            # Mutation
            child1 = polynomial_mutation(child1, lower, upper, pm)
            child2 = polynomial_mutation(child2, lower, upper, pm)
            
            new_population[i] = child1
            if i + 1 < pop_size:
                new_population[i + 1] = child2
        
        # PARALLEL EVALUATION
        if fes + pop_size <= max_fes:
            fitness = fitness_func_batch(new_population)
            fes += pop_size
        else:
            # Evaluate remaining budget sequentially
            remaining = max_fes - fes
            for i in range(remaining):
                fitness[i] = fitness_func_batch(new_population[i:i+1]).item()
            fes = max_fes
        
        # Update best
        current_best_idx = th.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx].item()
            best_solution = new_population[current_best_idx].clone()
        
        population = new_population
        history.append(best_fitness)
        
        print(f"FEs: {fes}, Best Fitness: {best_fitness:.6f}", end='\r')
    
    return best_solution.cpu(), best_fitness, history


def particle_swarm_optimization_parallel(fitness_func_batch, dim, bounds, max_fes=10000, 
                                        pop_size=30, w=0.7298, c1=1.49618, c2=1.49618):
    """
    Particle Swarm Optimization (PSO) - PARALLEL VERSION
    """
    bounds = th.tensor(bounds, dtype=th.float32, device=device)
    lower, upper = bounds[:, 0], bounds[:, 1]
    
    # Initialize particles
    particles = th.rand(pop_size, dim, device=device) * (upper - lower) + lower
    velocities = th.rand(pop_size, dim, device=device) * (upper - lower) * 0.1
    
    # PARALLEL EVALUATION
    pbest = particles.clone()
    pbest_fitness = fitness_func_batch(particles)
    fes = pop_size
    
    # Global best
    gbest_idx = th.argmin(pbest_fitness)
    gbest = pbest[gbest_idx].clone()
    gbest_fitness = pbest_fitness[gbest_idx].item()
    
    history = [gbest_fitness]
    
    # Main loop
    while fes < max_fes:
        # Update all particles (vectorized)
        r1 = th.rand(pop_size, dim, device=device)
        r2 = th.rand(pop_size, dim, device=device)
        
        velocities = (w * velocities + 
                     c1 * r1 * (pbest - particles) +
                     c2 * r2 * (gbest.unsqueeze(0) - particles))
        
        particles = particles + velocities
        particles = th.clamp(particles, lower, upper)
        
        # PARALLEL EVALUATION
        if fes + pop_size <= max_fes:
            fitness = fitness_func_batch(particles)
            fes += pop_size
        else:
            remaining = max_fes - fes
            fitness = fitness_func_batch(particles[:remaining])
            fes = max_fes
        
        # Update personal bests (vectorized)
        improved = fitness < pbest_fitness[:len(fitness)]
        pbest_fitness[:len(fitness)] = th.where(improved, fitness, pbest_fitness[:len(fitness)])
        pbest[:len(fitness)] = th.where(improved.unsqueeze(1), particles[:len(fitness)], pbest[:len(fitness)])
        
        # Update global best
        current_best_idx = th.argmin(pbest_fitness)
        if pbest_fitness[current_best_idx] < gbest_fitness:
            gbest_fitness = pbest_fitness[current_best_idx].item()
            gbest = pbest[current_best_idx].clone()
        
        history.append(gbest_fitness)
        print(f"FEs: {fes}, Best Fitness: {gbest_fitness:.6f}", end='\r')
    
    return gbest.cpu(), gbest_fitness, history


def differential_evolution_parallel(fitness_func_batch, dim, bounds, max_fes=10000, 
                                    pop_size=50, F=0.5, CR=0.9):
    """
    Differential Evolution (DE) - PARALLEL VERSION
    """
    bounds = th.tensor(bounds, dtype=th.float32, device=device)
    lower, upper = bounds[:, 0], bounds[:, 1]
    
    # Initialize population
    population = th.rand(pop_size, dim, device=device) * (upper - lower) + lower
    
    # PARALLEL EVALUATION
    fitness = fitness_func_batch(population)
    fes = pop_size
    
    best_idx = th.argmin(fitness)
    best_solution = population[best_idx].clone()
    best_fitness = fitness[best_idx].item()
    
    history = [best_fitness]
    
    # Main loop
    while fes < max_fes:
        # Generate all mutations and trials at once
        trials = th.zeros_like(population)
        
        for i in range(pop_size):
            # Mutation
            indices = th.randperm(pop_size, device=device)[:3]
            while i in indices:
                indices = th.randperm(pop_size, device=device)[:3]
            
            r1, r2, r3 = indices
            mutant = population[r1] + F * (population[r2] - population[r3])
            mutant = th.clamp(mutant, lower, upper)
            
            # Crossover
            trial = population[i].clone()
            cross_points = th.rand(dim, device=device) < CR
            if not cross_points.any():
                cross_points[th.randint(0, dim, (1,))] = True
            trial[cross_points] = mutant[cross_points]
            
            trials[i] = trial
        
        # PARALLEL EVALUATION
        if fes + pop_size <= max_fes:
            trial_fitness = fitness_func_batch(trials)
            fes += pop_size
        else:
            remaining = max_fes - fes
            trial_fitness = fitness_func_batch(trials[:remaining])
            trial_fitness = th.cat([trial_fitness, fitness[remaining:]])
            fes = max_fes
        
        # Selection (vectorized)
        improved = trial_fitness < fitness
        population = th.where(improved.unsqueeze(1), trials, population)
        fitness = th.where(improved, trial_fitness, fitness)
        
        # Update best
        current_best_idx = th.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx].item()
            best_solution = population[current_best_idx].clone()
        
        history.append(best_fitness)
        print(f"FEs: {fes}, Best Fitness: {best_fitness:.6f}", end='\r')
    
    return best_solution.cpu(), best_fitness, history


def cuckoo_search_parallel(fitness_func_batch, dim, bounds, max_fes=10000, 
                          pop_size=25, pa=0.25, beta=1.5):
    """
    Cuckoo Search (CS) - PARALLEL VERSION
    """
    bounds = th.tensor(bounds, dtype=th.float32, device=device)
    lower, upper = bounds[:, 0], bounds[:, 1]
    
    # Initialize nests
    nests = th.rand(pop_size, dim, device=device) * (upper - lower) + lower
    
    # PARALLEL EVALUATION
    fitness = fitness_func_batch(nests)
    fes = pop_size
    
    best_idx = th.argmin(fitness)
    best_nest = nests[best_idx].clone()
    best_fitness = fitness[best_idx].item()
    
    history = [best_fitness]
    
    # Main loop
    while fes < max_fes:
        # Generate new nests via Levy flights (vectorized)
        steps = levy_flight_batch(pop_size, dim, beta, device)
        step_sizes = 0.01 * steps * (nests - best_nest.unsqueeze(0))
        new_nests = nests + step_sizes * th.randn(pop_size, dim, device=device)
        new_nests = th.clamp(new_nests, lower, upper)
        
        # PARALLEL EVALUATION
        if fes + pop_size <= max_fes:
            new_fitness = fitness_func_batch(new_nests)
            fes += pop_size
        else:
            remaining = max_fes - fes
            new_fitness = fitness_func_batch(new_nests[:remaining])
            new_fitness = th.cat([new_fitness, fitness[remaining:]])
            fes = max_fes
        
        # Selection (vectorized)
        improved = new_fitness < fitness
        nests = th.where(improved.unsqueeze(1), new_nests, nests)
        fitness = th.where(improved, new_fitness, fitness)
        
        # Update best
        current_best_idx = th.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx].item()
            best_nest = nests[current_best_idx].clone()
        
        # Abandon worst nests
        num_abandon = int(pa * pop_size)
        if num_abandon > 0 and fes + num_abandon <= max_fes:
            worst_indices = th.argsort(fitness, descending=True)[:num_abandon]
            nests[worst_indices] = th.rand(num_abandon, dim, device=device) * (upper - lower) + lower
            fitness[worst_indices] = fitness_func_batch(nests[worst_indices])
            fes += num_abandon
            
            current_best_idx = th.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx].item()
                best_nest = nests[current_best_idx].clone()
        
        history.append(best_fitness)
        print(f"FEs: {fes}, Best Fitness: {best_fitness:.6f}", end='\r')
    
    return best_nest.cpu(), best_fitness, history


# ============== HELPER FUNCTIONS ==============

def tournament_selection(population, fitness, tournament_size):
    """Tournament selection for GA"""
    indices = th.randint(0, len(population), (tournament_size,), device=device)
    tournament_fitness = fitness[indices]
    winner_idx = indices[th.argmin(tournament_fitness)]
    return population[winner_idx].clone()


def sbx_crossover(parent1, parent2, lower, upper, eta=20):
    """Simulated Binary Crossover (SBX)"""
    child1, child2 = parent1.clone(), parent2.clone()
    
    u = th.rand_like(parent1)
    beta = th.where(u <= 0.5, (2*u)**(1/(eta+1)), (1/(2*(1-u)))**(1/(eta+1)))
    
    child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
    
    child1 = th.clamp(child1, lower, upper)
    child2 = th.clamp(child2, lower, upper)
    
    return child1, child2


def polynomial_mutation(individual, lower, upper, pm, eta=20):
    """Polynomial mutation"""
    mutated = individual.clone()
    
    for i in range(len(individual)):
        if th.rand(1).item() < pm:
            u = th.rand(1).item()
            
            if u < 0.5:
                delta = (2*u)**(1/(eta+1)) - 1
            else:
                delta = 1 - (2*(1-u))**(1/(eta+1))
            
            mutated[i] = individual[i] + delta * (upper[i] - lower[i])
            mutated[i] = th.clamp(mutated[i], lower[i], upper[i])
    
    return mutated


def levy_flight(dim, beta, device):
    """Generate Levy flight step"""
    sigma_u = (th.exp(th.lgamma(th.tensor(1+beta, device=device)) - 
                      th.lgamma(th.tensor((1+beta)/2, device=device))) * 
               th.sin(th.tensor(np.pi*beta/2, device=device)) / 
               ((beta-1)/2 * beta * 2**((beta-1)/2)))**(1/beta)
    
    u = th.randn(dim, device=device) * sigma_u
    v = th.randn(dim, device=device)
    
    step = u / th.abs(v)**(1/beta)
    return step


def levy_flight_batch(batch_size, dim, beta, device):
    """Generate Levy flight steps for multiple solutions"""
    sigma_u = (th.exp(th.lgamma(th.tensor(1+beta, device=device)) - 
                      th.lgamma(th.tensor((1+beta)/2, device=device))) * 
               th.sin(th.tensor(np.pi*beta/2, device=device)) / 
               ((beta-1)/2 * beta * 2**((beta-1)/2)))**(1/beta)
    
    u = th.randn(batch_size, dim, device=device) * sigma_u
    v = th.randn(batch_size, dim, device=device)
    
    step = u / th.abs(v)**(1/beta)
    return step