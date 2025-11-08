import torch as th
import numpy as np

device = 'cuda' if th.cuda.is_available() else 'cpu'

def genetic_algorithm(fitness_func, dim, bounds, max_fes=10000, pop_size=50, 
                     pc=0.8, pm=0.1, tournament_size=3):
    """
    Genetic Algorithm (GA)
    
    Args:
        fitness_func: Function to minimize, receives tensor of shape (dim,)
        dim: Problem dimensionality
        bounds: List of [min, max] for each dimension, shape (dim, 2)
        max_fes: Maximum function evaluations (for fair comparison)
        pop_size: Population size
        pc: Crossover probability
        pm: Mutation probability
        tournament_size: Tournament selection size
    
    Returns:
        best_solution: Best parameters found (tensor)
        best_fitness: Best fitness value (float)
        history: List of best fitness per generation
    """
    bounds = th.tensor(bounds, dtype=th.float32, device=device)
    lower, upper = bounds[:, 0], bounds[:, 1]
    
    # Initialize population randomly
    population = th.rand(pop_size, dim, device=device) * (upper - lower) + lower
    fitness = th.zeros(pop_size, device=device)
    
    # Evaluate initial population
    fes = 0
    for i in range(pop_size):
        fitness[i] = fitness_func(population[i])
        fes += 1
    
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
        
        # Evaluate new population
        for i in range(pop_size):
            if fes >= max_fes:
                break
            fitness[i] = fitness_func(new_population[i])
            fes += 1
            
            # Update best
            if fitness[i] < best_fitness:
                best_fitness = fitness[i].item()
                best_solution = new_population[i].clone()
        
        population = new_population
        history.append(best_fitness)
    
    return best_solution.cpu(), best_fitness, history


def particle_swarm_optimization(fitness_func, dim, bounds, max_fes=10000, 
                                pop_size=30, w=0.7298, c1=1.49618, c2=1.49618):
    """
    Particle Swarm Optimization (PSO)
    
    Args:
        fitness_func: Function to minimize
        dim: Problem dimensionality
        bounds: List of [min, max] for each dimension
        max_fes: Maximum function evaluations
        pop_size: Swarm size
        w: Inertia weight
        c1: Cognitive coefficient
        c2: Social coefficient
    
    Returns:
        best_solution, best_fitness, history
    """
    bounds = th.tensor(bounds, dtype=th.float32, device=device)
    lower, upper = bounds[:, 0], bounds[:, 1]
    
    # Initialize particles and velocities
    particles = th.rand(pop_size, dim, device=device) * (upper - lower) + lower
    velocities = th.rand(pop_size, dim, device=device) * (upper - lower) * 0.1
    
    # Personal best
    pbest = particles.clone()
    pbest_fitness = th.zeros(pop_size, device=device)
    
    # Evaluate initial swarm
    fes = 0
    for i in range(pop_size):
        pbest_fitness[i] = fitness_func(particles[i])
        fes += 1
    
    # Global best
    gbest_idx = th.argmin(pbest_fitness)
    gbest = pbest[gbest_idx].clone()
    gbest_fitness = pbest_fitness[gbest_idx].item()
    
    history = [gbest_fitness]
    
    # Main loop
    while fes < max_fes:
        for i in range(pop_size):
            if fes >= max_fes:
                break
            
            # Update velocity
            r1 = th.rand(dim, device=device)
            r2 = th.rand(dim, device=device)
            
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (pbest[i] - particles[i]) +
                           c2 * r2 * (gbest - particles[i]))
            
            # Update position
            particles[i] = particles[i] + velocities[i]
            
            # Boundary handling
            particles[i] = th.clamp(particles[i], lower, upper)
            
            # Evaluate
            fitness = fitness_func(particles[i])
            fes += 1
            
            # Update personal best
            if fitness < pbest_fitness[i]:
                pbest_fitness[i] = fitness
                pbest[i] = particles[i].clone()
                
                # Update global best
                if fitness < gbest_fitness:
                    gbest_fitness = fitness.item()
                    gbest = particles[i].clone()
        
        history.append(gbest_fitness)
    
    return gbest.cpu(), gbest_fitness, history


def differential_evolution(fitness_func, dim, bounds, max_fes=10000, 
                          pop_size=50, F=0.5, CR=0.9):
    """
    Differential Evolution (DE/rand/1/bin)
    
    Args:
        fitness_func: Function to minimize
        dim: Problem dimensionality
        bounds: List of [min, max] for each dimension
        max_fes: Maximum function evaluations
        pop_size: Population size
        F: Mutation factor
        CR: Crossover rate
    
    Returns:
        best_solution, best_fitness, history
    """
    bounds = th.tensor(bounds, dtype=th.float32, device=device)
    lower, upper = bounds[:, 0], bounds[:, 1]
    
    # Initialize population
    population = th.rand(pop_size, dim, device=device) * (upper - lower) + lower
    fitness = th.zeros(pop_size, device=device)
    
    # Evaluate initial population
    fes = 0
    for i in range(pop_size):
        fitness[i] = fitness_func(population[i])
        fes += 1
    
    best_idx = th.argmin(fitness)
    best_solution = population[best_idx].clone()
    best_fitness = fitness[best_idx].item()
    
    history = [best_fitness]
    
    # Main loop
    while fes < max_fes:
        for i in range(pop_size):
            if fes >= max_fes:
                break
            
            # Mutation: DE/rand/1
            indices = th.randperm(pop_size, device=device)[:3]
            while i in indices:
                indices = th.randperm(pop_size, device=device)[:3]
            
            r1, r2, r3 = indices
            mutant = population[r1] + F * (population[r2] - population[r3])
            
            # Boundary handling
            mutant = th.clamp(mutant, lower, upper)
            
            # Crossover: binomial
            trial = population[i].clone()
            cross_points = th.rand(dim, device=device) < CR
            if not cross_points.any():
                cross_points[th.randint(0, dim, (1,))] = True
            trial[cross_points] = mutant[cross_points]
            
            # Selection
            trial_fitness = fitness_func(trial)
            fes += 1
            
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness.item()
                    best_solution = trial.clone()
        
        history.append(best_fitness)
    
    return best_solution.cpu(), best_fitness, history


def cuckoo_search(fitness_func, dim, bounds, max_fes=10000, pop_size=25, 
                 pa=0.25, beta=1.5):
    """
    Cuckoo Search (CS) with Levy flights
    
    Args:
        fitness_func: Function to minimize
        dim: Problem dimensionality
        bounds: List of [min, max] for each dimension
        max_fes: Maximum function evaluations
        pop_size: Number of nests
        pa: Probability of discovering alien eggs
        beta: Levy exponent
    
    Returns:
        best_solution, best_fitness, history
    """
    bounds = th.tensor(bounds, dtype=th.float32, device=device)
    lower, upper = bounds[:, 0], bounds[:, 1]
    
    # Initialize nests
    nests = th.rand(pop_size, dim, device=device) * (upper - lower) + lower
    fitness = th.zeros(pop_size, device=device)
    
    # Evaluate initial nests
    fes = 0
    for i in range(pop_size):
        fitness[i] = fitness_func(nests[i])
        fes += 1
    
    best_idx = th.argmin(fitness)
    best_nest = nests[best_idx].clone()
    best_fitness = fitness[best_idx].item()
    
    history = [best_fitness]
    
    # Main loop
    while fes < max_fes:
        # Generate new solutions via Levy flights
        for i in range(pop_size):
            if fes >= max_fes:
                break
            
            # Levy flight
            step = levy_flight(dim, beta, device)
            step_size = 0.01 * step * (nests[i] - best_nest)
            new_nest = nests[i] + step_size * th.randn(dim, device=device)
            
            # Boundary handling
            new_nest = th.clamp(new_nest, lower, upper)
            
            # Evaluate
            new_fitness = fitness_func(new_nest)
            fes += 1
            
            # Random walk/selective evaluation
            if new_fitness < fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness
                
                if new_fitness < best_fitness:
                    best_fitness = new_fitness.item()
                    best_nest = new_nest.clone()
        
        # Abandon some nests (fraction pa)
        num_abandon = int(pa * pop_size)
        worst_indices = th.argsort(fitness, descending=True)[:num_abandon]
        
        for idx in worst_indices:
            if fes >= max_fes:
                break
            
            # Generate new random solution
            nests[idx] = th.rand(dim, device=device) * (upper - lower) + lower
            fitness[idx] = fitness_func(nests[idx])
            fes += 1
            
            if fitness[idx] < best_fitness:
                best_fitness = fitness[idx].item()
                best_nest = nests[idx].clone()
        
        history.append(best_fitness)
    
    return best_nest.cpu(), best_fitness, history

# Part 1.1 - Improved Variants
def improved_particle_swarm_optimization(): # IPSO
    pass

def two_phase_particle_swarm_optimization(): # TPPSO
    pass

# Part 2 - Hybrid Algorithms

def hybrid_genetic_algorithm_pso(): # GA-PSO
    pass

def gray_wolf_cuckoo_search(): # GWO-CS
    pass

def adaptative_weighted_gray_wolf_optimizer(): # AW-GWO
    pass

def newton_raphson_genetic_algorithm(): # NR-GA
    pass

# Part 3 - Other Bioinspired Algorithms
def gray_wolf_optimizer(): # GWO    
    pass

def cat_swarm_optimization(): # CS
    pass

def bacterial_foraging_optimization(): # BFO
    pass

def artificial_bee_colony(): # ABC
    pass

# Part 4 - Needed Review
def group_meaning_based_optimization(): # GMB-OPT
    pass


# Helper Functions
# TODO : Move to utils.py 

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

