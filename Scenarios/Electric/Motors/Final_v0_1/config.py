"""
Configuraciones generales del proyecto
"""

import math

# PARÁMETROS DEL MOTOR


# Parámetros verdaderos (valores de referencia para validación)
TRUE_PARAMS = { # Parametros para motore entre 2 a 3 kW
    'rs':  0.70,       # ohm
    'rr':  0.50,       # ohm (referido al estator)
    'Lls': 0.0030,     # H
    'Llr': 0.0030,     # H
    'Lm':  0.0800,     # H
    'J':   0.012,      # kg·m^2
    'B':   1.4e-3      # N·m·s/rad
}

PARAM_NAMES = [
    'rs', 
    'rr', 
    'Lls', 
    'Llr', 
    'Lm', 
    'J', 
    'B'
    ]

NAMEPLATE = {
    'rated_power_kw': 2.2,       # kW de salida nominal
    'rated_speed_rpm': 1710.0,   # rpm nominal
    'rated_efficiency': 0.85,    # eficiencia nominal (0..1)
    'friction_frac_of_output': 0.02,  # 2% de P_out ~ P_fw típico 1–3%

    'rated_voltage_ll': 220.0,  # tensión línea-línea RMS
    'power_factor': 0.82,   # cos φ típico a plena carga
    'imag_frac': 0.30  # corriente de magnetización ~30% de la corriente nominal
}

# Pesos y factores del prior de B
PRIORS_CONFIG = {
    'use_b_prior': True,
    'b_prior_weight': 1.0,   # peso de regularización del prior en el fitness / 0.5
    'b_bounds_low': 0.5,     # límites = [low, high] * B_prior / 0.5
    'b_bounds_high': 2.0,  # / 2.0

    # prior sobre Lm ---
    'uselmprior': True,       # activar/desactivar prior de Lm
    'lmpriorweight': 0.3,
}

# Límites de búsqueda para cada parámetro
BOUNDS = [
    [0.05, 2.0],     # rs (Resistencia estator)
    [0.1, 2.0],      # rr (Resistencia rotor)
    [0.0001, 0.01],  # Lls (Inductancia dispersión estator)
    [0.0001, 0.01],  # Llr (Inductancia dispersión rotor)
    [0.01, 0.20],    # Lm (Inductancia magnetización)
    [0.01, 0.20],    # J (Momento de inercia)
    [0.00001, 0.001] # B (Coeficiente fricción)
]

# Voltajes de alimentación
V_line_rms = 220.0  # Voltaje línea-línea RMS
V_phase_peak = V_line_rms / math.sqrt(3) * math.sqrt(2)  # ≈179.6V
MOTOR_VOLTAGE = {
    'vqs': V_phase_peak,  # Alineado con eje q para torque máximo
    'vds': 0.0
}

# CONFIGURACIÓN DE EXPERIMENTOS

EXPERIMENT_CONFIG = {
    'num_runs': 1,         # Número de ejecuciones por algoritmo
    'base_seed': 40,        # Semilla base para reproducibilidad
    'save_frequency': 5,    # Guardar cada N runs
}

# CONFIGURACIÓN DE OPTIMIZACIÓN

OPTIMIZATION_CONFIG = {
    'max_fes': 3000,       # Evaluaciones de función máximas
    'time_total': 1,      # Tiempo total de simulación en segundos
    'time_steps': 1000,      # Pasos de tiempo para ODE (balance velocidad/precisión) 
    'rtol': 1e-3,          # Tolerancia relativa ODE
    'atol': 1e-4,          # Tolerancia absoluta ODE
    'ode_method': 'rk4',   # Método de integración

    'l2_weight': 3e-3  # Peso para la regularización L2 (Weight Decay)
}

# ============================================
# CONFIGURACIÓN DE ALGORITMOS
# ============================================

ALGORITHM_CONFIGS = {
    'GA': {
        'name': 'Genetic Algorithm',
        'short_name': 'GA',
        'max_fes': OPTIMIZATION_CONFIG['max_fes'],
        'pop_size': 80,
        'pc': 0.8,
        'pm': 0.20,
        'tournament_size': 2,
        'eta_c': 10,
        'eta_m': 15,
        'immigrants_frac': 0.15,
        'stagnation_gens': 6,
        'color': '#1f77b4'
    },
    'PSO': {
        'name': 'Particle Swarm Optimization',
        'short_name': 'PSO',
        'max_fes': OPTIMIZATION_CONFIG['max_fes'],
        'pop_size': 60,
        'w': 0.7298,           # Peso de inercia
        'c1': 1.49618,         # Coeficiente cognitivo
        'c2': 1.49618,         # Coeficiente social
        'color': '#ff7f0e'
    },
    "CMAES": {
        "name": "CMA-ES",
        "short_name": "CMAES",
        "max_fes": OPTIMIZATION_CONFIG["max_fes"],
        # si no especificas popsize/mu,sigma0 se calculan en el código
        "pop_size": 16,
        "mu": 8,
        "sigma0": 0.3,
        "color": "#2ca02c",
    },

    # ---------- 4º algoritmo: PSO + L-BFGS ----------
    "HYBRID_PSO_LBFGS": {
        "name": "PSO + L-BFGS",
        "short_name": "HYB",
        "max_fes": OPTIMIZATION_CONFIG["max_fes"],
        "pop_size": 60,
        "global_frac": 0.8,      # 80% FEs PSO, 20% L-BFGS
        "max_local_iters": 30,
        "lbfgs_lr": 1.0,
        "color": "#d62728",
    },

}


# ============================================
# FASES DEL PROYECTO
# ============================================

PHASES = {
    'phase1': {
        'name': 'Validación con datos sintéticos',
        'description': 'Optimización usando datos generados del modelo conocido',
        'output_dir': 'results/phase1_synthetic',
        'scenarios': ['synthetic']
    },
    'phase2': {
        'name': 'Datos experimentales - Escenario 1',
        'description': 'Optimización con datos reales - condición nominal',
        'output_dir': 'results/phase2_experimental_nominal',
        'scenarios': ['experimental_nominal']
    },
    'phase3': {
        'name': 'Datos experimentales - Escenario 2',
        'description': 'Optimización con datos reales - condición perturbada',
        'output_dir': 'results/phase2_experimental_perturbed',
        'scenarios': ['experimental_perturbed']
    }
}

# ============================================
# CONFIGURACIÓN DE ARCHIVOS
# ============================================

DATA_FILES = {
    'current': 'data/current_measured.txt',
    'rpm': 'data/rpm_measured.txt',
    'torque': 'data/torque_measured.txt'
}

# ============================================
# CONFIGURACIÓN DE PyTorch
# ============================================

PYTORCH_CONFIG = {
    'cudnn_benchmark': True,
    'cudnn_deterministic': False,
    'matmul_precision': 'high'  # Usar Tensor Cores
}

# ============================================
# MÉTRICAS A GUARDAR POR RUN
# ============================================

METRICS_TO_SAVE = [
    'run_id',
    'seed',
    'best_fitness',
    'best_params',
    'param_errors',          # Error de cada parámetro vs true_params
    'param_errors_percent',  # Error porcentual
    'convergence_history',   # Fitness en cada generación
    'execution_time',        # Tiempo de ejecución en segundos
    'final_generation',      # Generación donde terminó
    'stagnation_count'       # Cuántas generaciones sin mejora
]

# ============================================
# ESTADÍSTICAS A CALCULAR
# ============================================

STATISTICS_TO_COMPUTE = [
    'mean',
    'std',
    'median',
    'min',
    'max',
    'q25',      # Cuartil 25%
    'q75',      # Cuartil 75%
    'iqr',      # Rango intercuartílico
    'cv'        # Coeficiente de variación
]
