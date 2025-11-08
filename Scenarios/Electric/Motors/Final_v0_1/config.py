"""
Configuraciones generales del proyecto
"""

import math

# ============================================
# PARÁMETROS DEL MOTOR
# ============================================

# Parámetros verdaderos (valores de referencia para validación)
TRUE_PARAMS = {
    'rs': 0.435,
    'rr': 0.816,
    'Lls': 0.002,
    'Llr': 0.002,
    'Lm': 0.0693,
    'J': 0.089,
    'B': 0.0001
}

PARAM_NAMES = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']

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

# ============================================
# CONFIGURACIÓN DE OPTIMIZACIÓN
# ============================================

OPTIMIZATION_CONFIG = {
    'max_fes': 10000,       # Evaluaciones de función máximas
    'time_steps': 120,      # Pasos de tiempo para ODE (balance velocidad/precisión)
    'rtol': 1e-3,          # Tolerancia relativa ODE
    'atol': 1e-4,          # Tolerancia absoluta ODE
    'ode_method': 'rk4',   # Método de integración
}

# ============================================
# CONFIGURACIÓN DE ALGORITMOS
# ============================================

ALGORITHM_CONFIGS = {
    'GA': {
        'name': 'Genetic Algorithm',
        'short_name': 'GA',
        'pop_size': 80,
        'pc': 0.8,              # Probabilidad de cruce
        'pm': 0.1,              # Probabilidad de mutación
        'tournament_size': 3,
        'color': '#FF6B6B'      # Para gráficas
    },
    'PSO': {
        'name': 'Particle Swarm Optimization',
        'short_name': 'PSO',
        'pop_size': 60,
        'w': 0.7298,           # Peso de inercia
        'c1': 1.49618,         # Coeficiente cognitivo
        'c2': 1.49618,         # Coeficiente social
        'color': '#4ECDC4'
    },
    'DE': {
        'name': 'Differential Evolution',
        'short_name': 'DE',
        'pop_size': 80,
        'F': 0.5,              # Factor de mutación
        'CR': 0.9,             # Tasa de cruce
        'color': '#95E1D3'
    },
    'CS': {
        'name': 'Cuckoo Search',
        'short_name': 'CS',
        'pop_size': 50,
        'pa': 0.25,            # Probabilidad de abandono
        'beta': 1.5,           # Exponente de Levy
        'color': '#F38181'
    }
}

# ============================================
# CONFIGURACIÓN DE EXPERIMENTOS
# ============================================

EXPERIMENT_CONFIG = {
    'num_runs': 30,         # Número de ejecuciones por algoritmo
    'base_seed': 42,        # Semilla base para reproducibilidad
    'save_frequency': 5,    # Guardar cada N runs
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
