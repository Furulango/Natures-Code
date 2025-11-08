# Sistema de Identificación de Parámetros de Motor de Inducción

## Estructura del Proyecto

```
proyecto/
├── config.py                          # Configuraciones generales
├── utils.py                           # Funciones auxiliares
├── experiment_manager.py              # Gestor de experimentos
├── motor_dynamic_batch.py             # Modelo del motor (batch)
├── BIA_algorithms_parallel.py         # Algoritmos bioinspirados
├── main_phase1.py                     # EJECUTAR - Fase 1
├── analyze_results.py                 # Análisis de resultados
├── data/
│   ├── current_measured.txt           # Corriente medida
│   ├── rpm_measured.txt               # RPM medido
│   └── torque_measured.txt            # Torque medido
└── results/
  └── phase1_synthetic/              # Resultados Fase 1
    ├── GA_results.json            # Resultados GA (30 runs)
    ├── PSO_results.json           # Resultados PSO (30 runs)
    ├── DE_results.json            # Resultados DE (30 runs)
    ├── CS_results.json            # Resultados CS (30 runs)
    ├── summary_statistics.json    # Resumen comparativo
    └── convergence_data.json      # Datos de convergencia (opcional)
```

---

## Guía de Uso Rápida

### **Paso 1: Ejecutar Fase 1 (Datos Sintéticos)**

```bash
python main_phase1.py
```

**Esto ejecutará:**
- 30 runs de GA
- 30 runs de PSO
- 30 runs de DE
- 30 runs de CS

**Tiempo estimado:** ~6-7 horas (dependiendo de tu GPU)

**Salida:** 4 archivos JSON + 1 resumen en `results/phase1_synthetic/`

---

### **Paso 2: Analizar Resultados**

#### Comparación de todos los algoritmos:
```bash
python analyze_results.py --phase phase1
```

#### Análisis detallado de un algoritmo:
```bash
python analyze_results.py --phase phase1 --algorithm GA
```

#### Exportar datos de convergencia:
```bash
python analyze_results.py --phase phase1 --export-convergence
```

---

## Estructura de Archivos de Resultados

### **Archivo por algoritmo** (`GA_results.json`, etc.):

```json
{
  "metadata": {
  "algorithm": "GA",
  "created": "2025-01-15T10:30:00",
  "total_runs": 30
  },
  "runs": [
  {
    "run_id": 1,
    "seed": 1042,
    "best_fitness": 0.012345,
    "best_params": {
    "rs": 0.435123,
    "rr": 0.816234,
    "Lls": 0.002001,
    "Llr": 0.001998,
    "Lm": 0.069312,
    "J": 0.089045,
    "B": 0.000100
    },
    "param_errors": {
    "rs": 0.000123,
    "rr": 0.000234,
    ...
    },
    "param_errors_percent": {
    "rs": 0.28,
    "rr": 0.29,
    ...
    },
    "convergence_history": [0.5, 0.3, 0.15, ...],
    "execution_time": 210.5,
    "final_generation": 100,
    "stagnation_count": 15,
    "timestamp": "2025-01-15T10:33:30"
  },
  {
    "run_id": 2,
    ...
  }
  // ... 30 runs en total
  ],
  "statistics": {
  "fitness": {
    "mean": 0.0125,
    "std": 0.0023,
    "median": 0.0122,
    "min": 0.0098,
    "max": 0.0156,
    "q25": 0.0110,
    "q75": 0.0135,
    "iqr": 0.0025,
    "cv": 0.184
  },
  "parameters": {
    "rs": {
    "mean": 0.35,
    "std": 0.12,
    ...
    },
    // ... estadísticas por parámetro
  },
  "execution_time": {
    "mean": 210.3,
    "total": 6309.0
  }
  }
}
```

### **Archivo resumen** (`summary_statistics.json`):

```json
{
  "phase": "Validación con datos sintéticos",
  "generated_at": "2025-01-15T17:45:00",
  "num_runs": 30,
  "algorithms": {
  "GA": {
    "fitness_statistics": { ... },
    "parameter_statistics": { ... }
  },
  "PSO": { ... },
  "DE": { ... },
  "CS": { ... }
  }
}
```

---

## Configuración

### Modificar parámetros de optimización:

Edita `config.py`:

```python
OPTIMIZATION_CONFIG = {
  'max_fes': 10000,      # Cambiar evaluaciones
  'time_steps': 120,     # Cambiar resolución temporal
}

EXPERIMENT_CONFIG = {
  'num_runs': 30,        # Cambiar número de runs
}

ALGORITHM_CONFIGS = {
  'GA': {
    'pop_size': 80,    # Cambiar tamaño población
    ...
  }
}
```

---

## Datos que se Guardan por Run

Para cada run de cada algoritmo se guarda:

| Dato | Descripción | Uso para Paper |
|------|-------------|----------------|
| `best_fitness` | Mejor fitness alcanzado | Comparar algoritmos |
| `best_params` | Parámetros estimados | Verificar precisión |
| `param_errors` | Error absoluto por parámetro | Análisis detallado |
| `param_errors_percent` | Error porcentual | Tablas comparativas |
| `convergence_history` | Fitness por generación | Gráficas de convergencia |
| `execution_time` | Tiempo de ejecución | Eficiencia computacional |
| `stagnation_count` | Generaciones sin mejora | Análisis de convergencia |

---

## Para tu Paper

### Tablas que puedes generar:

1. **Tabla I: Estadísticas de Fitness**
   ```
   Algorithm | Mean ± Std | Median | Best | Worst
   ```

2. **Tabla II: Error de Parámetros (%)**
   ```
   Parámetro | GA | PSO | DE | CS
   ```

3. **Tabla III: Tiempo de Ejecución**
   ```
   Algorithm | Mean Time | Total Time
   ```

### Gráficas recomendadas:

1. Convergencia promedio (30 runs con banda de std)
2. Boxplots de fitness por algoritmo
3. Distribución de errores por parámetro
4. Comparación best-worst por algoritmo

---

## Siguientes Fases

Una vez completada la Fase 1:

### **Fase 2: Datos experimentales - Escenario nominal**

1. Copia `main_phase1.py` → `main_phase2.py`
2. Cambia los archivos de datos en `config.py`:
   ```python
   DATA_FILES = {
     'current': 'data/experimental_current_nominal.txt',
     'rpm': 'data/experimental_rpm_nominal.txt',
     'torque': 'data/experimental_torque_nominal.txt'
   }
   ```
3. Cambia `output_dir` a `phase2`
4. Ejecuta: `python main_phase2.py`

### **Fase 3: Datos experimentales - Escenario perturbado**

Similar a Fase 2, con datos de condiciones perturbadas.

---

## Troubleshooting

### Error: "Out of memory"
- Reduce `pop_size` en `config.py`
- Reduce `time_steps` de 120 a 100

### Error: "Archivo no encontrado"
- Verifica que los archivos `.txt` estén en `data/`
- Verifica nombres exactos en `config.py`

### Proceso muy lento
- Verifica que estés usando GPU: debe decir "Using device: cuda"
- Si usa CPU, instala drivers CUDA

### Resultados no reproducibles
- Verifica que `base_seed` sea fijo en `config.py`
- No ejecutes múltiples instancias simultáneas

---

## Resumen de Comandos

```bash
# Ejecutar Fase 1 completa (6-7 horas)
python main_phase1.py

# Ver resumen de todos los algoritmos
python analyze_results.py --phase phase1

# Ver detalles de un algoritmo
python analyze_results.py --phase phase1 --algorithm GA

# Exportar datos de convergencia para gráficas
python analyze_results.py --phase phase1 --export-convergence
```

---

## Qué Archivos Guardar para Paper

**Esenciales:**
- `results/phase1_synthetic/*.json` (todos)
- `config.py` (configuración exacta usada)
- Logs de ejecución (captura de pantalla)

**Para análisis estadístico:**
- `summary_statistics.json`
- `convergence_data.json`

**Para reproducibilidad:**
- Todos los archivos `.py`
- `requirements.txt` con versiones de librerías
- Descripción de hardware (GPU, CUDA version)

---

## Checklist Paper

- [ ] Ejecutar 30 runs de cada algoritmo
- [ ] Verificar reproducibilidad (re-ejecutar con misma seed)
- [ ] Calcular estadísticas (mean, std, median)
- [ ] Generar gráficas de convergencia
- [ ] Crear boxplots
- [ ] Tests estadísticos (Wilcoxon, Friedman)
- [ ] Documentar configuración exacta
- [ ] Guardar tiempo de ejecución
- [ ] Comparar con valores verdaderos
- [ ] Analizar casos best/worst

---

## Citas Recomendadas

Para justificar 30 runs:

> "Statistical significance requires at least 30 independent runs to 
> ensure reliable comparisons [Derrac et al., 2011]"

Para métodos de comparación:

> "Non-parametric tests (Wilcoxon, Friedman) are recommended for 
> comparing metaheuristic algorithms [García et al., 2010]"
