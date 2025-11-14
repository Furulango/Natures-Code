"""
Gestor de experimentos para múltiples runs
"""

import time
import os
from datetime import datetime
from config import (
    EXPERIMENT_CONFIG, 
    ALGORITHM_CONFIGS,
    METRICS_TO_SAVE,
    OPTIMIZATION_CONFIG,
    PARAM_NAMES
)
from utils import (
    set_seed,
    calculate_parameter_errors,
    detect_stagnation,
    save_run_data,
    print_run_summary,
    print_algorithm_statistics,
    compute_statistics,
    create_output_directory,
    format_time
)


class ExperimentManager:
    """
    Gestiona la ejecución de múltiples runs de algoritmos de optimización
    """
    
    def __init__(self, output_dir, phase_name='phase1'):
        """
        Inicializa el gestor de experimentos
        
        Args:
            output_dir: Directorio donde guardar resultados
            phase_name: Nombre de la fase del experimento
        """
        self.output_dir = output_dir
        self.phase_name = phase_name
        self.num_runs = EXPERIMENT_CONFIG['num_runs']
        self.base_seed = EXPERIMENT_CONFIG['base_seed']
        self.save_frequency = EXPERIMENT_CONFIG['save_frequency']
        
        # Crear directorio de salida
        create_output_directory(output_dir)
        
        # Resultados por algoritmo
        self.results = {}
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENTO: {phase_name}")
        print(f"{'='*70}")
        print(f"Runs por algoritmo: {self.num_runs}")
        print(f"Directorio de salida: {output_dir}")
        print(f"{'='*70}\n")
    
    
    def run_algorithm_experiment(self, algorithm_name, algorithm_func, 
                                fitness_func, bounds, dim=7):
        """
        Ejecuta múltiples runs de un algoritmo
        
        Args:
            algorithm_name: Nombre del algoritmo ('GA', 'PSO', 'DE', 'CS')
            algorithm_func: Función del algoritmo a ejecutar
            fitness_func: Función de fitness (ya configurada con datos)
            bounds: Límites de búsqueda
            dim: Dimensionalidad del problema
        
        Returns:
            dict con resultados de todos los runs
        """
        config = ALGORITHM_CONFIGS[algorithm_name]
        
        print(f"\n{'='*70}")
        print(f"{config['name']} ({algorithm_name})")
        print(f"{'='*70}")
        
        # Archivo de salida para este algoritmo
        output_file = os.path.join(self.output_dir, f"{algorithm_name}_results.json")
        
        # Inicializar estructura de resultados
        if os.path.exists(output_file):
            print(f"-> El archivo {output_file} ya existe. Continuando desde último run...")
            # TODO: Implementar recuperación de runs previos
        
        experiment_start = time.time()
        
        for run in range(1, self.num_runs + 1):
            # Semilla única pero reproducible
            seed = self.base_seed + run * 1000
            set_seed(seed)
            
            run_start = time.time()
            
            # Ejecutar optimización
            best_params, best_fitness, history = algorithm_func(
                fitness_func,
                dim=dim,
                bounds=bounds,
                max_fes=OPTIMIZATION_CONFIG['max_fes'],
                pop_size=OPTIMIZATION_CONFIG.get('pop_size', config['pop_size']),
                **{k: v for k, v in config.items() 
                   if k not in ['name', 'short_name', 'pop_size', 'color', 'max_fes']}
            )
            
            run_time = time.time() - run_start
            
            # Calcular errores de parámetros
            param_errors, param_errors_percent = calculate_parameter_errors(best_params)
            
            # Detectar estancamiento
            stagnation = detect_stagnation(history)
            
            # Preparar datos del run
            run_data = {
                'run_id': run,
                'seed': seed,
                'algorithm': algorithm_name,
                'best_fitness': float(best_fitness),
                'best_params': {
                    name: float(best_params[i]) 
                    for i, name in enumerate(PARAM_NAMES)
                },
                'param_errors': param_errors,
                'param_errors_percent': param_errors_percent,
                'convergence_history': [float(h) for h in history],
                'execution_time': run_time,
                'final_generation': len(history),
                'stagnation_count': stagnation,
                'timestamp': datetime.now().isoformat()
            }
            
            # Guardar run
            save_run_data(output_file, run_data, append=(run > 1))
            
            # Imprimir progreso
            elapsed = time.time() - experiment_start
            print_run_summary(
                run, self.num_runs, best_fitness, run_time,
                param_errors_percent, elapsed, algorithm_name
            )
        
        total_time = time.time() - experiment_start
        
        print(f"\n-> {algorithm_name} completado en {format_time(total_time)}")
        print(f"  Archivo: {output_file}")
        
        # Calcular y mostrar estadísticas
        self._compute_and_display_statistics(algorithm_name, output_file)
        
        return output_file
    
    
    def _compute_and_display_statistics(self, algorithm_name, results_file):
        """
        Calcula y muestra estadísticas del algoritmo
        """
        import json
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        runs = data['runs']
        
        # Estadísticas de fitness
        fitness_values = [run['best_fitness'] for run in runs]
        fitness_stats = compute_statistics(fitness_values)
        
        # Estadísticas por parámetro
        param_stats = {}
        for param_name in PARAM_NAMES:
            param_errors = [run['param_errors_percent'][param_name] for run in runs]
            param_stats[param_name] = compute_statistics(param_errors)
        
        # Mostrar
        print_algorithm_statistics(algorithm_name, fitness_stats, param_stats)
        
        # Guardar estadísticas en el archivo
        data['statistics'] = {
            'fitness': fitness_stats,
            'parameters': param_stats,
            'execution_time': {
                'mean': float(sum(run['execution_time'] for run in runs) / len(runs)),
                'total': float(sum(run['execution_time'] for run in runs))
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    
    def run_all_algorithms(self, algorithms_dict, fitness_func, bounds, dim=7):
        """
        Ejecuta todos los algoritmos especificados
        
        Args:
            algorithms_dict: Dict con {nombre: función} de algoritmos
            fitness_func: Función de fitness
            bounds: Límites de búsqueda
            dim: Dimensionalidad
        
        Returns:
            dict con rutas de archivos de resultados
        """
        total_start = time.time()
        result_files = {}
        
        for algo_name, algo_func in algorithms_dict.items():
            result_file = self.run_algorithm_experiment(
                algo_name, algo_func, fitness_func, bounds, dim
            )
            result_files[algo_name] = result_file
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*70}")
        print(f"TODOS LOS EXPERIMENTOS COMPLETADOS")
        print(f"{'='*70}")
        print(f"Tiempo total: {format_time(total_time)}")
        print(f"Archivos generados:")
        for algo, filepath in result_files.items():
            print(f"  • {algo}: {filepath}")
        print(f"{'='*70}\n")
        
        # Generar resumen comparativo
        self._generate_comparative_summary(result_files)
        
        return result_files
    
    
    def _generate_comparative_summary(self, result_files):
        """
        Genera resumen comparativo de todos los algoritmos
        """
        import json
        
        summary = {
            'phase': self.phase_name,
            'generated_at': datetime.now().isoformat(),
            'num_runs': self.num_runs,
            'algorithms': {}
        }
        
        print(f"\n{'='*70}")
        print(f"RESUMEN COMPARATIVO")
        print(f"{'='*70}\n")
        
        # Tabla de resultados
        print(f"{'Algoritmo':<15} {'Mean Fitness':<15} {'Std':<12} {'Best':<12} {'Worst':<12}")
        print(f"{'-'*66}")
        
        for algo_name, filepath in result_files.items():
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            stats = data['statistics']['fitness']
            summary['algorithms'][algo_name] = {
                'fitness_statistics': stats,
                'parameter_statistics': data['statistics']['parameters'],
                'execution_time': data['statistics']['execution_time']
            }
            
            print(f"{algo_name:<15} "
                  f"{stats['mean']:<15.6f} "
                  f"{stats['std']:<12.6f} "
                  f"{stats['min']:<12.6f} "
                  f"{stats['max']:<12.6f}")
        
        print(f"{'-'*66}\n")
        
        # Guardar resumen
        summary_file = os.path.join(self.output_dir, 'summary_statistics.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Resumen guardado en: {summary_file}\n")
        
        # Determinar mejor algoritmo
        best_algo = min(summary['algorithms'].items(), 
                       key=lambda x: x[1]['fitness_statistics']['mean'])
        
        print(f"   Mejor algoritmo (promedio): {best_algo[0]}")
        print(f"   Mean Fitness: {best_algo[1]['fitness_statistics']['mean']:.6f}")
        print(f"   Std: {best_algo[1]['fitness_statistics']['std']:.6f}\n")
