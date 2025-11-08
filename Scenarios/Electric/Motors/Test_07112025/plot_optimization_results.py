import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torchdiffeq import odeint
import motor_dynamic

def plot_convergence(hist_ga, hist_pso, hist_de, hist_cs, save_path='convergence.png'):
    """
    Grafica las curvas de convergencia de los 4 algoritmos
    
    Args:
        hist_ga, hist_pso, hist_de, hist_cs: Listas con historial de fitness
        save_path: Ruta para guardar la figura
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(hist_ga, label='GA', linewidth=2, alpha=0.8)
    plt.plot(hist_pso, label='PSO', linewidth=2, alpha=0.8)
    plt.plot(hist_de, label='DE', linewidth=2, alpha=0.8)
    plt.plot(hist_cs, label='CS', linewidth=2, alpha=0.8)
    
    plt.xlabel('Generación / Iteración', fontsize=12)
    plt.ylabel('Error (Fitness)', fontsize=12)
    plt.title('Convergencia de Algoritmos Bioinspirados', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Escala logarítmica para ver mejor la convergencia
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica de convergencia guardada: {save_path}")
    plt.show()


def plot_comparison_table(best_params_dict, param_names, true_params=None, save_path='comparison_table.png'):
    """
    Tabla comparativa de parámetros encontrados por cada algoritmo
    
    Args:
        best_params_dict: Dict con {algoritmo: tensor_params}
        param_names: Lista con nombres de parámetros
        true_params: Lista con parámetros verdaderos (opcional)
        save_path: Ruta para guardar la figura
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Preparar datos para la tabla
    algorithms = list(best_params_dict.keys())
    n_params = len(param_names)
    
    # Headers
    headers = ['Parámetro'] + algorithms
    if true_params is not None:
        headers.append('Verdadero')
    
    # Construir filas
    table_data = []
    for i, name in enumerate(param_names):
        row = [name]
        for algo in algorithms:
            row.append(f"{best_params_dict[algo][i]:.6f}")
        if true_params is not None:
            row.append(f"{true_params[i]:.6f}")
        table_data.append(row)
    
    # Crear tabla
    table = ax.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colWidths=[0.12] * len(headers))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Estilizar header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternar colores de filas
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Comparación de Parámetros Optimizados', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Tabla comparativa guardada: {save_path}")
    plt.show()


def plot_motor_responses(best_params, current_measured, rpm_measured, torque_measured, 
                        algorithm_name='Best', save_path='motor_response.png'):
    """
    Grafica las respuestas simuladas vs medidas del motor
    
    Args:
        best_params: Tensor con mejores parámetros encontrados
        current_measured, rpm_measured, torque_measured: Datos medidos (tensores)
        algorithm_name: Nombre del algoritmo
        save_path: Ruta para guardar la figura
    """
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    
    # Simular con mejores parámetros
    params_list = best_params.tolist()
    model = motor_dynamic.InductionMotorModel(params_list, vqs=220.0, vds=0.0).to(device)
    
    x0 = th.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=th.float64, device=device)
    t = th.linspace(0, 1, 1000, device=device)
    
    sol = odeint(model, x0, t)
    
    current_sim = model.calculate_stator_current(sol).cpu().numpy()
    rpm_sim = model.calculate_rpm(sol).cpu().numpy()
    torque_sim = model.calculate_torque(sol).cpu().numpy()
    
    # Convertir datos medidos a numpy
    current_meas_np = current_measured.cpu().numpy()
    rpm_meas_np = rpm_measured.cpu().numpy()
    torque_meas_np = torque_measured.cpu().numpy()
    t_np = t.cpu().numpy()
    
    # Crear figura con 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    
    # Subplot 1: Corriente
    axs[0].plot(t_np, current_meas_np, 'b-', label='Medida', linewidth=1.5, alpha=0.7)
    axs[0].plot(t_np, current_sim, 'r--', label=f'Simulada ({algorithm_name})', linewidth=2)
    axs[0].set_ylabel('Corriente [A]', fontsize=11)
    axs[0].legend(fontsize=10)
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title(f'Comparación: Respuesta Simulada vs Medida ({algorithm_name})', 
                     fontsize=13, fontweight='bold')
    
    # Subplot 2: RPM
    axs[1].plot(t_np, rpm_meas_np, 'b-', label='Medida', linewidth=1.5, alpha=0.7)
    axs[1].plot(t_np, rpm_sim, 'r--', label=f'Simulada ({algorithm_name})', linewidth=2)
    axs[1].set_ylabel('RPM', fontsize=11)
    axs[1].legend(fontsize=10)
    axs[1].grid(True, alpha=0.3)
    
    # Subplot 3: Torque
    axs[2].plot(t_np, torque_meas_np, 'b-', label='Medida', linewidth=1.5, alpha=0.7)
    axs[2].plot(t_np, torque_sim, 'r--', label=f'Simulada ({algorithm_name})', linewidth=2)
    axs[2].set_xlabel('Tiempo [s]', fontsize=11)
    axs[2].set_ylabel('Torque [N·m]', fontsize=11)
    axs[2].legend(fontsize=10)
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Respuesta del motor guardada: {save_path}")
    plt.show()
    
    # Calcular errores
    mse_current = np.mean((current_sim - current_meas_np)**2)
    mse_rpm = np.mean((rpm_sim - rpm_meas_np)**2)
    mse_torque = np.mean((torque_sim - torque_meas_np)**2)
    
    print(f"\nErrores MSE para {algorithm_name}:")
    print(f"  Corriente: {mse_current:.6f}")
    print(f"  RPM:       {mse_rpm:.6f}")
    print(f"  Torque:    {mse_torque:.6f}")


def plot_all_responses_comparison(best_params_dict, current_measured, rpm_measured, 
                                 torque_measured, save_path='all_responses.png'):
    """
    Compara las respuestas de todos los algoritmos en una sola figura
    
    Args:
        best_params_dict: Dict con {algoritmo: tensor_params}
        current_measured, rpm_measured, torque_measured: Datos medidos
        save_path: Ruta para guardar
    """
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    
    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    
    t = th.linspace(0, 1, 1000, device=device)
    t_np = t.cpu().numpy()
    
    # Datos medidos
    current_meas_np = current_measured.cpu().numpy()
    rpm_meas_np = rpm_measured.cpu().numpy()
    torque_meas_np = torque_measured.cpu().numpy()
    
    # Plotear datos medidos
    axs[0].plot(t_np, current_meas_np, 'k-', label='Medida', linewidth=2, alpha=0.8)
    axs[1].plot(t_np, rpm_meas_np, 'k-', label='Medida', linewidth=2, alpha=0.8)
    axs[2].plot(t_np, torque_meas_np, 'k-', label='Medida', linewidth=2, alpha=0.8)
    
    colors = {'GA': 'red', 'PSO': 'blue', 'DE': 'green', 'CS': 'orange'}
    
    # Simular para cada algoritmo
    for algo_name, params in best_params_dict.items():
        params_list = params.tolist()
        model = motor_dynamic.InductionMotorModel(params_list, vqs=220.0, vds=0.0).to(device)
        
        x0 = th.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=th.float64, device=device)
        sol = odeint(model, x0, t)
        
        current_sim = model.calculate_stator_current(sol).cpu().numpy()
        rpm_sim = model.calculate_rpm(sol).cpu().numpy()
        torque_sim = model.calculate_torque(sol).cpu().numpy()
        
        color = colors.get(algo_name, 'gray')
        
        axs[0].plot(t_np, current_sim, '--', label=algo_name, linewidth=1.5, 
                   color=color, alpha=0.7)
        axs[1].plot(t_np, rpm_sim, '--', label=algo_name, linewidth=1.5, 
                   color=color, alpha=0.7)
        axs[2].plot(t_np, torque_sim, '--', label=algo_name, linewidth=1.5, 
                   color=color, alpha=0.7)
    
    # Configurar subplots
    axs[0].set_ylabel('Corriente [A]', fontsize=11)
    axs[0].legend(fontsize=9, loc='best')
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title('Comparación de Todos los Algoritmos vs Mediciones', 
                    fontsize=13, fontweight='bold')
    
    axs[1].set_ylabel('RPM', fontsize=11)
    axs[1].legend(fontsize=9, loc='best')
    axs[1].grid(True, alpha=0.3)
    
    axs[2].set_xlabel('Tiempo [s]', fontsize=11)
    axs[2].set_ylabel('Torque [N·m]', fontsize=11)
    axs[2].legend(fontsize=9, loc='best')
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparación de todas las respuestas guardada: {save_path}")
    plt.show()


def plot_error_bars(fitness_dict, save_path='error_comparison.png'):
    """
    Gráfica de barras comparando el error final de cada algoritmo
    
    Args:
        fitness_dict: Dict con {algoritmo: fitness_value}
        save_path: Ruta para guardar
    """
    algorithms = list(fitness_dict.keys())
    errors = list(fitness_dict.values())
    
    plt.figure(figsize=(10, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = plt.bar(algorithms, errors, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Añadir valores sobre las barras
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.6f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.ylabel('Error Final (Fitness)', fontsize=12)
    plt.xlabel('Algoritmo', fontsize=12)
    plt.title('Comparación de Error Final entre Algoritmos', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica de barras de error guardada: {save_path}")
    plt.show()


# ============================================================================
# Función principal para generar todos los gráficos
# ============================================================================

def generate_all_plots(hist_ga, hist_pso, hist_de, hist_cs,
                      best_ga, best_pso, best_de, best_cs,
                      fit_ga, fit_pso, fit_de, fit_cs,
                      current_measured, rpm_measured, torque_measured,
                      true_params=None):
    """
    Genera todos los gráficos de análisis
    
    Args:
        hist_*: Historiales de convergencia
        best_*: Mejores parámetros encontrados
        fit_*: Fitness finales
        current_measured, rpm_measured, torque_measured: Datos medidos
        true_params: Parámetros verdaderos (opcional)
    """
    print("\n" + "="*60)
    print("GENERANDO GRÁFICOS DE ANÁLISIS")
    print("="*60 + "\n")
    
    # 1. Convergencia
    plot_convergence(hist_ga, hist_pso, hist_de, hist_cs)
    
    # 2. Tabla comparativa
    best_params_dict = {
        'GA': best_ga,
        'PSO': best_pso,
        'DE': best_de,
        'CS': best_cs
    }
    param_names = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
    plot_comparison_table(best_params_dict, param_names, true_params)
    
    # 3. Barras de error
    fitness_dict = {
        'GA': fit_ga,
        'PSO': fit_pso,
        'DE': fit_de,
        'CS': fit_cs
    }
    plot_error_bars(fitness_dict)
    
    # 4. Respuesta del mejor algoritmo
    best_algo = min(fitness_dict, key=fitness_dict.get)
    best_params = best_params_dict[best_algo]
    plot_motor_responses(best_params, current_measured, rpm_measured, 
                        torque_measured, algorithm_name=best_algo,
                        save_path=f'motor_response_{best_algo}.png')
    
    # 5. Comparación de todas las respuestas
    plot_all_responses_comparison(best_params_dict, current_measured, 
                                 rpm_measured, torque_measured)
    
    print("\n✓ Todos los gráficos generados exitosamente!")
    print(f"\nMejor algoritmo: {best_algo} con error = {fitness_dict[best_algo]:.8f}")
