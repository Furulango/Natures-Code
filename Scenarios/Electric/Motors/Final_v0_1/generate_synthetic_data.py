"""
Generador de datos sintéticos para Fase 1
Usa parámetros verdaderos para simular señales nominales
"""

import torch as th
try:
    from torchdiffeq import odeint
except ImportError:
    raise ImportError("Instala torchdiffeq: pip install torchdiffeq")
from config import TRUE_PARAMS, MOTOR_VOLTAGE, OPTIMIZATION_CONFIG, PYTORCH_CONFIG, PARAM_NAMES
from motor_dynamic_batch import InductionMotorModelBatch
from utils import setup_pytorch
import numpy as np
import os

def generate_synthetic_data():
    # Configurar PyTorch
    device = setup_pytorch(PYTORCH_CONFIG)
    
    # Parámetros verdaderos como tensor (batch_size=1)
    true_params_array = th.tensor([TRUE_PARAMS[name] for name in PARAM_NAMES], 
                                  dtype=th.float32, device=device).unsqueeze(0)
    
    # Crear y configurar modelo
    model = InductionMotorModelBatch(vqs=MOTOR_VOLTAGE['vqs'], vds=MOTOR_VOLTAGE['vds']).to(device)
    model.update_params_batch(true_params_array)
    
    # Condiciones iniciales
    x0 = th.zeros(1, 5, dtype=th.float32, device=device)
    
    # Tiempo de simulación (mismo que en fitness)
    t = th.linspace(0, 0.1, OPTIMIZATION_CONFIG['time_steps'], device=device)
    
    # Resolver EDO
    sol = odeint(model, x0, t, method=OPTIMIZATION_CONFIG['ode_method'],
                 rtol=OPTIMIZATION_CONFIG['rtol'], atol=OPTIMIZATION_CONFIG['atol'])
    sol = sol.permute(1, 0, 2)  # (1, time_steps, 5)
    
    # Calcular señales
    current_sim = model.calculate_stator_current(sol).squeeze(0).cpu().numpy()
    rpm_sim = model.calculate_rpm(sol).squeeze(0).cpu().numpy()
    torque_sim = model.calculate_torque(sol).squeeze(0).cpu().numpy()
    
    # Crear directorio data
    os.makedirs('data', exist_ok=True)
    
    # Guardar como TXT (una columna por archivo)
    np.savetxt('data/current_measured.txt', current_sim)
    np.savetxt('data/rpm_measured.txt', rpm_sim)
    np.savetxt('data/torque_measured.txt', torque_sim)
    
    print("\n✅ Datos sintéticos generados y guardados:")
    print(f" • Puntos: {len(current_sim)}")
    print(f" • Corriente: [{current_sim.min():.2f}, {current_sim.max():.2f}] A")
    print(f" • RPM: [{rpm_sim.min():.2f}, {rpm_sim.max():.2f}]")
    print(f" • Torque: [{torque_sim.min():.2f}, {torque_sim.max():.2f}] N·m")
    print("\nAhora ejecuta: python main_phase1.py")

if __name__ == "__main__":
    generate_synthetic_data()
