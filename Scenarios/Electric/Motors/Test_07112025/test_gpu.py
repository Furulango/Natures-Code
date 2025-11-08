import torch as th
from torchdiffeq import odeint
import motor_dynamic
import time

device = 'cuda'
# Mostrar grafica que se esta usando
print(f"Using device: {device}")


params = [0.5, 0.8, 0.002, 0.002, 0.08, 0.1, 0.0001]
model = motor_dynamic.InductionMotorModel(params, vqs=220.0, vds=0.0).to(device)
x0 = th.zeros(5, dtype=th.float64, device=device)
t = th.linspace(0, 1, 1000, device=device)

#Medir tiempo de ejecucion

start_time = time.time()
sol = odeint(model, x0, t)
end_time = time.time()

print(f"Tiempo de ejecución en GPU: {end_time - start_time:.4f} segundos")