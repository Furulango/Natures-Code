import torch as th
from torchdiffeq import odeint
import numpy as np
import motor_dynamic

# Motor parameters [rs, rr, Lls, Llr, Lm, J, B] - Parámetros "verdaderos"
true_params = [0.435, 0.816, 0.002, 0.002, 0.0693, 0.089, 0.0001]
vqs, vds = 220.0, 0.0

print("Generando datos sintéticos del motor...")

# Crear modelo con parámetros verdaderos
model = motor_dynamic.InductionMotorModel(true_params, vqs, vds)

# Initial state: [iqs, ids, iqr, idr, wr]
x0 = th.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=th.float64)

# Time points (1 segundo, 1000 puntos)
t = th.linspace(0, 1, 1000)

print("Resolviendo ODE...")
sol = odeint(model, x0, t)
print("ODE resuelta!")

# Calcular las variables de salida
current = model.calculate_stator_current(sol).cpu().numpy()
rpm = model.calculate_rpm(sol).cpu().numpy()
torque = model.calculate_torque(sol).cpu().numpy()

# Añadir ruido realista a las mediciones
np.random.seed(42)

# Ruido gaussiano (SNR ~30dB)
noise_current = np.random.normal(0, 0.05 * np.std(current), size=current.shape)
noise_rpm = np.random.normal(0, 0.03 * np.std(rpm), size=rpm.shape)
noise_torque = np.random.normal(0, 0.04 * np.std(torque), size=torque.shape)

current_measured = current + noise_current
rpm_measured = rpm + noise_rpm
torque_measured = torque + noise_torque

# Guardar datos en archivos de texto
print("\nGuardando archivos...")

# Archivo 1: current_measured.txt
with open('current_measured.txt', 'w') as f:
    f.write("# Corriente medida del estator (A)\n")
    f.write("# Tiempo: 0 a 1 segundo, 1000 puntos\n")
    for val in current_measured:
        f.write(f"{val:.6f}\n")

# Archivo 2: rpm_measured.txt
with open('rpm_measured.txt', 'w') as f:
    f.write("# RPM medido del motor\n")
    f.write("# Tiempo: 0 a 1 segundo, 1000 puntos\n")
    for val in rpm_measured:
        f.write(f"{val:.6f}\n")

# Archivo 3: torque_measured.txt
with open('torque_measured.txt', 'w') as f:
    f.write("# Torque medido (N·m)\n")
    f.write("# Tiempo: 0 a 1 segundo, 1000 puntos\n")
    for val in torque_measured:
        f.write(f"{val:.6f}\n")

# Archivo 4: time.txt (opcional)
with open('time.txt', 'w') as f:
    f.write("# Vector de tiempo (segundos)\n")
    for val in t.numpy():
        f.write(f"{val:.6f}\n")

print("✓ current_measured.txt")
print("✓ rpm_measured.txt")
print("✓ torque_measured.txt")
print("✓ time.txt")

# Estadísticas
print("\n=== Estadísticas de los datos generados ===")
print(f"Corriente - Min: {current_measured.min():.4f} A, Max: {current_measured.max():.4f} A, Mean: {current_measured.mean():.4f} A")
print(f"RPM       - Min: {rpm_measured.min():.2f}, Max: {rpm_measured.max():.2f}, Mean: {rpm_measured.mean():.2f}")
print(f"Torque    - Min: {torque_measured.min():.4f} N·m, Max: {torque_measured.max():.4f} N·m, Mean: {torque_measured.mean():.4f} N·m")

print(f"\nParámetros verdaderos usados:")
print(f"rs={true_params[0]}, rr={true_params[1]}, Lls={true_params[2]}, Llr={true_params[3]}")
print(f"Lm={true_params[4]}, J={true_params[5]}, B={true_params[6]}")

print("\n¡Datos sintéticos generados exitosamente!")
print("Ahora puedes cargarlos con:")
print("  current_measured = torch.tensor(np.loadtxt('current_measured.txt'))")
print("  rpm_measured = torch.tensor(np.loadtxt('rpm_measured.txt'))")
print("  torque_measured = torch.tensor(np.loadtxt('torque_measured.txt'))")