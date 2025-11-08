
import numpy as np
import torch as th
import matplotlib.pyplot as plt

from motor_dynamic import InductionMotorModel as IModel

def plot_results(model, sol, t):

    """
    Args:
        model: Instance of InductionMotorModel
        sol: Tensor of shape (time_steps, 5) with the solution from the ODE solver
        t: Tensor of time points corresponding to the solution
    """

    sol_np = sol.cpu().numpy()
    t_np = t.cpu().numpy()
    
    Te = model.calculate_torque(sol_np)
    Rpm = model.calculate_rpm(sol_np)
    Is = model.calculate_stator_current(sol_np)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # RPM
    ax1.plot(t_np, Rpm, label='Rotor Speed (Rpm)', color='blue')
    ax1.set_ylabel('Speed (Rpm)')
    ax1.set_title('Induction Motor Rotor Speed')
    ax1.legend()
    ax1.grid(True)

    # Current
    ax2.plot(t_np, Is, label='Stator current (Is)', linestyle='-')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Current (A)')
    ax2.legend()
    ax2.grid(True)

    # Torque
    ax3.plot(t_np, Te, label='Electromagnetic Torque (Te)', color='green')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Torque (Nm)')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
