
# DQ Induction Motor ODE Simulation using torchdiffeq
from torchdiffeq import odeint
import torch as th

# Other imports
import motor_dynamic
import general_tools

device = 'cuda' if th.cuda.is_available() else 'cpu'

# Motor parameters [rs, rr, Lls, Llr, Lm, J, B]
params = [0.435, 0.816, 0.002, 0.002, 0.0693, 0.089, 0.0001]
vqs, vds = 220.0, 0.0

model = motor_dynamic.InductionMotorModel(params, vqs, vds).to(device)

# Initial state: [iqs, ids, iqr, idr, wr]
x0 = th.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=th.float64, device=device)
# Time points
t = th.linspace(0, 1, 1000, device=device)

print("Solving ODE")
sol = odeint(model, x0, t)
print("Done")

# Show results using matplotlib 
general_tools.plot_results(model, sol, t)



