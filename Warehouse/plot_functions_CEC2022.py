
import numpy as np
import matplotlib.pyplot as plt

import functions_CEC2022 as func_2022


x_min, x_max = -5, 5
y_min, y_max = -5, 5

# Bent Cigar
X1 , Y1 = np.meshgrid(np.linspace(x_min, x_max, 100), 
                    np.linspace(y_min, y_max, 100))
Z1 = np.vectorize(func_2022.bent_cigar_function)(X1, Y1)

# Rastrigin
X2 , Y2 = np.meshgrid(np.linspace(x_min, x_max, 100), 
                    np.linspace(y_min, y_max, 100))
Z2 = np.vectorize(func_2022.rastrigin_function)(X2, Y2)

# Rosenbrock
X3 , Y3 = np.meshgrid(np.linspace(x_min, x_max, 100), 
                    np.linspace(y_min, y_max, 100))
Z3 = np.vectorize(func_2022.rosenbrock_function)(X3, Y3)

# Griewank
X4 , Y4 = np.meshgrid(np.linspace(x_min - 15, x_max + 15, 100), 
                    np.linspace(y_min - 15, y_max + 15, 100))
Z4 = np.vectorize(func_2022.griewank_function)(X4, Y4)

# Ackley
X5 , Y5 = np.meshgrid(np.linspace(x_min, x_max, 100), 
                    np.linspace(y_min, y_max, 100))
Z5 = np.vectorize(func_2022.ackley_function)(X5, Y5)

# Graficas
fig = plt.figure(figsize=(10, 8))

ax1 = fig.add_subplot(2, 3, 1, projection='3d')
surf1 = ax1.plot_surface(X1, Y1, Z1, cmap='plasma')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

ax2 = fig.add_subplot(2, 3, 2, projection='3d')
surf2 = ax2.plot_surface(X2, Y2, Z2, cmap='plasma')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

ax3 = fig.add_subplot(2, 3, 3, projection='3d')
surf3 = ax3.plot_surface(X3, Y3, Z3, cmap='plasma')
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)

ax4 = fig.add_subplot(2, 3, 4, projection='3d')
surf4 = ax4.plot_surface(X4, Y4, Z4, cmap='plasma')
fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10)

ax5 = fig.add_subplot(2, 3, 5, projection='3d')
surf5 = ax5.plot_surface(X5, Y5, Z5, cmap='plasma')
fig.colorbar(surf5, ax=ax5, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()

