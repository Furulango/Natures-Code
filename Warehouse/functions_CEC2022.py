
import numpy as np

def bent_cigar_function(x, y):
    """
    Bent Cigar function
    Global minimum at (0,0) with value 0
    """

    return x**2 + 1e6 * y**2

def rastrigin_function(x, y, A = 10):
    """
    Rastrigin function
    Global minimum at (0,0) with value 0
    """

    return 2*A + x**2 - A * np.cos(2 * np.pi * x) + y**2 - A * np.cos(2 * np.pi * y)

def rosenbrock_function(x, y, A = 1 ,B = 10):
    """
    Rosenbrock function
    Global minimum at (A, A^2) with value 0
    """

    return (A - x)**2 + B * (y - x**2)**2

def griewank_function(x, y): 
    """
    Griewank function
    Global minimum at (0,0) with value 0
    """

    sum_term = (x**2) / 4000 + (y**2) / 4000
    prod_term = np.cos(x / np.sqrt(1)) * np.cos(y / np.sqrt(2))
    return 1 + sum_term - prod_term

def ackley_function(x, y, a = 20, b = 0.2, c = 2 * np.pi):
    """
    Ackley function
    Global minimum at (0,0) with value 0
    """

    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
    return term1 + term2 + a + np.exp(1)