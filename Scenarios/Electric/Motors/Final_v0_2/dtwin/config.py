# dtwin/config.py
from dataclasses import dataclass

@dataclass
class MotorConfig:
    poles: int = 4
    fe_hz: float = 60.0
    # Límites físicos razonables
    rs_bounds: tuple = (0.1, 10.0)
    rr_bounds: tuple = (0.1, 10.0)
    lls_bounds: tuple = (1e-4, 5e-2)
    llr_bounds: tuple = (1e-4, 5e-2)
    lm_bounds:  tuple = (1e-3, 5e-1)
    j_bounds:   tuple = (1e-4, 5e-1)
    b_bounds:   tuple = (1e-5, 1e-2)

@dataclass
class LossWeights:
    w_i_mag: float = 1.0
    w_i_components: float = 0.2
    w_torque: float = 0.5
    w_rpm: float = 0.3
    penalty_phys: float = 1e3  # penalizaciones por inconsistencia física

@dataclass
class PSOConfig:
    swarm_size: int = 256
    iters: int = 400
    w: float = 0.72           # inercia (constriction-friendly)
    c1: float = 1.49          # cognitivo
    c2: float = 1.49          # social
    vmax_frac: float = 0.2    # límite de velocidad relativo a rango
    seed: int = 42

@dataclass
class ThermalCoeffs:
    # coeficientes por °C respecto a Tref
    rs_alpha: float = 0.004
    rr_alpha: float = 0.004
    lm_alpha: float = -0.0005
    tref_c: float = 20.0

@dataclass
class Paths:
    data_stage1_csv: str = "data/measurements_stage1.csv"
    data_stage2_csv: str = "data/measurements_stage2_hot.csv"
    out_dir: str = "outputs"

motor_cfg = MotorConfig()
loss_w = LossWeights()
pso_cfg = PSOConfig()
thermal = ThermalCoeffs()
paths = Paths()
