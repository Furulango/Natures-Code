# check_b_prior.py

from config import NAMEPLATE
from utils import compute_b_prior_from_nameplate

if __name__ == "__main__":
    print("NAMEPLATE:", NAMEPLATE)
    b_prior = compute_b_prior_from_nameplate(NAMEPLATE)
    print(f"B_prior calculado = {b_prior:.6e}")
