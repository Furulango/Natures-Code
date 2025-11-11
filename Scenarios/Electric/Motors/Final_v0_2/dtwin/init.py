# dtwin/__init__.py
__version__ = "0.1.0"
__all__ = ["config", "model", "sim", "objective", "optim", "pipeline", "scenarios", "data"]

def has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
