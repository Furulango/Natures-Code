import numpy as np
import torch as th

class InductionMotorModel(th.nn.Module):
    def __init__(self, params, vqs, vds):
        super().__init__()

        rs, rr, Lls, Llr, Lm, J, B = params
        self.rs, self.rr, self.J, self.B = rs, rr, J, B
        
        self.Ls = Lls + Lm
        self.Lr = Llr + Lm
        self.Lm = Lm
        
        self.we = 2 * th.pi * 60
        
        # Voltaje in DQ frame (tensors)
        self.vqs = th.tensor(vqs)
        self.vds = th.tensor(vds)
        
        self.P_pairs = 2  # Number of pole pairs

        # Inductance matrix
        self.L = th.tensor([
            [self.Ls, 0, self.Lm, 0], 
            [0, self.Ls, 0, self.Lm], 
            [self.Lm, 0, self.Lr, 0], 
            [0, self.Lm, 0, self.Lr]
        ], dtype=th.float32)

    # El solver llamará a 'forward' en cada paso
    def forward(self, t, x):
        # x es un tensor de Torch: [iqs, ids, iqr, idr, wr]
        iqs, ids, iqr, idr, wr = x[0], x[1], x[2], x[3], x[4]
        
        ws = self.we - wr
        
        lqs = self.Ls*iqs + self.Lm*iqr
        lds = self.Ls*ids + self.Lm*idr
        lqr = self.Lr*iqr + self.Lm*iqs
        ldr = self.Lr*idr + self.Lm*ids
        
        # Mover L y v al dispositivo correcto (ej. 'cuda:0')
        L_device = self.L.to(x.device)
        vqs_device = self.vqs.to(x.device)
        vds_device = self.vds.to(x.device)
        
        # Construir el vector v en la GPU
        v = th.tensor([
            vqs_device - self.rs*iqs - self.we*lds, 
            vds_device - self.rs*ids + self.we*lqs,
            -self.rr*iqr - ws*ldr, 
            -self.rr*idr + ws*lqr
        ], dtype=th.float32, device=x.device)

        di_dt = th.linalg.solve(L_device, v)
        
        Te = (3.0 / 2.0) * self.P_pairs * self.Lm * (iqs * idr - ids * iqr)
        dwr_dt = (Te - self.B*wr) / self.J
        
        # Concatenar en la GPU
        return th.cat((di_dt, th.tensor([dwr_dt], device=x.device, dtype=th.float32)))

    def calculate_torque(self, sol):
        iqs = sol[:, 0]
        ids = sol[:, 1]
        iqr = sol[:, 2]
        idr = sol[:, 3]

        Lm = self.Lm 

        Te = (3.0 / 2.0) * self.P_pairs * Lm * (iqs * idr - ids * iqr)
        
        return Te
    
    def calculate_rpm(self, sol):
        wr = sol[:, 4]
        rpm = wr * 60 / (2 * th.pi)
        return rpm

    def calculate_stator_current(self, sol):
        iqs = sol[:, 0]
        ids = sol[:, 1]

        # Handle both numpy arrays and torch tensors
        if isinstance(sol, th.Tensor):
            Is_mag = th.sqrt(iqs**2 + ids**2)
        else:
            Is_mag = np.sqrt(iqs**2 + ids**2)
            
        return Is_mag
    

    

    
