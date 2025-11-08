import torch as th

class InductionMotorModelBatch(th.nn.Module):
    """
    Induction Motor Model - BATCH VERSION
    Procesa múltiples conjuntos de parámetros simultáneamente
    """
    def __init__(self, vqs=220.0, vds=0.0):
        super().__init__()
        
        self.we = 2 * th.pi * 60
        self.vqs = th.tensor(vqs, dtype=th.float32)
        self.vds = th.tensor(vds, dtype=th.float32)
        self.P_pairs = 2
        
        # Batch parameters (will be set by update_params_batch)
        self.batch_size = 1
        self.device = 'cuda' if th.cuda.is_available() else 'cpu'
    
    def update_params_batch(self, params_batch):
        """
        Update parameters for batch processing
        
        Args:
            params_batch: Tensor of shape (batch_size, 7)
                         Each row: [rs, rr, Lls, Llr, Lm, J, B]
        """
        self.batch_size = params_batch.shape[0]
        self.device = params_batch.device
        
        # Extract parameters (batch_size,)
        self.rs = params_batch[:, 0]
        self.rr = params_batch[:, 1]
        Lls = params_batch[:, 2]
        Llr = params_batch[:, 3]
        self.Lm = params_batch[:, 4]
        self.J = params_batch[:, 5]
        self.B = params_batch[:, 6]
        
        # Derived parameters
        self.Ls = Lls + self.Lm
        self.Lr = Llr + self.Lm
        
        # Create batch inductance matrices (batch_size, 4, 4)
        self.L_batch = th.zeros(self.batch_size, 4, 4, device=self.device, dtype=th.float32)
        self.L_batch[:, 0, 0] = self.Ls
        self.L_batch[:, 1, 1] = self.Ls
        self.L_batch[:, 2, 2] = self.Lr
        self.L_batch[:, 3, 3] = self.Lr
        self.L_batch[:, 0, 2] = self.Lm
        self.L_batch[:, 1, 3] = self.Lm
        self.L_batch[:, 2, 0] = self.Lm
        self.L_batch[:, 3, 1] = self.Lm
    
    def forward(self, t, x):
        """
        ODE function for batch integration
        
        Args:
            t: time
            x: state tensor of shape (batch_size, 5)
               [iqs, ids, iqr, idr, wr] for each motor
        
        Returns:
            dx_dt: derivatives (batch_size, 5)
        """
        # Extract states (batch_size,)
        iqs = x[:, 0]
        ids = x[:, 1]
        iqr = x[:, 2]
        idr = x[:, 3]
        wr = x[:, 4]
        
        # Slip frequency
        ws = self.we - wr
        
        # Flux linkages
        lqs = self.Ls * iqs + self.Lm * iqr
        lds = self.Ls * ids + self.Lm * idr
        lqr = self.Lr * iqr + self.Lm * iqs
        ldr = self.Lr * idr + self.Lm * ids
        
        # Move voltage to correct device
        vqs_device = self.vqs.to(x.device)
        vds_device = self.vds.to(x.device)
        
        # Voltage equations (batch_size, 4)
        v = th.stack([
            vqs_device - self.rs * iqs - self.we * lds,
            vds_device - self.rs * ids + self.we * lqs,
            -self.rr * iqr - ws * ldr,
            -self.rr * idr + ws * lqr
        ], dim=1)
        
        # Solve for current derivatives (batch_size, 4)
        di_dt = th.linalg.solve(self.L_batch, v.unsqueeze(-1)).squeeze(-1)
        
        # Torque (batch_size,)
        Te = (3.0 / 2.0) * self.P_pairs * self.Lm * (iqs * idr - ids * iqr)
        
        # Speed derivative
        dwr_dt = (Te - self.B * wr) / self.J
        
        # Concatenate derivatives (batch_size, 5)
        return th.cat([di_dt, dwr_dt.unsqueeze(-1)], dim=1)
    
    def calculate_torque(self, sol):
        """
        Calculate torque from solution
        
        Args:
            sol: (batch_size, time_steps, 5)
        
        Returns:
            Te: (batch_size, time_steps)
        """
        iqs = sol[:, :, 0]
        ids = sol[:, :, 1]
        iqr = sol[:, :, 2]
        idr = sol[:, :, 3]
        
        Te = (3.0 / 2.0) * self.P_pairs * self.Lm.unsqueeze(1) * (iqs * idr - ids * iqr)
        return Te
    
    def calculate_rpm(self, sol):
        """
        Calculate RPM from solution
        
        Args:
            sol: (batch_size, time_steps, 5)
        
        Returns:
            rpm: (batch_size, time_steps)
        """
        wr = sol[:, :, 4]
        rpm = wr * 60 / (2 * th.pi)
        return rpm
    
    def calculate_stator_current(self, sol):
        """
        Calculate stator current magnitude from solution
        
        Args:
            sol: (batch_size, time_steps, 5)
        
        Returns:
            Is_mag: (batch_size, time_steps)
        """
        iqs = sol[:, :, 0]
        ids = sol[:, :, 1]
        Is_mag = th.sqrt(iqs**2 + ids**2)
        return Is_mag