import torch as th

class InductionMotorModelBatch(th.nn.Module):
    """
    Induction Motor Model - BATCH VERSION
    Procesa múltiples conjuntos de parámetros simultáneamente
    """
    def __init__(self, vqs=220.0, vds=0.0, p_pairs=2):
        super().__init__()
        self.we = 2 * th.pi * 60  # eléctrica (rad/s) a 60 Hz
        self.vqs_base = float(vqs)
        self.vds_base = float(vds)
        self.P_pairs = int(p_pairs)

        # Programación de voltaje (constante por defecto)
        self.t_off = None  # si se define, V=0 para t >= t_off
        # Torque de carga (constante por defecto)
        self.TL_const = 0.0

        # Batch parameters (will be set by update_params_batch)
        self.batch_size = 1
        self.device = 'cuda' if th.cuda.is_available() else 'cpu'

    def set_voltage_schedule(self, t_off=None, vqs_on=None, vds_on=None):
        if vqs_on is not None:
            self.vqs_base = float(vqs_on)
        if vds_on is not None:
            self.vds_base = float(vds_on)
        self.t_off = float(t_off) if t_off is not None else None

    def set_load_torque(self, TL=0.0):
        self.TL_const = float(TL)

    def _voltage_at(self, t, device):
        # t puede llegar como tensor o float
        t_val = float(t) if isinstance(t, (float, int)) else float(t.item())
        if self.t_off is not None and t_val >= self.t_off:
            return th.tensor(self.vqs_base*0.0, dtype=th.float32, device=device), th.tensor(self.vds_base*0.0, dtype=th.float32, device=device)
        return th.tensor(self.vqs_base, dtype=th.float32, device=device), th.tensor(self.vds_base, dtype=th.float32, device=device)

    def update_params_batch(self, params_batch):
        """
        Update parameters for batch processing
        params_batch shape: (batch_size, 7) -> [rs, rr, Lls, Llr, Lm, J, B]
        """
        self.batch_size = params_batch.shape[0]
        self.device = params_batch.device

        self.rs = params_batch[:, 0]
        self.rr = params_batch[:, 1]
        Lls = params_batch[:, 2]
        Llr = params_batch[:, 3]
        self.Lm = params_batch[:, 4]
        self.J = params_batch[:, 5]
        self.B = params_batch[:, 6]

        self.Ls = Lls + self.Lm
        self.Lr = Llr + self.Lm

        # Inductance matrix (batch_size, 4, 4)
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
        x: (batch_size, 5) -> [iqs, ids, iqr, idr, wr]
        returns dx_dt: (batch_size, 5)
        """
        iqs = x[:, 0]
        ids = x[:, 1]
        iqr = x[:, 2]
        idr = x[:, 3]
        wr  = x[:, 4]

        # Frecuencia de deslizamiento
        ws = self.we - self.P_pairs * wr

        # Enlaces de flujo
        lqs = self.Ls * iqs + self.Lm * iqr
        lds = self.Ls * ids + self.Lm * idr
        lqr = self.Lr * iqr + self.Lm * iqs
        ldr = self.Lr * idr + self.Lm * ids

        # Voltaje programado en el tiempo
        vqs_t, vds_t = self._voltage_at(t, x.device)

        # Ecuaciones de tensión
        v = th.stack([
            vqs_t - self.rs * iqs - self.we * lds,
            vds_t - self.rs * ids + self.we * lqs,
            -self.rr * iqr - ws * ldr,
            -self.rr * idr + ws * lqr
        ], dim=1)

        # Derivadas de corriente
        di_dt = th.linalg.solve(self.L_batch, v.unsqueeze(-1)).squeeze(-1)

        # Par electromagnético
        Te = (3.0 / 2.0) * self.P_pairs * self.Lm * (iqs * idr - ids * iqr)

        # Torque de carga (constante por defecto)
        TL = th.tensor(self.TL_const, dtype=th.float32, device=x.device)

        # Dinámica mecánica
        dwr_dt = (Te - TL - self.B * wr) / self.J

        return th.cat([di_dt, dwr_dt.unsqueeze(-1)], dim=1)

    def calculate_torque(self, sol):
        iqs = sol[:, :, 0]
        ids = sol[:, :, 1]
        iqr = sol[:, :, 2]
        idr = sol[:, :, 3]
        Te = (3.0 / 2.0) * self.P_pairs * self.Lm.unsqueeze(1) * (iqs * idr - ids * iqr)
        return Te

    def calculate_rpm(self, sol):
        wr = sol[:, :, 4]
        rpm = wr * 60 / (2 * th.pi)
        return rpm

    def calculate_stator_current(self, sol):
        iqs = sol[:, :, 0]
        ids = sol[:, :, 1]
        Is_mag = th.sqrt(iqs**2 + ids**2)
        return Is_mag
