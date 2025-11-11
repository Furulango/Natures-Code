# dtwin/objective/losses.py
import torch

def physical_penalty(params):
    # penaliza negatividad y Lm excesivo respecto a Lls/Llr
    Rs, Rr, Lls, Llr, Lm, J, Bm = params.unbind(dim=1)
    pen = torch.zeros(params.shape[0], device=params.device, dtype=params.dtype)
    pen += torch.relu(-Rs) + torch.relu(-Rr) + torch.relu(-Lls) + torch.relu(-Llr) + torch.relu(-Lm) + torch.relu(-J) + torch.relu(-Bm)
    Ls = Lls + Lm
    Lr = Llr + Lm
    # penaliza si Lm > 0.98*min(Ls,Lr)
    lim = torch.minimum(Ls, Lr) * 0.98
    pen += torch.relu(Lm - lim)
    return pen

def multichannel_loss(sim, meas, loss_w, params, penalty_scale):
    # sim, meas: dict con iqs, ids, torque, rpm, i_mag
    i_mag_sim = sim["i_mag"]
    i_mag_mea = torch.sqrt(meas["iqs"]**2 + meas["ids"]**2)
    li_mag = torch.mean((i_mag_sim - i_mag_mea)**2, dim=1)

    li_comp = torch.mean((sim["states"][:,:,0] - meas["iqs"])**2 + (sim["states"][:,:,1] - meas["ids"])**2, dim=1)
    ltorque = torch.mean((sim["torque"] - meas["torque"])**2, dim=1)
    lrpm = torch.mean((sim["rpm"] - meas["rpm"])**2, dim=1)

    L = (loss_w.w_i_mag*li_mag +
         loss_w.w_i_components*li_comp +
         loss_w.w_torque*ltorque +
         loss_w.w_rpm*lrpm)
    pen = physical_penalty(params) * penalty_scale
    return L + pen
