import torch

def custom_loss(pred, target, radar_mask_batch, smoothing=1e-4):
    radar_indices = radar_mask_batch
    non_radar_indices = ~radar_indices    
    loss_radar = (
        torch.mean((pred[radar_indices] - target[radar_indices]) ** 2)
        if radar_indices.sum() > 0
        else torch.tensor(0.0, device=pred.device)
    )
    loss_bedmachine = (
        torch.mean((pred[non_radar_indices] - target[non_radar_indices]) ** 2)
        if non_radar_indices.sum() > 0
        else torch.tensor(0.0, device=pred.device)
    )

    gamma_sum = loss_radar + loss_bedmachine + smoothing
    gamma_i_dynamic = loss_bedmachine / gamma_sum
    gamma_j_dynamic = loss_radar / gamma_sum

    total_loss = gamma_i_dynamic * loss_radar + gamma_j_dynamic * loss_bedmachine
    return loss_radar, loss_bedmachine, total_loss