import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def compute_metrics(pred, target):
    """
    Compute evaluation metrics for predictions.
    """
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()    
    mae = np.mean(np.abs(pred_np - target_np))
    rmse = np.sqrt(np.mean((pred_np - target_np) ** 2))
    ss_total = np.sum((target_np - target_np.mean()) ** 2)
    ss_residual = np.sum((target_np - pred_np) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    ssim_value, _ = ssim(target_np, pred_np, full=True, data_range=target_np.max() - target_np.min())    
    psnr_value = psnr(target_np, pred_np, data_range=target_np.max() - target_np.min())

    return mae, rmse, r2, ssim_value, psnr_value

def calculate_TRI(data):
    """
    Calculate Terrain Ruggedness Index (TRI) for a given 2D grid.
    """
    data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
    tri = np.sqrt(np.mean([
        (data_np[i, j] - data_np[i + di, j + dj]) ** 2
        for i in range(data_np.shape[0] - 1)
        for j in range(data_np.shape[1] - 1)
        for di, dj in [(0, 1), (1, 0), (1, 1), (1, -1)]
    ]))
    return tri