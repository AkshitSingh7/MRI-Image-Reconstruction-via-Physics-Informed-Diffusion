import torch
from monai.metrics import PSNRMetric, SSIMMetric

def get_metrics():
    # setup standard metrics
    psnr_metric = PSNRMetric(max_val=1.0)
    ssim_metric = SSIMMetric(data_range=1.0, spatial_dims=2)
    return psnr_metric, ssim_metric

def compute_stats(recon, clean, blurry):
    # clamp to be safe for metric calc
    recon = torch.clamp(recon, 0, 1)
    clean = torch.clamp(clean, 0, 1)
    blurry = torch.clamp(blurry, 0, 1)
    
    psnr_func, ssim_func = get_metrics()
    
    score_ai = psnr_func(recon, clean).item()
    score_ssim = ssim_func(recon, clean).item()
    score_input = psnr_func(blurry, clean).item()
    
    return score_ai, score_ssim, score_input
