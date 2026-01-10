import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.networks.schedulers import DDPMScheduler
import argparse
import glob 
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, CenterSpatialCropd

# local imports
from src.physics import simulate_acquisition, apply_data_consistency
from src.model import build_model
from src.utils import compute_stats

def run_inference(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading from {ckpt_path} on {device}")
    
    # load model
    model = build_model(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # quick data loader for testing
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]), 
        CenterSpatialCropd(keys=["image"], roi_size=(160, 160, 100)),
        Resized(keys=["image"], spatial_size=(128, 128, 64)), 
    ])
    files = [{"image": f} for f in glob.glob("dataset/*.nii.gz")][:5]
    loader = DataLoader(CacheDataset(files, transforms), batch_size=1)
    
    # grab a sample
    data = next(iter(loader))
    vol = data["image"].to(device)
    clean_img = vol[0, :, :, :, vol.shape[-1]//2].unsqueeze(0) # middle slice

    # prep physics stuff
    print("Running Physics-Guided Reconstruction...")
    
    rows, cols = clean_img.shape[-2:]
    k_space = torch.fft.fft2(clean_img, dim=(-2, -1))
    k_shifted = torch.fft.fftshift(k_space, dim=(-2, -1))
    
    # make mask
    mask = torch.zeros_like(k_shifted)
    cr, cc = rows // 2, cols // 2
    cw = int(rows * 0.08)
    mask[..., cr-cw : cr+cw, cc-cw : cc+cw] = 1
    
    blurry_img, _ = simulate_acquisition(clean_img)
    blurry_img = blurry_img.to(device)
    
    known_kspace = k_shifted.to(device) * mask.to(device)
    mask = mask.to(device)
    
    # start from 50% noise (refinement strategy)
    START_STEP = 500
    noise = torch.randn_like(clean_img).to(device)
    latents = scheduler.add_noise(blurry_img, noise, torch.tensor([START_STEP], device=device))
    
    # denoising loop
    with torch.no_grad():
        timesteps = scheduler.timesteps[scheduler.timesteps < START_STEP]
        
        for t in tqdm(timesteps):
            # predict noise
            inp = torch.cat([latents, blurry_img], dim=1)
            pred = model(inp, torch.tensor([t], device=device))
            
            # step
            latents, _ = scheduler.step(pred, t, latents)
            
            # FORCE PHYSICS
            latents = apply_data_consistency(latents, known_kspace, mask)

    # calc metrics
    psnr, ssim, blur_psnr = compute_stats(latents, clean_img, blurry_img)
    print(f"Results -> Input PSNR: {blur_psnr:.2f} | AI PSNR: {psnr:.2f}")
    
    # save plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1); plt.imshow(blurry_img[0,0].cpu(), cmap='gray'); plt.title(f"Input {blur_psnr:.2f}")
    plt.subplot(1,3,2); plt.imshow(latents[0,0].cpu(), cmap='gray'); plt.title(f"AI {psnr:.2f}")
    plt.subplot(1,3,3); plt.imshow(clean_img[0,0].cpu(), cmap='gray'); plt.title("Ground Truth")
    plt.savefig("results.png")
    print("Saved result to results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth file")
    args = parser.parse_args()
    
    run_inference(args.checkpoint)
