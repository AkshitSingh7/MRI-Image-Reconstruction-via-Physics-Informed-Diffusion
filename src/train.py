import os
import glob
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, CenterSpatialCropd
from monai.networks.schedulers import DDPMScheduler

# local imports
from src.physics import simulate_acquisition
from src.model import build_model

# --- Config ---
EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-4
DATA_URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    # change these paths if running locally vs colab
    data_dir = "dataset"
    if not os.path.exists(data_dir):
        print("Downloading dataset...")
        os.makedirs(data_dir, exist_ok=True)
        os.system(f"wget {DATA_URL} -O ixi.tar")
        os.system(f"tar -xf ixi.tar -C {data_dir}")

    # setup monai transforms
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]), 
        CenterSpatialCropd(keys=["image"], roi_size=(160, 160, 100)),
        Resized(keys=["image"], spatial_size=(128, 128, 64)), 
    ])
    
    files = [{"image": f} for f in glob.glob(f"{data_dir}/*.nii.gz")]
    ds = CacheDataset(data=files, transform=transforms, cache_rate=1.0)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 2. Setup Model
    model = build_model(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler('cuda')

    # 3. Training Loop
    print("Starting training...")
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}")
        
        for step, batch in pbar:
            vol_images = batch["image"].to(device)
            
            # slice 3d -> 2d (pick random depth)
            depth_dim = vol_images.shape[-1]
            idx = torch.randint(0, depth_dim, (1,)).item()
            clean_2d = vol_images[..., idx]
            
            # create the problem (blur)
            blurry_2d, _ = simulate_acquisition(clean_2d)
            blurry_2d = blurry_2d.to(device)
            
            # add noise
            noise = torch.randn_like(clean_2d).to(device)
            timesteps = torch.randint(0, 1000, (clean_2d.shape[0],), device=device).long()
            noisy_2d = scheduler.add_noise(original_samples=clean_2d, noise=noise, timesteps=timesteps)
            
            # train step
            optimizer.zero_grad()
            
            model_input = torch.cat([noisy_2d, blurry_2d], dim=1)
            
            with autocast('cuda'):
                pred = model(model_input, timesteps)
                loss = torch.nn.functional.mse_loss(pred, noise)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
        
        # save checkpoint every 10
        if (epoch + 1) % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
