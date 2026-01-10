import torch
from monai.networks.nets import DiffusionModelUNet

def build_model(device):
    # standard monai conditional unet
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=2,      # input is noise + blurry image
        out_channels=1,     # output is just the noise prediction
        channels=(64, 128, 256), 
        attention_levels=(False, True, True),
        num_res_blocks=2,
        num_head_channels=32, 
    ).to(device)
    return model
