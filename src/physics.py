import torch

def simulate_acquisition(image_tensor):
    # input: [Batch, Channel, Height, Width]
    # mimics the MRI scanner skipping frequencies (4x accel)
    rows, cols = image_tensor.shape[-2:] 
    
    # move to frequency domain
    k_space = torch.fft.fft2(image_tensor, dim=(-2, -1))
    k_space_shifted = torch.fft.fftshift(k_space, dim=(-2, -1))
    
    # create the mask (keep center 8%)
    mask = torch.zeros_like(k_space_shifted)
    center_r, center_c = rows // 2, cols // 2
    center_width = int(rows * 0.08)
    
    mask[..., center_r - center_width : center_r + center_width, 
            center_c - center_width : center_c + center_width] = 1
            
    # apply mask and go back to image space
    masked_k_space = k_space_shifted * mask
    ishifted = torch.fft.ifftshift(masked_k_space, dim=(-2, -1))
    undersampled_image = torch.fft.ifft2(ishifted, dim=(-2, -1)).real
    
    return undersampled_image, mask

def apply_data_consistency(curr_image, original_kspace, mask):
    # the enforcer: forces known frequencies to match the scanner data
    curr_kspace = torch.fft.fft2(curr_image, dim=(-2, -1))
    curr_kspace_shifted = torch.fft.fftshift(curr_kspace, dim=(-2, -1))
    
    # where mask is 1, use REAL data. where mask is 0, use AI guess
    updated_kspace = (original_kspace * mask) + (curr_kspace_shifted * (1 - mask))
    
    # back to image
    ishifted = torch.fft.ifftshift(updated_kspace, dim=(-2, -1))
    updated_image = torch.fft.ifft2(ishifted, dim=(-2, -1)).real
    
    return updated_image
