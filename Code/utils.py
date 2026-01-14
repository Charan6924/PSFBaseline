import torch
import numpy as np

def radial_average(psd):
    y, x = np.indices(psd.shape)
    center = np.array(psd.shape) / 2.0
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-8)
    return radial_profile


# takes in a 2d mtf
def generate_images(I_smooth, I_sharp, mtf_smooth, mtf_sharp, epsilon=1e-8):
    """
    Args:
        I_smooth, I_sharp: [B, 1, H, W] spatial domain images
        mtf_smooth, mtf_sharp: [B, 1, H, W] frequency domain MTF maps (output of KernelEstimator)
        epsilon: Small constant to prevent division by zero
    
    Returns:
        I_gen_sharp, I_gen_smooth: [1, 1, H, W] generated images (averaged over batch)
    """
    
    # 1. Convert Images to Frequency Domain
    # We use fftshift so the zero-frequency component moves from corners (0,0) 
    # to the center (H//2, W//2), matching the alignment of your mtf_smooth/sharp maps.
    F_smooth = torch.fft.fftshift(torch.fft.fft2(I_smooth))
    F_sharp = torch.fft.fftshift(torch.fft.fft2(I_sharp))

    OTF_s2h = mtf_sharp / (mtf_smooth + epsilon)
    OTF_h2s = mtf_smooth / (mtf_sharp + epsilon)
    F_gen_sharp = F_smooth * OTF_s2h
    F_gen_smooth = F_sharp * OTF_h2s
    I_gen_sharp = torch.real(torch.fft.ifft2(torch.fft.ifftshift(F_gen_sharp)))
    I_gen_smooth = torch.real(torch.fft.ifft2(torch.fft.ifftshift(F_gen_smooth)))

    I_gen_sharp = I_gen_sharp.mean(dim=0, keepdim=True).clamp(0, 1)
    I_gen_smooth = I_gen_smooth.mean(dim=0, keepdim=True).clamp(0, 1)

    return I_gen_sharp, I_gen_smooth

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)