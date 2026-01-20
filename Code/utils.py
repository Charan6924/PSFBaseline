import torch
import numpy as np
import matplotlib.pyplot as plt
import os 
from collections import defaultdict

device = 'cuda'

def radial_average(psd):
    y, x = np.indices(psd.shape)
    center = np.array(psd.shape) / 2.0
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-8)
    return radial_profile

def plot_final_loss_curves(loss_history, save_path='training_metrics.png'):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Metrics - Complete History', fontsize=16)
    
    epochs = list(range(1, len(loss_history['total_G_loss']) + 1))
    
    axes[0, 0].plot(epochs, loss_history['total_G_loss'], label='G Loss', color='blue', alpha=0.6)
    axes[0, 0].plot(epochs, loss_history['total_G_loss_ma'], label='G Loss (MA)', color='blue', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Generator Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, loss_history['total_D_loss'], label='D Loss', color='red', alpha=0.6)
    axes[0, 1].plot(epochs, loss_history['total_D_loss_ma'], label='D Loss (MA)', color='red', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Discriminator Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(epochs, loss_history['recon_loss'], label='Recon Loss', color='green', alpha=0.6)
    axes[0, 2].plot(epochs, loss_history['recon_loss_ma'], label='Recon Loss (MA)', color='green', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Reconstruction Loss (L1)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, loss_history['gan_loss'], label='GAN Loss', color='purple', alpha=0.6)
    axes[1, 0].plot(epochs, loss_history['gan_loss_ma'], label='GAN Loss (MA)', color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('GAN Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, loss_history['D_sharp_loss'], label='D Sharp', color='orange', alpha=0.6)
    axes[1, 1].plot(epochs, loss_history['D_smooth_loss'], label='D Smooth', color='cyan', alpha=0.6)
    axes[1, 1].plot(epochs, loss_history['D_sharp_loss_ma'], label='D Sharp (MA)', color='orange', linewidth=2)
    axes[1, 1].plot(epochs, loss_history['D_smooth_loss_ma'], label='D Smooth (MA)', color='cyan', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Individual Discriminator Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(epochs, loss_history['G_grad_norm'], label='Generator', color='blue', alpha=0.6)
    axes[1, 2].plot(epochs, loss_history['D_grad_norm'], label='Discriminators', color='red', alpha=0.6)
    axes[1, 2].plot(epochs, loss_history['G_grad_norm_ma'], label='Generator (MA)', color='blue', linewidth=2)
    axes[1, 2].plot(epochs, loss_history['D_grad_norm_ma'], label='Discriminators (MA)', color='red', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Gradient Norm')
    axes[1, 2].set_title('Gradient Norms')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to {save_path}")

def save_checkpoint(epoch, net, D_sharp, D_smooth, opt_G, opt_Ds, scaler, loss_history, filepath):
    checkpoint = {
        'epoch': epoch,
        'net_state_dict': net.state_dict(),
        'D_sharp_state_dict': D_sharp.state_dict(),
        'D_smooth_state_dict': D_smooth.state_dict(),
        'opt_G_state_dict': opt_G.state_dict(),
        'opt_Ds_state_dict': opt_Ds.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss_history': dict(loss_history),
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")
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

def load_checkpoint(filepath, net, D_sharp, D_smooth, opt_G, opt_Ds, scaler):
    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}")
        return 0, defaultdict(list)
    
    checkpoint = torch.load(filepath, map_location=device)
    net.load_state_dict(checkpoint['net_state_dict'])
    D_sharp.load_state_dict(checkpoint['D_sharp_state_dict'])
    D_smooth.load_state_dict(checkpoint['D_smooth_state_dict'])
    opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
    opt_Ds.load_state_dict(checkpoint['opt_Ds_state_dict'])
    
    if scaler and checkpoint['scaler_state_dict']:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    epoch = checkpoint['epoch']
    loss_history = defaultdict(list, checkpoint.get('loss_history', {}))
    print(f"Checkpoint loaded: {filepath} (epoch {epoch})")
    return epoch + 1, loss_history

def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def update_moving_average(history, window=10):
    if len(history) < window:
        return sum(history) / len(history)
    return sum(history[-window:]) / window