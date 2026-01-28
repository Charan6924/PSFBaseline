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
def generate_images(I_smooth, I_sharp, mtf_smooth, mtf_sharp, epsilon=1e-6):
    """
    I_smooth, I_sharp: [B, 1, H, W] images
    mtf_smooth, mtf_sharp: [B, 1, H, W] 2D MTF maps
    Returns single [1, 1, H, W] images for discriminator
    """
    B, _, H, W = I_smooth.shape
    
    # FFT with fftshift to center zero frequency
    F_smooth = torch.fft.fftshift(torch.fft.fft2(I_smooth))
    F_sharp = torch.fft.fftshift(torch.fft.fft2(I_sharp))

    # Compute frequency-dependent transfer functions
    mtf_smooth_safe = torch.clamp(mtf_smooth, min=epsilon)
    mtf_sharp_safe = torch.clamp(mtf_sharp, min=epsilon)
    
    H_s2h = mtf_sharp_safe / (mtf_smooth_safe + epsilon)
    H_h2s = mtf_smooth_safe / (mtf_sharp_safe + epsilon)
    
    # Clamp to prevent extreme amplification/suppression
    H_s2h = torch.clamp(H_s2h, min=0.1, max=5.0)
    H_h2s = torch.clamp(H_h2s, min=0.1, max=5.0)

    # Apply transfer functions in frequency domain
    F_gen_sharp = F_smooth * H_s2h
    F_gen_smooth = F_sharp * H_h2s

    # IFFT back to spatial domain
    I_gen_sharp = torch.real(torch.fft.ifft2(torch.fft.ifftshift(F_gen_sharp)))
    I_gen_smooth = torch.real(torch.fft.ifft2(torch.fft.ifftshift(F_gen_smooth)))

    # Normalize WITHOUT in-place operations (create new tensors)
    # Method 1: Normalize globally across the batch
    sharp_min = I_gen_sharp.min()
    sharp_max = I_gen_sharp.max()
    if sharp_max > sharp_min:
        I_gen_sharp = (I_gen_sharp - sharp_min) / (sharp_max - sharp_min)
    else:
        I_gen_sharp = torch.zeros_like(I_gen_sharp)
    
    smooth_min = I_gen_smooth.min()
    smooth_max = I_gen_smooth.max()
    if smooth_max > smooth_min:
        I_gen_smooth = (I_gen_smooth - smooth_min) / (smooth_max - smooth_min)
    else:
        I_gen_smooth = torch.zeros_like(I_gen_smooth)
    
    I_gen_sharp = torch.clamp(I_gen_sharp, 0, 1)
    I_gen_smooth = torch.clamp(I_gen_smooth, 0, 1)

    # Average over batch dimension
    I_gen_sharp = I_gen_sharp.mean(dim=0, keepdim=True)
    I_gen_smooth = I_gen_smooth.mean(dim=0, keepdim=True)

    return I_gen_sharp, I_gen_smooth
def compute_radial_profile(image_2d, H, W):
    """
    Compute radial average of a 2D array
    
    Args:
        image_2d: [H, W] numpy array
        H, W: dimensions
    
    Returns:
        radial_profile: 1D array of radial averages
    """
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    
    max_r = int(np.sqrt(cy**2 + cx**2))
    radial_profile = np.zeros(max_r)
    
    for radius in range(max_r):
        mask = (r == radius)
        if mask.sum() > 0:
            radial_profile[radius] = image_2d[mask].mean()
    
    return radial_profile

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

def validate(net, D_sharp, D_smooth, val_loader, device, bce, l1):
    net.eval()
    D_sharp.eval()
    D_smooth.eval()
    
    val_metrics = {
        'G_loss': 0.0,
        'D_loss': 0.0,
        'recon_loss': 0.0,
        'gan_loss': 0.0,
        'D_sharp_loss': 0.0,
        'D_smooth_loss': 0.0,
    }
    
    with torch.no_grad():
        for I_smooth, I_sharp, _, _ in val_loader:
            I_smooth = I_smooth.to(device)
            I_sharp = I_sharp.to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == "cuda")):
                # Generate scalar kernels [B, 1]
                k_smooth = net(I_smooth)
                k_sharp = net(I_sharp)
                
                # Generate images - returns [1, 1, H, W] (batch-averaged)
                I_gen_sharp, I_gen_smooth = generate_images(I_smooth, I_sharp, k_smooth, k_sharp)
                
                # Generated images are [1, 1, H, W], so B_gen = 1
                B_gen = 1
                real_label = torch.ones(B_gen, 1, device=device)
                fake_label = torch.zeros(B_gen, 1, device=device)
                
                # Use single real sample to match generated image size
                I_sharp_single = I_sharp[0:1]
                I_smooth_single = I_smooth[0:1]
                
                # Discriminator losses
                out_real_sharp = D_sharp(I_sharp_single)
                out_fake_sharp = D_sharp(I_gen_sharp)
                loss_D_sharp = 0.5 * (bce(out_real_sharp, real_label) + bce(out_fake_sharp, fake_label))
                
                out_real_smooth = D_smooth(I_smooth_single)
                out_fake_smooth = D_smooth(I_gen_smooth)
                loss_D_smooth = 0.5 * (bce(out_real_smooth, real_label) + bce(out_fake_smooth, fake_label))
                
                loss_D = loss_D_sharp + loss_D_smooth
                
                # Generator losses
                out_fake_sharp_G = D_sharp(I_gen_sharp)
                out_fake_smooth_G = D_smooth(I_gen_smooth)
                gan_loss = 0.5 * (bce(out_fake_sharp_G, real_label) + bce(out_fake_smooth_G, real_label))
                
                # Reconstruction loss: compare against batch-averaged real images
                I_sharp_avg = I_sharp.mean(dim=0, keepdim=True)
                I_smooth_avg = I_smooth.mean(dim=0, keepdim=True)
                recon_loss = l1(I_gen_sharp, I_sharp_avg) + l1(I_gen_smooth, I_smooth_avg)
                
                loss_G = 0.1 * recon_loss + 0.01 * gan_loss
                
                val_metrics['G_loss'] += loss_G.item()
                val_metrics['D_loss'] += loss_D.item()
                val_metrics['recon_loss'] += recon_loss.item()
                val_metrics['gan_loss'] += gan_loss.item()
                val_metrics['D_sharp_loss'] += loss_D_sharp.item()
                val_metrics['D_smooth_loss'] += loss_D_smooth.item()
    
    for key in val_metrics:
        val_metrics[key] /= len(val_loader)
    
    return val_metrics