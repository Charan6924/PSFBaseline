import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PSDDataset import PSDDataset
from Generator import KernelEstimator
from Discriminator import Discriminator
from utils import generate_images, normalize, save_checkpoint, load_checkpoint, plot_final_loss_curves, compute_gradient_norm, validate, compute_radial_profile
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import os
import logging
from tqdm import tqdm

def train():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training_log.txt'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    file_handler = logging.FileHandler('training_log.txt')
    file_handler.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    save_dir = "generated_images_spline"
    kernel_dir = "generated_mtfs"
    checkpoint_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(kernel_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("="*80)
    logger.info(f"Training Configuration - Device: {device}")
    logger.info("="*80)
    
    net = KernelEstimator().to(device)
    D_sharp = Discriminator().to(device)
    D_smooth = Discriminator().to(device)
    bce = nn.BCEWithLogitsLoss().to(device)
    l1 = nn.L1Loss().to(device)
    
    # Balanced learning rates
    opt_G = torch.optim.Adam(net.parameters(), lr=1e-4)
    opt_Ds = torch.optim.Adam(
        list(D_sharp.parameters()) + list(D_smooth.parameters()),
        lr=1e-5  # Slower discriminator learning
    )
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None        #type : ignore
    
    BATCH_SIZE = 64  # Increased from 32 - you have GPU headroom!
    NUM_WORKERS = 8   # Increased for faster data loading
    full_dataset = PSDDataset(
        root_dir=r"D:\Charan work file\KernelEstimator\Data_Root",
        sampling_strategy='all',
        use_ct_windowing=True,
    )
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Total dataset: {len(full_dataset)} slices")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    start_epoch = 0
    num_epochs = 100
    latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    
    start_epoch, loss_history = load_checkpoint(
        latest_checkpoint, net, D_sharp, D_smooth, opt_G, opt_Ds, scaler
    )
    if 'val_G_loss' not in loss_history:
        loss_history['val_G_loss'] = []
        loss_history['val_D_loss'] = []
        loss_history['val_recon_loss'] = []
        loss_history['val_gan_loss'] = []

    logger.info(f"Starting from Epoch: {start_epoch + 1}, Total Epochs: {num_epochs}")
    logger.info(f"Batch size: {BATCH_SIZE}, Steps per epoch: {len(train_loader)}")
    logger.info("="*80)
    
    for epoch in range(start_epoch, num_epochs):
        net.train()
        D_sharp.train()
        D_smooth.train()
        
        epoch_metrics = {
            'G_loss': 0.0,
            'D_loss': 0.0,
            'recon_loss': 0.0,
            'gan_loss': 0.0,
            'D_sharp_loss': 0.0,
            'D_smooth_loss': 0.0,
            'G_grad_norm': 0.0,
            'D_grad_norm': 0.0,
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (I_smooth, I_sharp, psd_smooth, psd_sharp) in enumerate(pbar):
            I_smooth = I_smooth.to(device)
            I_sharp = I_sharp.to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == "cuda")): #type: ignore
                mtf_smooth = net(I_smooth)
                mtf_sharp = net(I_sharp)
                I_gen_sharp, I_gen_smooth = generate_images(
                    I_smooth, I_sharp, mtf_smooth, mtf_sharp
                )
            
            # Visualization (first batch only)
            if batch_idx == 0:
                with torch.no_grad():
                    n_samples = min(4, I_sharp.size(0))
                    
                    for i in range(n_samples):
                        I_s_single = I_smooth[i:i+1]
                        I_h_single = I_sharp[i:i+1]
                        mtf_s_single = mtf_smooth[i:i+1]
                        mtf_h_single = mtf_sharp[i:i+1]
                        
                        F_s = torch.fft.fftshift(torch.fft.fft2(I_s_single))
                        F_h = torch.fft.fftshift(torch.fft.fft2(I_h_single))
                        
                        mtf_s_safe = mtf_s_single.clamp(min=1e-6)
                        mtf_h_safe = mtf_h_single.clamp(min=1e-6)
                        
                        H_s2h = (mtf_h_safe / (mtf_s_safe + 1e-6)).clamp(0.1, 5.0)
                        H_h2s = (mtf_s_safe / (mtf_h_safe + 1e-6)).clamp(0.1, 5.0)
                        
                        F_gen_sharp = F_s * H_s2h
                        F_gen_smooth = F_h * H_h2s
                        
                        I_gen_sharp_vis = torch.real(torch.fft.ifft2(torch.fft.ifftshift(F_gen_sharp)))
                        I_gen_smooth_vis = torch.real(torch.fft.ifft2(torch.fft.ifftshift(F_gen_smooth)))
                        
                        sharp_min, sharp_max = I_gen_sharp_vis.min(), I_gen_sharp_vis.max()
                        if sharp_max > sharp_min:
                            I_gen_sharp_vis = (I_gen_sharp_vis - sharp_min) / (sharp_max - sharp_min)
                        
                        smooth_min, smooth_max = I_gen_smooth_vis.min(), I_gen_smooth_vis.max()
                        if smooth_max > smooth_min:
                            I_gen_smooth_vis = (I_gen_smooth_vis - smooth_min) / (smooth_max - smooth_min)
                        
                        I_gen_sharp_vis = I_gen_sharp_vis.clamp(0, 1)
                        I_gen_smooth_vis = I_gen_smooth_vis.clamp(0, 1)
                        
                        imgs = torch.stack([
                            I_s_single[0],
                            I_gen_smooth_vis[0],
                            I_h_single[0],
                            I_gen_sharp_vis[0],
                        ])
                        save_path = os.path.join(save_dir, f"epoch{epoch + 1}_sample{i + 1}.png")
                        save_image(imgs, save_path, nrow=2)
                    
                    # MTF visualization
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    
                    mtf_smooth_center = mtf_smooth[0, 0].cpu().numpy()
                    mtf_sharp_center = mtf_sharp[0, 0].cpu().numpy()
                    
                    im0 = axes[0, 0].imshow(np.log10(mtf_smooth_center + 1e-8), cmap='hot')
                    axes[0, 0].set_title('MTF Smooth (log scale)')
                    plt.colorbar(im0, ax=axes[0, 0])
                    
                    im1 = axes[0, 1].imshow(np.log10(mtf_sharp_center + 1e-8), cmap='hot')
                    axes[0, 1].set_title('MTF Sharp (log scale)')
                    plt.colorbar(im1, ax=axes[0, 1])
                    
                    mtf_diff = mtf_sharp_center - mtf_smooth_center
                    im2 = axes[0, 2].imshow(mtf_diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
                    axes[0, 2].set_title('MTF Difference (Sharp - Smooth)')
                    plt.colorbar(im2, ax=axes[0, 2])
                    
                    H, W = mtf_smooth_center.shape
                    radial_smooth = compute_radial_profile(mtf_smooth_center, H, W)
                    radial_sharp = compute_radial_profile(mtf_sharp_center, H, W)
                    
                    num_points = len(radial_smooth)
                    freq_axis = np.linspace(0, 0.5, num_points)
                    
                    axes[1, 0].plot(freq_axis, radial_smooth, label='Smooth MTF', linewidth=2)
                    axes[1, 0].plot(freq_axis, radial_sharp, label='Sharp MTF', linewidth=2)
                    axes[1, 0].set_xlabel('Normalized Frequency')
                    axes[1, 0].set_ylabel('MTF')
                    axes[1, 0].set_title('Radial MTF Profiles')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                    axes[1, 0].set_xlim(0, 0.5)
                    axes[1, 0].set_ylim(0, 1.05)
                    
                    center_y, center_x = H // 2, W // 2
                    
                    axes[1, 1].plot(mtf_smooth_center[center_y, :], label='Smooth', linewidth=2)
                    axes[1, 1].plot(mtf_sharp_center[center_y, :], label='Sharp', linewidth=2)
                    axes[1, 1].set_xlabel('Position')
                    axes[1, 1].set_ylabel('MTF')
                    axes[1, 1].set_title('Horizontal Cross-section')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    axes[1, 2].plot(mtf_smooth_center[:, center_x], label='Smooth', linewidth=2)
                    axes[1, 2].plot(mtf_sharp_center[:, center_x], label='Sharp', linewidth=2)
                    axes[1, 2].set_xlabel('Position')
                    axes[1, 2].set_ylabel('MTF')
                    axes[1, 2].set_title('Vertical Cross-section')
                    axes[1, 2].legend()
                    axes[1, 2].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(kernel_dir, f'mtf_epoch{epoch + 1}.png'), dpi=150)
                    plt.close()
                    
                    logger.info(f"  MTF smooth range: [{mtf_smooth.min():.6f}, {mtf_smooth.max():.6f}]")
                    logger.info(f"  MTF sharp range: [{mtf_sharp.min():.6f}, {mtf_sharp.max():.6f}]")
            
            # Train Discriminators
            B_gen = 1
            real_label = torch.ones(B_gen, 1, device=device) * 0.9  # Label smoothing
            fake_label = torch.zeros(B_gen, 1, device=device) + 0.1  # Label smoothing
            
            I_sharp_single = I_sharp[0:1]
            I_smooth_single = I_smooth[0:1]
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == "cuda")):  #type: ignore
                out_real_sharp = D_sharp(I_sharp_single)
                out_fake_sharp = D_sharp(I_gen_sharp.detach())
                loss_D_sharp = 0.5 * (bce(out_real_sharp, real_label) + bce(out_fake_sharp, fake_label))
                
                out_real_smooth = D_smooth(I_smooth_single)
                out_fake_smooth = D_smooth(I_gen_smooth.detach())
                loss_D_smooth = 0.5 * (bce(out_real_smooth, real_label) + bce(out_fake_smooth, fake_label))
                
                loss_D = loss_D_sharp + loss_D_smooth
            
            opt_Ds.zero_grad()
            if scaler:
                scaler.scale(loss_D).backward()
                scaler.unscale_(opt_Ds)
                torch.nn.utils.clip_grad_norm_(
                    list(D_sharp.parameters()) + list(D_smooth.parameters()),
                    max_norm=1.0
                )
                scaler.step(opt_Ds)
            else:
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(D_sharp.parameters()) + list(D_smooth.parameters()),
                    max_norm=1.0
                )
                opt_Ds.step()
            
            D_grad_norm = compute_gradient_norm(D_sharp) + compute_gradient_norm(D_smooth)
            
            # Train Generator once but with stronger weight
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == "cuda")): #type: ignore
                out_fake_sharp = D_sharp(I_gen_sharp)
                out_fake_smooth = D_smooth(I_gen_smooth)
                gan_loss = 0.5 * (bce(out_fake_sharp, real_label) + bce(out_fake_smooth, real_label))
                
                I_sharp_avg = I_sharp.mean(dim=0, keepdim=True)
                I_smooth_avg = I_smooth.mean(dim=0, keepdim=True)
                recon_loss = l1(I_gen_sharp, I_sharp_avg) + l1(I_gen_smooth, I_smooth_avg)
                
                loss_G = 0.1 * recon_loss + 0.2 * gan_loss  # Increase GAN weight instead of training 2x

            opt_G.zero_grad()
            if scaler:
                scaler.scale(loss_G).backward()
                scaler.unscale_(opt_G)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(opt_G)
                scaler.update()
            else:
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                opt_G.step()

            G_grad_norm = compute_gradient_norm(net)
            
            epoch_metrics['G_loss'] += loss_G.item()
            epoch_metrics['D_loss'] += loss_D.item()
            epoch_metrics['recon_loss'] += recon_loss.item()
            epoch_metrics['gan_loss'] += gan_loss.item()
            epoch_metrics['D_sharp_loss'] += loss_D_sharp.item()
            epoch_metrics['D_smooth_loss'] += loss_D_smooth.item()
            epoch_metrics['G_grad_norm'] += G_grad_norm
            epoch_metrics['D_grad_norm'] += D_grad_norm

            pbar.set_postfix({
                'G': f"{loss_G.item():.4f}",
                'D': f"{loss_D.item():.4f}",
                'Recon': f"{recon_loss.item():.4f}"
            })
        
        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_loader)
        
        loss_history['total_G_loss'].append(epoch_metrics['G_loss'])
        loss_history['total_D_loss'].append(epoch_metrics['D_loss'])
        loss_history['recon_loss'].append(epoch_metrics['recon_loss'])
        loss_history['gan_loss'].append(epoch_metrics['gan_loss'])
        loss_history['D_sharp_loss'].append(epoch_metrics['D_sharp_loss'])
        loss_history['D_smooth_loss'].append(epoch_metrics['D_smooth_loss'])
        loss_history['G_grad_norm'].append(epoch_metrics['G_grad_norm'])
        loss_history['D_grad_norm'].append(epoch_metrics['D_grad_norm'])
        
        val_metrics = validate(net, D_sharp, D_smooth, val_loader, device, bce, l1)
        loss_history['val_G_loss'].append(val_metrics['G_loss'])
        loss_history['val_D_loss'].append(val_metrics['D_loss'])
        loss_history['val_recon_loss'].append(val_metrics['recon_loss'])
        loss_history['val_gan_loss'].append(val_metrics['gan_loss'])
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
        logger.info(f"  Train - G: {epoch_metrics['G_loss']:.4f} | D: {epoch_metrics['D_loss']:.4f} | Recon: {epoch_metrics['recon_loss']:.4f} | GAN: {epoch_metrics['gan_loss']:.4f}")
        logger.info(f"  Val   - G: {val_metrics['G_loss']:.4f} | D: {val_metrics['D_loss']:.4f} | Recon: {val_metrics['recon_loss']:.4f} | GAN: {val_metrics['gan_loss']:.4f}")
        logger.info(f"  D_sharp: {epoch_metrics['D_sharp_loss']:.4f} | D_smooth: {epoch_metrics['D_smooth_loss']:.4f}")
        logger.info(f"  Grad_G: {epoch_metrics['G_grad_norm']:.4f} | Grad_D: {epoch_metrics['D_grad_norm']:.4f}")
        logger.info("-"*80)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train G: {epoch_metrics['G_loss']:.4f} | Val G: {val_metrics['G_loss']:.4f} | Val Recon: {val_metrics['recon_loss']:.4f}")
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        save_checkpoint(epoch, net, D_sharp, D_smooth, opt_G, opt_Ds, scaler, loss_history, checkpoint_path)
        logger.info(f"  Checkpoint saved: {checkpoint_path}")
        save_checkpoint(epoch, net, D_sharp, D_smooth, opt_G, opt_Ds, scaler, loss_history, latest_checkpoint)

    logger.info("="*80)
    logger.info("Training Completed Successfully!")
    logger.info("="*80)

    return net, loss_history

if __name__ == "__main__":
    net, loss_history = train()
    plot_final_loss_curves(loss_history, 'training_metrics_final.png')
    torch.save(net.state_dict(), "mtf_estimator_final.pth")
    print("Training completed!")