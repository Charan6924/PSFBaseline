import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PSDDataset import PSDDataset
from Generator import KernelEstimator
from Code.Discriminator import Discriminator
from utils import generate_images, normalize, save_checkpoint, load_checkpoint, plot_final_loss_curves, compute_gradient_norm, update_moving_average
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from collections import defaultdict

save_dir = "generated_images_spline"
kernel_dir = "generated_mtfs"
checkpoint_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(kernel_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
net = KernelEstimator().to(device)
D_sharp = Discriminator().to(device)
D_smooth = Discriminator().to(device)
bce = nn.BCELoss().to(device)
l1 = nn.L1Loss().to(device)
opt_G = torch.optim.Adam(net.parameters(), lr=1e-4)
opt_Ds = torch.optim.Adam(
    list(D_sharp.parameters()) + list(D_smooth.parameters()), lr=1e-4
)
scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None #type: ignore
dataset = PSDDataset(root_dir=r"D:\Charan work file\KernelEstimator\Data_Root")
loader = DataLoader(dataset, batch_size=1, shuffle=True)
start_epoch = 0
num_epochs = 40
save_interval = 1
moving_avg_window = 10
latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
start_epoch, loss_history = load_checkpoint(latest_checkpoint, net, D_sharp, D_smooth, opt_G, opt_Ds, scaler)

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
    
    for idx, (I_smooth, I_sharp) in enumerate(loader):
        I_smooth = I_smooth.to(device)
        I_sharp = I_sharp.to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == "cuda")): #type: ignore
            mtf_smooth = net(I_smooth)
            mtf_sharp = net(I_sharp)
            I_gen_sharp, I_gen_smooth = generate_images(I_smooth, I_sharp, mtf_smooth, mtf_sharp)
            
            if idx == 0:
                with torch.no_grad():
                    n_samples = min(4, I_sharp.size(0))
                    for i in range(n_samples):
                        imgs = torch.stack([
                            normalize(I_smooth[i]),
                            normalize(I_gen_smooth[i]),
                            normalize(I_sharp[i]),
                            normalize(I_gen_sharp[i]),
                        ])
                        save_path = os.path.join(save_dir, f"epoch{epoch + 1}_sample{i + 1}.png")
                        save_image(imgs, save_path, nrow=2)
                    
                    curve_smooth_np = mtf_smooth[0, 0].cpu().numpy()
                    curve_sharp_np = mtf_sharp[0, 0].cpu().numpy()
                    num_points = len(curve_smooth_np)
                    freq_axis = np.linspace(0, 0.5, num_points)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(freq_axis, curve_smooth_np, label='Predicted Smooth MTF', color='blue', linewidth=2)
                    plt.plot(freq_axis, curve_sharp_np, label='Predicted Sharp MTF', color='red', linewidth=2)
                    plt.title(f'B-Spline at - Epoch {epoch + 1}')
                    plt.xlabel('Frequency')
                    plt.ylabel('MTF')
                    plt.xlim(0, 0.5)
                    plt.ylim(0, 1.05)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(kernel_dir, f'spline_epoch{epoch + 1}.png'))
                    plt.close()
            
            B = I_gen_sharp.size(0)
            real_label = torch.ones(B, 1, device=device)
            fake_label = torch.zeros(B, 1, device=device)
            
            out_real_sharp = D_sharp(I_sharp)
            out_fake_sharp = D_sharp(I_gen_sharp.detach())
            loss_D_sharp = 0.5 * (bce(out_real_sharp, real_label) + bce(out_fake_sharp, fake_label))
            
            out_real_smooth = D_smooth(I_smooth)
            out_fake_smooth = D_smooth(I_gen_smooth.detach())
            loss_D_smooth = 0.5 * (bce(out_real_smooth, real_label) + bce(out_fake_smooth, fake_label))
            
            loss_D = loss_D_sharp + loss_D_smooth
        
        opt_Ds.zero_grad()
        if scaler:
            scaler.scale(loss_D).backward()
            scaler.step(opt_Ds)
        else:
            loss_D.backward()
            opt_Ds.step()
        
        D_grad_norm = compute_gradient_norm(D_sharp) + compute_gradient_norm(D_smooth)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == "cuda")): #type: ignore
            out_fake_sharp = D_sharp(I_gen_sharp)
            out_fake_smooth = D_smooth(I_gen_smooth)
            
            gan_loss = 0.5 * (bce(out_fake_sharp, real_label) + bce(out_fake_smooth, real_label))
            recon_loss = l1(I_gen_sharp, I_sharp) + l1(I_gen_smooth, I_smooth)
            loss_G = recon_loss + 0.001 * gan_loss

        opt_G.zero_grad()
        if scaler:
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
        else:
            loss_G.backward()
            opt_G.step()
        
        G_grad_norm = compute_gradient_norm(net)
        
        if scaler:
            scaler.update()
        
        epoch_metrics['G_loss'] += loss_G.item()
        epoch_metrics['D_loss'] += loss_D.item()
        epoch_metrics['recon_loss'] += recon_loss.item()
        epoch_metrics['gan_loss'] += gan_loss.item()
        epoch_metrics['D_sharp_loss'] += loss_D_sharp.item()
        epoch_metrics['D_smooth_loss'] += loss_D_smooth.item()
        epoch_metrics['G_grad_norm'] += G_grad_norm
        epoch_metrics['D_grad_norm'] += D_grad_norm
    
    for key in epoch_metrics:
        epoch_metrics[key] /= len(loader)
    
    loss_history['total_G_loss'].append(epoch_metrics['G_loss'])
    loss_history['total_D_loss'].append(epoch_metrics['D_loss'])
    loss_history['recon_loss'].append(epoch_metrics['recon_loss'])
    loss_history['gan_loss'].append(epoch_metrics['gan_loss'])
    loss_history['D_sharp_loss'].append(epoch_metrics['D_sharp_loss'])
    loss_history['D_smooth_loss'].append(epoch_metrics['D_smooth_loss'])
    loss_history['G_grad_norm'].append(epoch_metrics['G_grad_norm'])
    loss_history['D_grad_norm'].append(epoch_metrics['D_grad_norm'])
    loss_history['total_G_loss_ma'].append(update_moving_average(loss_history['total_G_loss'], moving_avg_window))
    loss_history['total_D_loss_ma'].append(update_moving_average(loss_history['total_D_loss'], moving_avg_window))
    loss_history['recon_loss_ma'].append(update_moving_average(loss_history['recon_loss'], moving_avg_window))
    loss_history['gan_loss_ma'].append(update_moving_average(loss_history['gan_loss'], moving_avg_window))
    loss_history['D_sharp_loss_ma'].append(update_moving_average(loss_history['D_sharp_loss'], moving_avg_window))
    loss_history['D_smooth_loss_ma'].append(update_moving_average(loss_history['D_smooth_loss'], moving_avg_window))
    loss_history['G_grad_norm_ma'].append(update_moving_average(loss_history['G_grad_norm'], moving_avg_window))
    loss_history['D_grad_norm_ma'].append(update_moving_average(loss_history['D_grad_norm'], moving_avg_window))
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  G_Loss: {epoch_metrics['G_loss']:.4f} | D_Loss: {epoch_metrics['D_loss']:.4f}")
    print(f"  Recon: {epoch_metrics['recon_loss']:.4f} | GAN: {epoch_metrics['gan_loss']:.4f}")
    print(f"  D_sharp: {epoch_metrics['D_sharp_loss']:.4f} | D_smooth: {epoch_metrics['D_smooth_loss']:.4f}")
    print(f"  Grad_G: {epoch_metrics['G_grad_norm']:.4f} | Grad_D: {epoch_metrics['D_grad_norm']:.4f}")
    
    if (epoch + 1) % save_interval == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        save_checkpoint(epoch, net, D_sharp, D_smooth, opt_G, opt_Ds, scaler, loss_history, checkpoint_path)
    
    save_checkpoint(epoch, net, D_sharp, D_smooth, opt_G, opt_Ds, scaler, loss_history, latest_checkpoint)

plot_final_loss_curves(loss_history, 'training_metrics_final.png')

torch.save(net.state_dict(), "mtf_estimator_final.pth")
print("Training completed!")