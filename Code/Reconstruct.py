import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# Import your custom utils (assuming the file is named utils.py)
from utils import generate_images, radial_average 
from Generator import KernelEstimator

def reconstruct_nifti_2d(nii_path, model, output_path, device='cuda', save_debug_images=True):
    print(f"Loading {nii_path}...")
    img_obj = nib.load(nii_path)
    vol_data = img_obj.get_fdata()
    affine = img_obj.affine
    h, w, d = vol_data.shape
    debug_slice_idx = d // 2
    
    reconstructed_vol = np.zeros_like(vol_data)
    model.eval()
    model.to(device)
    
    # We define a "Target Sharp MTF" as a 2D Gaussian 
    # This matches the 'mtf_sharp' parameter required by your generate_images
    y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
    rho = torch.sqrt(x**2 + y**2).to(device)
    mtf_target_2d = torch.exp(-(rho**2) / (2 * 0.38**2)).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

    with torch.no_grad():
        for i in range(d):
            slice_img = vol_data[:, :, i]
            
            if slice_img.max() == 0:
                reconstructed_vol[:, :, i] = slice_img
                continue
            
            # Prepare input: [B, C, H, W]
            img_t = torch.from_numpy(slice_img).float().to(device).unsqueeze(0).unsqueeze(0)
            
            # Normalize to [0, 1] for the model/utils logic
            orig_min, orig_max = img_t.min(), img_t.max()
            img_norm = (img_t - orig_min) / (orig_max - orig_min + 1e-8)

            # 1. Use model to get the 2D MTF of the blurred slice
            mtf_predicted = model(img_norm) 
            
            # 2. Use your utils.generate_images to perform the frequency domain swap
            # We treat img_norm as I_smooth and a dummy zero tensor for I_sharp 
            # because we only care about the I_gen_sharp output.
            I_gen_sharp, _ = generate_images(
                I_smooth=img_norm, 
                I_sharp=img_norm, # Dummy
                mtf_smooth=mtf_predicted, 
                mtf_sharp=mtf_target_2d
            )

            # 3. Restore intensity range
            restored_np = I_gen_sharp.squeeze().cpu().numpy()
            restored_np = restored_np * (orig_max.item() - orig_min.item()) + orig_min.item()
            
            reconstructed_vol[:, :, i] = restored_np

            # 4. Debugging with radial_average from utils.py
            if i == debug_slice_idx and save_debug_images:
                # Calculate radial profiles of the MTFs
                prof_pred = radial_average(mtf_predicted.squeeze().cpu().numpy())
                prof_target = radial_average(mtf_target_2d.squeeze().cpu().numpy())
                
                fig, axes = plt.subplo, figsize=(18, 5))
                axes[0].imshow(slice_img, cmap='gray')
                axes[0].set_title("Original")
                
                axes[1].imshow(restored_np, cmap='gray')
                axes[1].set_title("Reconstructed (via generate_images)")
                
             
                
                plt.savefig(output_path.replace(".nii", "_debug.png"))
                plt.close()

    new_img = nib.Nifti1Image(reconstructed_vol, affine)
    nib.save(new_img, output_path)
    print(f"Saved to {output_path}")

net = KernelEstimator()
device = 'cuda'
filepath = r'D:\PSFBaseline\best_model.pth'
checkpoint = torch.load(filepath, map_location=device)
net.load_state_dict(checkpoint['net_state_dict'])
nii_path = r'D:\PSFBaseline\Data_Root\trainB\0B14X41758_filter_E.nii'
reconstruct_nifti_2d(nii_path=nii_path,model=net,output_path = 'reconstructed')