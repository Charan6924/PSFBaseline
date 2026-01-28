import re
import matplotlib.pyplot as plt

def parse_and_plot_logs(file_path):
    epochs = []
    # Training metrics
    train_g, train_d, train_recon, train_gan = [], [], [], []
    # Validation metrics
    val_g, val_d, val_recon, val_gan = [], [], [], []
    # Discriminator specific
    d_sharp, d_smooth = [], []

    # Regex patterns for specific lines in your logs
    epoch_pattern = re.compile(r"Epoch \[(\d+)/100\]")
    train_pattern = re.compile(r"Train - G: ([\d.]+) \| D: ([\d.]+) \| Recon: ([\d.]+) \| GAN: ([\d.]+)")
    val_pattern = re.compile(r"Val   - G: ([\d.]+) \| D: ([\d.]+) \| Recon: ([\d.]+) \| GAN: ([\d.]+)")
    d_metrics_pattern = re.compile(r"D_sharp: ([\d.]+) \| D_smooth: ([\d.]+)")

    with open(file_path, 'r') as f:
        for line in f:
            if "Epoch [" in line:
                epochs.append(int(epoch_pattern.search(line).group(1)))
            elif "Train - G:" in line:
                m = train_pattern.search(line)
                train_g.append(float(m.group(1)))
                train_d.append(float(m.group(2)))
                train_recon.append(float(m.group(3)))
                train_gan.append(float(m.group(4)))
            elif "Val   - G:" in line:
                m = val_pattern.search(line)
                val_g.append(float(m.group(1)))
                val_d.append(float(m.group(2)))
                val_recon.append(float(m.group(3)))
                val_gan.append(float(m.group(4)))
            elif "D_sharp:" in line:
                m = d_metrics_pattern.search(line)
                d_sharp.append(float(m.group(1)))
                d_smooth.append(float(m.group(2)))

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Kernel Estimator GAN Training Metrics', fontsize=16)

    # Subplot 1: Generator & Discriminator Total Loss
    axes[0, 0].plot(epochs, train_g, label='Train G Loss', color='blue')
    axes[0, 0].plot(epochs, train_d, label='Train D Loss', color='red')
    axes[0, 0].set_title('Adversarial Balance (G vs D)')
    axes[0, 0].legend()

    # Subplot 2: Reconstruction Loss (Structural Integrity)
    axes[0, 1].plot(epochs, train_recon, label='Train Recon', linestyle='--')
    axes[0, 1].plot(epochs, val_recon, label='Val Recon', linewidth=2)
    axes[0, 1].set_title('Reconstruction Error (L1/MSE)')
    axes[0, 1].legend()

    # Subplot 3: GAN Specific Loss
    axes[1, 0].plot(epochs, train_gan, label='Train GAN Loss', color='green')
    axes[1, 0].plot(epochs, val_gan, label='Val GAN Loss', color='orange')
    axes[1, 0].set_title('GAN Component Loss')
    axes[1, 0].legend()

    # Subplot 4: Discriminator Sharp vs Smooth
    axes[1, 1].plot(epochs, d_sharp, label='D Sharp Score')
    axes[1, 1].plot(epochs, d_smooth, label='D Smooth Score')
    axes[1, 1].set_title('Discriminator Kernel Branch Performance')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Run the parser (ensure your log file is named 'training.log')
parse_and_plot_logs('training_log.txt')