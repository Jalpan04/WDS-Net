import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def extract_global_features(img_norm):
    """
    Extracts global statistical descriptors (Mean, Variance, Intensity Histogram)
    from a normalized image.
    """
    mu = np.mean(img_norm)
    sigma = np.std(img_norm) + 1e-8
    
    # Range typically between -3 and 3 for normalized images
    hist, _ = np.histogram(img_norm, bins=16, range=(-3, 3), density=True)
    global_features = np.array([mu, sigma**2] + hist.tolist(), dtype=np.float32)
    return global_features

def plot_training_curves(train_losses, val_accuracies, save_path="training_curves.png"):
    """
    Plots training loss and validation accuracy curves.
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    
    # Handle case where validation wasn't done
    if val_accuracies:
        ax2.set_ylabel('Validation Accuracy', color=color)
        ax2.plot(epochs, val_accuracies, color=color, marker='s', label='Val Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("Training Loss and Validation Accuracy")
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, class_names=None, save_path="confusion_matrix.png"):
    """
    Plots the normalized confusion matrix with optional class names.
    """
    plt.figure(figsize=(24, 18))
    
    # Increase base font size for the large canvas
    sns.set_context("paper", font_scale=1.2)
    
    # Normalize by row (true labels) so values are percentages
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Render Heatmap
    sns.heatmap(cm_normalized, 
                annot=False, 
                cmap="GnBu", 
                linewidths=0.05, 
                linecolor='gray',
                xticklabels=class_names if class_names is not None else "auto",
                yticklabels=class_names if class_names is not None else "auto",
                cbar_kws={'label': 'Proportion of Predictions'})
                
    plt.title('Normalized Confusion Matrix (WDS-Net)', fontsize=25, pad=20)
    plt.ylabel('Actual Character', fontsize=18)
    plt.xlabel('Predicted Character', fontsize=18)
    
    # Fix cutting off labels
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Reset context
    sns.reset_orig()

def plot_roc_curves(fpr_dict, tpr_dict, roc_auc_dict, num_classes, save_path="roc_curves.png"):
    """
    Plots ROC curves for multi-class classification with publication-quality styling.
    """
    # 1. Use a modern, light grid style if available
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            pass # Fallback to default styling

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 2. Setup inset zoom bounding box [x, y, width, height]
    axins = ax.inset_axes([0.4, 0.2, 0.45, 0.45])
    
    # 3. Setup sequential colormap for high-class scaling
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0, 0.9, num_classes)] # 0.9 prevents getting too close to invisible yellow

    # Plot all classes 
    for i in range(num_classes):
        if i in fpr_dict and i in tpr_dict:
            # Main plot curve
            ax.plot(fpr_dict[i], tpr_dict[i], color=colors[i], lw=1.5, alpha=0.7, 
                     label=f'Class {i} (AUC = {roc_auc_dict[i]:.4f})')
            # Inset plot curve
            axins.plot(fpr_dict[i], tpr_dict[i], color=colors[i], lw=1.5, alpha=0.7)

    # Diagonal "Chance" Line
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', alpha=0.5)
    axins.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', alpha=0.5)

    # Main plot configurations
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=15, fontweight='bold', pad=15)

    # Inset configurations
    axins.set_xlim(0.0, 0.1) # Zoom into [0, 0.1] for FPR
    axins.set_ylim(0.9, 1.05) # Zoom into [0.9, 1.0] for TPR
    axins.tick_params(labelsize=9)
    ax.indicate_inset_zoom(axins, edgecolor="black")

    # Legend configurations (Brought outside the main box)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10)

    # Final rendering
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight') # High DPI for publication
    plt.close()
    
    # Reset style to default to avoid bleeding into other plots globally
    plt.style.use('default')

def save_checkpoint(model, optimizer, epoch, path="wds_net_checkpoint.pth"):
    """Saves the model state, optimizer state, and current epoch."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path} at epoch {epoch}")

def load_checkpoint(model, optimizer, path="wds_net_checkpoint.pth", device='cpu'):
    """Loads model and optimizer states to resume training."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {path}. Resuming from epoch {epoch}")
    return model, optimizer, epoch

def save_model(model, path="wds_net_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="wds_net_model.pth", device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")
    return model
