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
    plt.grid(True, linestyle='--', alpha=0.6) # Enhanced readability

    color = 'tab:red'
    ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', color=color, fontsize=12, fontweight='bold')
    ax1.plot(epochs, train_losses, color=color, marker='o', markersize=6, linewidth=2, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    
    # Handle case where validation wasn't done
    if val_accuracies:
        ax2.set_ylabel('Validation Accuracy', color=color, fontsize=12, fontweight='bold')
        ax2.plot(epochs, val_accuracies, color=color, marker='s', markersize=6, linewidth=2, label='Val Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("Training Loss and Validation Accuracy", fontsize=15, fontweight='bold', pad=15)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(cm, class_names=None, save_path="confusion_matrix.png"):
    """
    Plots the normalized confusion matrix with optional class names and integer annotations.
    """
    # Scale dimension to number of classes dynamically (at least 10x8)
    fig_size = max(10, min(24, int(len(cm) * 0.6)))
    plt.figure(figsize=(fig_size, fig_size * 0.75))
    
    sns.set_context("paper", font_scale=1.2)
    
    # Normalize by row for color grading
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Render Heatmap utilizing `annot=cm` (raw counts) on top of the normalized color gradients
    sns.heatmap(cm_normalized, 
                annot=cm, 
                fmt="d", # Raw integers
                cmap="Blues", 
                linewidths=0.05, 
                linecolor='gray',
                xticklabels=class_names if class_names is not None else "auto",
                yticklabels=class_names if class_names is not None else "auto",
                annot_kws={"size": 10},
                cbar_kws={'label': 'Proportion of Predictions'})
                
    plt.title('Confusion Matrix', fontsize=25, fontweight='bold', pad=20)
    plt.ylabel('Actual', fontsize=18, fontweight='bold')
    plt.xlabel('Prediction', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
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

def save_text_as_image(text, filename, figsize=(10, 8), fontsize=12):
    """Saves a raw text string as a simple graphical PNG image with a white background."""
    plt.figure(figsize=figsize, facecolor='white')
    plt.text(0.01, 0.98, text, fontsize=fontsize, color='black', fontfamily='monospace', 
             verticalalignment='top', horizontalalignment='left')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def load_model(model, path="wds_net_model.pth", device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")
    return model

def plot_pr_curves(prec_dict, rec_dict, pr_auc_dict, num_classes, save_path="pr_curves.png"):
    """Plots Precision-Recall curves with publication-quality styling and a top-right zoom inset."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            pass 

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Setup inset zoom bounding box [x, y, width, height] for the perfect top-right corner
    axins = ax.inset_axes([0.15, 0.15, 0.45, 0.45])
    
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0, 0.9, num_classes)] 

    for i in range(num_classes):
        if i in prec_dict and i in rec_dict:
            # PR Curve plots Recall on X and Precision on Y
            ax.plot(rec_dict[i], prec_dict[i], color=colors[i], lw=1.5, alpha=0.7, 
                     label=f'Class {i} (AP = {pr_auc_dict[i]:.4f})')
            axins.plot(rec_dict[i], prec_dict[i], color=colors[i], lw=1.5, alpha=0.7)

    # Main plot configurations
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve (PR)', fontsize=15, fontweight='bold', pad=15)

    # Inset configurations
    axins.set_xlim(0.9, 1.0) # Zoom into [0.9, 1.0] for Recall
    axins.set_ylim(0.9, 1.05) # Zoom into [0.9, 1.0] for Precision
    axins.tick_params(labelsize=9)
    ax.indicate_inset_zoom(axins, edgecolor="black")

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    plt.close()
    plt.style.use('default')

def plot_class_f1_scores(class_f1, class_names=None, save_path="class_f1_scores.png"):
    """Plots a horizontal bar chart of F1 scores by class, sorted from worst to best."""
    if class_names is None:
        class_names = [str(i) for i in range(len(class_f1))]
        
    # Sort by F1
    sorted_indices = np.argsort(class_f1)
    sorted_f1 = [class_f1[i] for i in sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]
    
    # Scale height dynamically based on number of classes
    plt.figure(figsize=(12, max(8, len(class_names) * 0.35))) 
    sns.set_context("paper", font_scale=1.2)
    
    # Use RdYlGn to highlight low (red) vs high (green) performance
    palette = sns.color_palette("RdYlGn", len(class_names))
    sns.barplot(x=sorted_f1, y=sorted_names, palette=palette, hue=sorted_names, legend=False)
    
    # Add a mean line
    mean_f1 = sum(class_f1) / len(class_f1)
    plt.axvline(x=mean_f1, color='gray', linestyle='--', linewidth=2, label=f'Mean F1 ({mean_f1:.3f})')
    
    plt.title('Class-Wise F1-Scores (Ranked)', fontsize=18, pad=15)
    plt.xlabel('F1-Score', fontsize=14)
    plt.ylabel('Character Class (Worst to Best)', fontsize=14)
    plt.xlim(0, 1.05)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    sns.reset_orig()

def plot_error_gallery(images, true_labels, pred_labels, class_names=None, save_path="error_gallery.png"):
    """Plots a grid of misclassified images as a Gallery of Mistakes."""
    n = len(images)
    cols = 5
    rows = int(np.ceil(n / cols))
    if rows == 0: return 
    
    plt.figure(figsize=(3 * cols, 3.5 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        img = images[i]
        
        # Generalized handling for both 1-channel (Grayscale) and 3-channel (RGB) images
        if img.shape[0] == 1:
            img = np.squeeze(img, axis=0)
            cmap = 'gray'
        elif img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
            cmap = None
        else:
            img = np.squeeze(img)
            cmap = 'gray'
        
        # Max-min un-normalization to ensure visibility since CNN standard deviations skew colors
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        
        if cmap:
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(img)
            
        t_label = class_names[true_labels[i]] if class_names else true_labels[i]
        p_label = class_names[pred_labels[i]] if class_names else pred_labels[i]
        
        # Color the label red
        plt.title(f"True: {t_label}\nPred: {p_label}", color='darkred', fontsize=11, fontweight='bold')
        plt.axis('off')
        
    plt.suptitle("Gallery of Mistakes (Misclassifications)", fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
