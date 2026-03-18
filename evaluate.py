import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from utils import plot_confusion_matrix, plot_roc_curves

def evaluate_model(model, dataloader, num_classes, class_names=None, device='cpu', save_plots=True):
    """
    Runs the testing phase on unseen data and generates performance metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, global_feats, labels in dataloader:
            inputs = inputs.to(device)
            global_feats = global_feats.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs, global_feats)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Standard Decision Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    if save_plots:
        plot_confusion_matrix(cm, class_names=class_names)
    
    # ROC and AUC Calculation
    labels_bin = label_binarize(all_labels, classes=range(num_classes))
    all_probs = np.array(all_probs)
    
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}
    
    if num_classes > 1 and len(all_labels) > 0:
        for i in range(num_classes):
            if np.sum(labels_bin[:, i]) > 0: # Ensure class instances exist in the set
                fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
                fpr_dict[i] = fpr
                tpr_dict[i] = tpr
                roc_auc_dict[i] = auc(fpr, tpr)
                
        if save_plots and fpr_dict:
            plot_roc_curves(fpr_dict, tpr_dict, roc_auc_dict, num_classes)
            
    mean_auc = np.mean(list(roc_auc_dict.values())) if roc_auc_dict else 0.0

    print("\n--- Final Evaluation Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Mean AUC:  {mean_auc:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_auc': mean_auc,
        'confusion_matrix': cm
    }
    
    return metrics
