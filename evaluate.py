import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from utils import plot_confusion_matrix, plot_roc_curves, plot_pr_curves, plot_class_f1_scores, plot_error_gallery

def evaluate_model(model, dataloader, num_classes, class_names=None, device='cpu', save_plots=True):
    """
    Runs the testing phase on unseen data and generates performance metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Track errors for the gallery
    errors_images = []
    errors_true = []
    errors_pred = []
    
    with torch.no_grad():
        for inputs, global_feats, labels in dataloader:
            inputs = inputs.to(device)
            global_feats = global_feats.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs, global_feats)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Find misclassifications for Error Gallery
            mismatches = (preds != labels)
            if mismatches.any() and len(errors_images) < 25:
                mis_inputs = inputs[mismatches].cpu().numpy()
                mis_labels = labels[mismatches].cpu().numpy()
                mis_preds = preds[mismatches].cpu().numpy()
                for i in range(len(mis_inputs)):
                    if len(errors_images) < 25:
                        errors_images.append(mis_inputs[i])
                        errors_true.append(mis_labels[i])
                        errors_pred.append(mis_preds[i])
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Standard Decision Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    # Class-wise Metrics
    class_prec, class_rec, class_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion Matrix & F1 Bar Chart & Error Gallery
    cm = confusion_matrix(all_labels, all_preds)
    if save_plots:
        plot_confusion_matrix(cm, class_names=class_names)
        plot_class_f1_scores(class_f1, class_names=class_names if class_names else list(range(num_classes)))
        if len(errors_images) > 0:
            plot_error_gallery(errors_images, errors_true, errors_pred, class_names=class_names)
    
    # ROC, PR, and AUC Calculation
    labels_bin = label_binarize(all_labels, classes=range(num_classes))
    all_probs = np.array(all_probs)
    
    fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
    prec_dict, rec_dict, pr_auc_dict = {}, {}, {}
    
    if num_classes > 1 and len(all_labels) > 0:
        for i in range(num_classes):
            if np.sum(labels_bin[:, i]) > 0: # Ensure class instances exist in the set
                # ROC
                fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
                fpr_dict[i], tpr_dict[i] = fpr, tpr
                roc_auc_dict[i] = auc(fpr, tpr)
                
                # PR
                prec, rec, _ = precision_recall_curve(labels_bin[:, i], all_probs[:, i])
                prec_dict[i], rec_dict[i] = prec, rec
                pr_auc_dict[i] = average_precision_score(labels_bin[:, i], all_probs[:, i])
                
        if save_plots and fpr_dict:
            plot_roc_curves(fpr_dict, tpr_dict, roc_auc_dict, num_classes)
            plot_pr_curves(prec_dict, rec_dict, pr_auc_dict, num_classes)
            
    mean_auc = np.mean(list(roc_auc_dict.values())) if roc_auc_dict else 0.0
    mean_pr_auc = np.mean(list(pr_auc_dict.values())) if pr_auc_dict else 0.0

    print("\n--- Final Evaluation Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Mean ROC AUC:  {mean_auc:.4f}")
    print(f"Mean PR AUC:   {mean_pr_auc:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_roc_auc': mean_auc,
        'mean_pr_auc': mean_pr_auc,
        'confusion_matrix': cm
    }
    
    return metrics
