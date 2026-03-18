import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import UniversalDataset
from model import WDSNet
from train import train_model
from evaluate import evaluate_model
from utils import plot_training_curves, save_model

def main():
    parser = argparse.ArgumentParser(description="WDS-Net: Multi-Path Feature Disentanglement Framework")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training/evaluation')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam optimizer')
    parser.add_argument('--num_classes', type=int, default=60, help='Number of classes in the dataset')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run on (cuda or cpu)')
    parser.add_argument('--save_path', type=str, default='wds_net_model.pth', help='Path to save the final trained model weights')
    parser.add_argument('--checkpoint_path', type=str, default='wds_net_checkpoint.pth', help='Path to save/load intermediate training checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint if it exists')
    parser.add_argument('--train_dirs', nargs='+', default=[r'd:\python projects\WSDNET+++\charater_data\Train'], help='List of paths to training directories')
    parser.add_argument('--test_dirs', nargs='+', default=[r'd:\python projects\WSDNET+++\charater_data\Test'], help='List of paths to testing directories')
    
    args = parser.parse_args()
    print(f"Running WDS-Net on device: {args.device}")

    # 1. Dataset Initialization
    print("\nInitializing datasets...")
    
    train_dataset = UniversalDataset(args.train_dirs)
    val_dataset = UniversalDataset(args.test_dirs, class_to_idx=train_dataset.class_to_idx)
    
    num_classes = len(train_dataset.class_to_idx)
    print(f"Total Combined Classes Detected: {num_classes}")
    print(f"Training Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 2. Architecture Instantiation
    print("Building model architecture...")
    model = WDSNet(num_classes=num_classes)
    print(model)

    # 3. Training Loop
    print("\nStarting training phase...")
    train_losses = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, # Optional validation during training loop
        epochs=args.epochs, 
        learning_rate=args.lr, 
        device=args.device,
        save_path=args.checkpoint_path,
        resume_path=args.checkpoint_path if args.resume else None
    )
    
    # Plot training loss graph
    plot_training_curves(train_losses, val_accuracies=None)

    # 4. Evaluation Phase
    print("\nRunning final evaluation on validation set...")
    metrics = evaluate_model(
        model=model, 
        dataloader=val_loader, 
        num_classes=num_classes, 
        device=args.device,
        save_plots=True
    )
    
    # Save Model Weights
    save_model(model, path=args.save_path)
    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()
