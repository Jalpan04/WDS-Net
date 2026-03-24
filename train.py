import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import save_checkpoint, load_checkpoint

def train_model(model, train_loader, val_loader=None, epochs=10, learning_rate=0.001, device='cpu', 
                save_path="wds_net_checkpoint.pth", resume_path=None):
    """
    Iterates through dataset mini-batches, performs forward passes, calculates loss, 
    and updates model weights. Can resume from a saved checkpoint.
    """
    # Ensure model is on the correct device BEFORE initializing optimizer and loading checkpoints
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    start_epoch = 0
    
    if resume_path and os.path.exists(resume_path):
        model, optimizer, last_epoch = load_checkpoint(model, optimizer, path=resume_path, device=device)
        start_epoch = last_epoch + 1
        print(f"Resuming training from epoch {start_epoch+1}/{epochs}")
    
    train_losses = []
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, global_feats, labels in pbar:
            inputs = inputs.to(device)
            global_feats = global_feats.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, global_feats)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'Loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] Completed - Avg Training Loss: {epoch_loss:.4f}")
        
        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, epoch, path=save_path)
        
    return train_losses
