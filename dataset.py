import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from utils import extract_global_features

class UniversalDataset(Dataset):
    def __init__(self, data_dirs, class_to_idx=None):
        """
        Args:
            data_dirs: List of paths to dataset directories (Train or Test root)
            class_to_idx: Optional pre-computed mapping of class names to integer indices 
                          (vital for validation/test sets to match training set indices).
        """
        self.image_paths = []
        self.labels = []
        
        # If no mapping is provided, we establish a new one.
        # If provided, we freeze it and only accept known classes.
        self.class_to_idx = class_to_idx if class_to_idx is not None else {}
        is_mapping_frozen = class_to_idx is not None
        
        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                print(f"Warning: Directory {data_dir} not found. Skipping.")
                continue
                
            classes = sorted(os.listdir(data_dir))
            for class_name in classes:
                class_dir = os.path.join(data_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                
                # Assign universal class indices dynamically if not frozen
                if class_name not in self.class_to_idx:
                    if is_mapping_frozen:
                        # Skip this class because the model was entirely untrained on it
                        print(f"Warning: Class '{class_name}' not found in training set. Skipping.")
                        continue
                        
                    self.class_to_idx[class_name] = len(self.class_to_idx)
                    
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            # Just in case image is corrupted, fallback to zero image
            img = np.zeros((28, 28), dtype=np.uint8)
        
        # Resizing and Denoising
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28))
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Normalization: x_i' = (x_i - mu) / sigma
        mu = np.mean(img)
        sigma = np.std(img) + 1e-8
        img_norm = (img - mu) / sigma
        
        # Global Feature Extraction using utils
        global_features = extract_global_features(img_norm)
        
        # Tensor conversion
        img_tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0) # (1, 28, 28) adding channel dim
        global_tensor = torch.tensor(global_features, dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img_tensor, global_tensor, label_tensor
