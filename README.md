# WDS-Net: Multi-Path Feature Disentanglement Framework

WDS-Net is a general-purpose, PyTorch-based Deep Learning framework designed for image classification tasks. It leverages a unique multi-path architecture that simultaneously extracts Spatial (CNN), Structural (LSTM), and Global (Statistical) features from input images to improve predictive performance. 

While originally tailored for Devanagari characters and Hindi-MNIST numerals, the framework is generalized to train on **any image dataset**.

## Key Features
- **Parallel Feature Disentanglement**:
  - **Spatial Path**: Extracts scale-invariant features via Deep CNNs and Max Pooling.
  - **Structural Path**: Captures sequential dependencies (stroke orders/structural relationships) using an LSTM over the Convoluted feature slices.
  - **Global Path**: Statically computes Mean, Variance, and Flattened Intensity Histograms.
- **Unified Fusion Module**: Features are concatenated and processed through specialized FC (Fully Connected) dense blocks.
- **Robust Evaluation Engine**: Automatically computes Accuracy, Precision, Recall, F1-Scores, Multi-class ROC-AUC curves, and generates Confusion Matrix heatmaps.

---

## Getting Started

### 1. System Requirements

Ensure you have Python 3.8 or higher installed on your system. 

**For GPU Acceleration (Strongly Recommended)**:
To utilize your NVIDIA GPU for significantly faster training, you must install the version of PyTorch compiled for CUDA. 

If you currently have the default CPU-only version installed, you must uninstall it first:
```bash
pip uninstall torch torchvision torchaudio
```

Then, install the GPU-enabled version. The following command installs the version built for CUDA 11.8, which is highly compatible with most modern NVIDIA GPUs:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Install Remaining Dependencies**:
Run the following command to install the required data processing and visualization libraries:
```bash
pip install numpy opencv-python scikit-learn matplotlib seaborn tqdm
```

### 2. Detailed Dataset Preparation

WDS-Net is designed to seamlessly integrate with **any image dataset** without requiring manual code modifications. Follow these precise steps to prepare your custom data:

**Step A: Organize Your Directory Structure**
Ensure your dataset is strictly organized into distinct `Train` and `Test` (or Validation) root folders. Inside these roots, create a subfolder for each unique class category. The framework automatically parses these subfolder names to generate the `class_to_idx` mappings.

```text
my_custom_dataset/
├── Train/
│   ├── category_A/
│   │   ├── img_001.jpg
│   │   └── img_002.png
│   ├── category_B/
│   └── category_C/
└── Test/
    ├── category_A/
    ├── category_B/
    └── category_C/
```

**Step B: Understanding Automated Preprocessing**
You do not need to manually resize, crop, or convert your raw images beforehand. The `UniversalDataset` loader handles the entire ingestion pipeline:
1. **Resolution Standardization:** Automatically resizes all varying input images to a uniform `28x28` dimension.
2. **Channel Reduction:** Converts standard RGB images to 1-channel Grayscale structures.
3. **Denoising:** Applies a built-in Gaussian Blur to reduce high-frequency background noise.
4. **Statistical Normalization:** Normalizes pixel intensity distributions using static mean and variance adjustments necessary for the *Global Path*.

**Step C: Handling Multiple Data Sources**
The framework supports combining disjoint datasets dynamically. If your training data comes from multiple locations, simply pass a list of directory paths to the training command. The model will union the classes seamlessly.

### 3. Executing the Training

To begin training, execute the `main.py` script and provide the paths to your training and testing directories using the `--train_dirs` and `--test_dirs` arguments.

**Basic Training Command**:
```bash
python main.py --train_dirs "path/to/Dataset/Train" --test_dirs "path/to/Dataset/Test"
```

**Optional Hyperparameters and Settings**:
You can heavily customize the training process using the following arguments:
- `--epochs`: Number of full training iterations over the dataset (default: `10`).
- `--batch_size`: Number of images processed simultaneously before updating weights (default: `32`).
- `--lr`: Learning rate for the Adam optimizer (default: `0.001`).
- `--device`: Target processing unit (`cuda` or `cpu`). The script will automatically prioritize the GPU if it is correctly installed.

**Checkpointing and Resuming**:
The framework automatically saves your progress after every epoch.
- `--save_path`: Destination to save the final trained `pth` model dictionary.
- `--checkpoint_path`: Destination to continuously save intermediate epoch checkpoints.
- `--resume`: Include this flag to automatically load the latest checkpoint and resume training from the exact epoch it was interrupted.

**Example Resuming Command**:
```bash
python main.py --train_dirs "path/Train" --test_dirs "path/Test" --resume
```

---

## Project Structure

- `main.py`: The CLI entry point orchestrating dataset combination, training execution, and testing initialization.
- `model.py`: Contains the `WDSNet` PyTorch architecture (CNN, LSTM, and Fusion layers).
- `dataset.py`: Houses the `UniversalDataset` class for generalized directory parsing and automated preprocessing.
- `train.py`: The core training loop applying the Categorical Cross-Entropy loss criterion and Adam optimizer.
- `evaluate.py`: The testing execution loop capturing logic for comprehensive multi-class metrics (via Scikit-Learn).
- `utils.py`: Analytical helper formulas for statistical Global Feature extraction, plotting, and checkpoint management.

---

## Outputs and Visualizations

After the training phase concludes, the framework performs a final inference pass on the test set and generates visualization files in your root directory:
1. `training_curves.png`: A dual-axis plot illustrating loss decay alongside validation accuracy growth across all epochs.
2. `confusion_matrix.png`: A Seaborn heatmap detailing true labels versus predicted class overlap, useful for highlighting specific structural similarities between classes.
3. `roc_curves.png`: An Area Under the Curve (AUC) analysis mapping the False Positive Rate versus the True Positive Rate (if explicitly enabled in the `evaluate.py` script).
