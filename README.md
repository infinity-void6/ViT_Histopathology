
# 🧠 Vision Models for Healthcare

This Colab-based project demonstrates a full machine learning pipeline for **histopathology image classification** using PyTorch. It uses **custom HDF5 datasets**, a **Multi-Layer Perceptron (MLP)** model, and **self-supervised feature extraction** via **DINO (Self-Distillation with No Labels)** for feature embeddings. Visualizations are done using **t-SNE** to project high-dimensional embeddings into 2D space.

---

## 📁 Dataset Format

This project works with **custom image patches** stored in `.h5` (HDF5) format. These are pre-split into training, validation, and test sets:

```
/mlss24/data/
  ├── train_split_data.h5
  ├── val_split_data.h5
  ├── test_split_data.h5
  ├── train_split_labels.h5
  ├── val_split_labels.h5
  └── test_split_labels.h5
```

Each `.h5` file contains a single dataset — either a NumPy array of image patches or label vectors.

---

## 🔍 Code Components Explained

### 1. 📦 Data Loading

```python
def read_dataset(file_path: str):
    with h5py.File(file_path, 'r') as h5file:
        data = h5file[list(h5file.keys())[0]]
        return data[:]
```

- Custom datasets are loaded using `h5py`.
- Data is visualized using `matplotlib`.

```python
def visualize_images(data, seed):
    ...
```

### 2. 🧠 Model: Multi-Layer Perceptron (MLP)

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        ...
```

- A **simple feed-forward neural network**.
- Trained on **DINO embeddings** (128-dim vectors from a pretrained ViT).

### 3. 🦾 Feature Extraction via DINO

- Pretrained ViT models are used to extract features using DINO from Facebook AI.
- Code uses the `torchvision.models.vit_b_16` architecture.
- Extracted features are cached into `.npy` files to speed up repeated runs.

```python
def extract_features(...): 
    ...
```

### 4. 🏋️ Model Training

```python
def train(model, dataloader, criterion, optimizer):
    ...
```

- Optimizer: `Adam`
- Loss Function: `CrossEntropyLoss`
- Accuracy is logged per epoch.

### 5. 🎨 Visualization with t-SNE

```python
from sklearn.manifold import TSNE
```

- DINO features are reduced to 2D using t-SNE for visualization.
- Plots are colored by class label.

---

## 📊 Pipeline Summary

| Step            | Description                                           |
|-----------------|-------------------------------------------------------|
| 📥 Load Data    | HDF5 files loaded using `h5py`                        |
| 🧬 Feature Extractor | ViT + DINO (pretrained) embeddings                |
| 🔗 MLP Classifier | Classifies embeddings into tumor/normal             |
| 🏋️ Training     | Supervised learning using cross-entropy loss         |
| 📈 t-SNE        | Visualize embeddings in 2D space                      |

---
