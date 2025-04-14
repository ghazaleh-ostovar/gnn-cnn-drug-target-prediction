# gnn-cnn-drug-target-prediction
## 🔹 HybridGNN-CNN: Predicting Drug–Target Activation & Inhibition

This project implements a deep learning pipeline to classify drug–target interactions as activating or inhibiting. It combines structural and sequence-level biological information using:

- Graph Neural Networks (GNNs) on drug molecular graphs (from SMILES)
- Convolutional Neural Networks (CNNs) on protein embeddings (from ESM2)
- A fusion layer for binary classification

The model is lightweight, interpretable, and modular—ideal for adapting to related bioinformatics tasks.

## 🔹 Dataset

The raw dataset comes from:

[🔹 Zhang et al. (2023) — *DrugAI: a multi-view deep learning model for predicting drug–target activating/inhibiting mechanisms*](https://doi.org/10.1093/bib/bbac526)


🔹 **Class Distribution & Imbalance Handling**

The dataset contains **11,229** drug–target interaction pairs, labeled as:

- **Inhibition (0): 8,009 samples (~71%)**
- **Activation (1): 3,220 samples (~29%)**

To address this imbalance, we used the `pos_weight` parameter in the binary cross-entropy loss function (`BCEWithLogitsLoss`). The weight was calculated as the ratio of negative to positive samples (~2.33), helping the model pay more attention to the minority class (activations) during training.

- Raw and cleaned versions are located under the `Dataset/` directory.
- Cleaned data was split into training and test sets and saved to:
  - CSVs: `train.csv`, `test.csv`
  - PyTorch tensors: `train_graphs.pt`, `train_tokens.pt`, etc.
  - All saved under `Train_Test_Data/`

## 🔹 Model Architecture

| Component        | Description                                                             |
|------------------|-------------------------------------------------------------------------|
| Drug Encoder     | 2-layer GCNConv with ReLU, dropout, and global mean pooling             |
| Protein Encoder  | ESM2 (frozen) followed by a 2-layer 1D CNN and adaptive max pooling     |
| Fusion           | Concatenation → dropout → fully connected → sigmoid output              |

## 🔹 Final Evaluation Metrics (Epoch 35, Threshold = 0.69)

### Per-Class Metrics

| Class           | Precision | Recall | F1-score | Support |
|-----------------|-----------|--------|----------|---------|
| 0 (Inhibition)  | 0.8824    | 0.8937 | 0.8880   | 1561    |
| 1 (Activation)  | 0.7352    | 0.7125 | 0.7237   | 647     |

### Overall Performance

| Metric              | Value     |
|---------------------|-----------|
| Accuracy            | 84.06%    |
| Macro Precision     | 80.88%    |
| Macro Recall        | 80.31%    |
| Macro F1-score      | 80.58%    |
| Weighted F1-score   | 83.98%    |
| AUC-ROC             | 0.906     |

Threshold tuning was applied to maximize weighted F1-score.

## 🔹 Repository Contents

```
├── GNN_CNN_Preprocessing.ipynb       # Data cleaning, graph/token generation
├── GNN_CNN_Training.ipynb            # Training and evaluation pipeline
├── Dataset/
│   ├── Activate_Inhibit_Raw.csv
│   ├── Activate_Inhibit_Cleaned.csv
├── Train_Test_Data/
│   ├── train.csv / test.csv
│   ├── train_graphs.pt / test_graphs.pt
│   ├── train_tokens.pt / test_tokens.pt
├── requirements.txt
├── training_metrics_from_logs.csv
```

## 🔹 How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks in order:
   - `GNN_CNN_Preprocessing.ipynb`
   - `GNN_CNN_Training.ipynb`

Make sure the paths are correct; the default is `./Train_Test_Data`

## 🔹 Notes

- ESM2 model (facebook/esm2_t6_8M_UR50D) was frozen during training
- Class imbalance handled using `pos_weight` in `BCEWithLogitsLoss`
- Training metrics were logged and saved to CSV
- Plots were generated to track training performance

## 🔹 Contact

**Ghazaleh Ostovar**  
PhD in Physics | Machine Learning for Biology  
[LinkedIn](https://www.linkedin.com/in/ghazaleh-ostovar) • [GitHub](https://github.com/ghazaleh-ostovar)

