# GCN-Node-Classification-Fundamentals

[![GitHub Stars](https://img.shields.io/github/stars/wajason/GCN-Node-Classification-Fundamentals?style=for-the-badge&logo=github&color=6699CC)](https://github.com/wajason/GCN-Node-Classification-Fundamentals/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/wajason/GCN-Node-Classification-Fundamentals?style=for-the-badge&logo=github&color=6699CC)](https://github.com/wajason/GCN-Node-Classification-Fundamentals/network/members)
[![Issues](https://img.shields.io/github/issues/wajason/GCN-Node-Classification-Fundamentals?style=for-the-badge&color=6699CC)](https://github.com/wajason/GCN-Node-Classification-Fundamentals/issues)
[![License](https://img.shields.io/badge/License-MIT-6699CC?style=for-the-badge)](./LICENSE)
[![Run in Colab](https://img.shields.io/badge/Open%20in-Colab-6699CC?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/wajason/GCN-Node-Classification-Fundamentals/blob/main/GNN.ipynb)

---

## 💡 Project Overview: Fundamental Implementation of Graph Convolutional Network (GCN)

This repository provides a **concise, foundational implementation** of the **Graph Convolutional Network (GCN)** architecture for the task of **Node Classification**. The primary goal is to demonstrate the core mechanism of message passing and aggregation within a standard GCN layer using the high-level PyTorch Geometric (PyG) library.

The entire experiment is encapsulated within a single, reproducible Jupyter Notebook (`GNN.ipynb`).

### Experimental Setup and Methodology

| Component | Detail | Notes |
| :--- | :--- | :--- |
| **Model Architecture** | 2-Layer GCN (GCNConv) | Input $\to$ GCN(1433, 16) $\to$ ReLU $\to$ Dropout $\to$ GCN(16, 7) $\to$ LogSoftmax |
| **Task** | Node Classification | Predicting the category of each paper based on its features and connections. |
| **Dataset** | Cora Citation Network | A widely-used benchmark dataset in graph learning. |
| **Optimization** | Adam Optimizer | Learning Rate ($0.01$) with $L_2$ regularization (Weight Decay: $5e-4$). |
| **Loss Function** | Negative Log Likelihood Loss (NLLLoss) | Standard loss function for multi-class classification using log-probabilities. |
| **Training Device** | CUDA (GPU) enabled | Utilizing PyTorch for efficient computation on available GPU resources. |

### 📈 Core Results (Cora Dataset)

The training process tracks the model's convergence and generalization performance over 100 epochs, optimized by tracking the validation accuracy.

| Metric | Train Accuracy (Final) | Best Validation Accuracy | Test Accuracy (at Best Val) |
| :--- | :--- | :--- | :--- |
| **GCN Model** | $1.000$ | $0.784$ | $0.815$ |

The results demonstrate the model's ability to achieve strong performance (Test Accuracy of 81.5%) on the well-known Cora homophilic graph structure.

## 🚀 Reproduction

To run and reproduce this experiment:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/wajason/GCN-Node-Classification-Fundamentals.git](https://github.com/wajason/GCN-Node-Classification-Fundamentals.git)
    cd GCN-Node-Classification-Fundamentals
    ```
2.  **Launch the Notebook:** Click the **`Run in Colab`** badge above.
3.  **Execute Cells:** Run all cells sequentially (ensure the required libraries are installed in the first step).

## 🛠️ Requirements

* Python (3.x)
* PyTorch
* `torch-geometric`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
