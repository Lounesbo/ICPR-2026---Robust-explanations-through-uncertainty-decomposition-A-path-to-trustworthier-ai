# Uncertainty vs. Robust Explanations (XAI-UQ)

Source code for the study **"Robust Explanations Through Uncertainty Decomposition: A Path to Trustworthier AI"** (accepted at ICPR, see the [note](#️-note) below).

The project studies how **prediction uncertainty** relates to the **robustness of explanations**, for both **SHAP** (feature importance) and **counterfactual** explanations.

Maintained by **Louenas Bounia**.

## 🧠 Introduction

Recent advancements in machine learning have emphasized the need for transparency in model predictions, particularly as interpretability diminishes with increasingly complex architectures. In this work, we propose leveraging **prediction uncertainty** as a complementary approach to classical explainability methods. Specifically, we distinguish between **aleatoric** (data-related) and **epistemic** (model-related) uncertainty to guide the selection of appropriate explanations.

- **Epistemic uncertainty** acts as a rejection criterion for unreliable explanations and provides insight into undertraining.
- **Aleatoric uncertainty** informs the choice between feature-importance and counterfactual explanations.

This framework fosters **uncertainty-aware explainability**, enabling more robust and meaningful interpretations. Our experiments validate the utility of this approach in both classical and deep learning contexts, demonstrating significant gains in the robustness and relevance of explanations.

## 📁 Project structure

```
XAI-UQ/
├── main.py                     # Command-line entry point
├── requirements.txt
├── xai_uq/                     # Python package
│   ├── data.py                 # Dataset loading and standardization
│   ├── uncertainties.py        # Aleatoric / epistemic uncertainty estimators
│   ├── shap_xp.py              # SHAP explanations and their (un-)robustness
│   ├── cf_xp.py                # Counterfactual explanations and their dissimilarity
│   ├── paths.py                # Project paths (data, figures, output)
│   └── evidential/             # Evidential models (Dempster-Shafer theory)
│       ├── deep_eknn.py        # Evidential k-NN (EK-NN)
│       ├── ibelief.py          # Belief-function utilities
│       ├── utils.py
│       └── entropy_uncertainty.py  # Entropy-based uncertainty (random forest)
├── data/                       # UCI datasets
└── figures/
    └── example/                # Example figures shown below
```

## ⚙️ Set-up

Python 3.10+ is recommended. In a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Usage

Run on a single dataset (name is case-insensitive):

```bash
python main.py --dataset LIVER
```

Run on **all** datasets (useful for benchmarking):

```bash
python main.py --all
```

Choose the uncertainty estimator(s) (default: `eknn`, the one used in the paper):

```bash
python main.py --dataset GLASS --uncertainty eknn entropy
```

| Option | Values |
|---|---|
| `--dataset` | `BREAST_CANCER`, `ECOLI`, `GLASS`, `HEART`, `IONOSPHERE`, `IRIS`, `LIVER`, `PARKINSON`, `SONAR`, `WINE` |
| `--uncertainty` | `eknn` (evidential k-NN), `entropy` (random forest), `density` (binary datasets only), `centroids` |

The script prints, for each dataset, the correlation (and p-value) between uncertainty and:
- the **un-robustness** of SHAP explanations (Pearson);
- the **dissimilarity** of counterfactual explanations (Spearman).

`IRIS`, `WINE` and `BREAST_CANCER` are loaded from scikit-learn; the others are read from `data/`.

## 📌 Example (LIVER dataset)

```bash
python main.py --dataset LIVER
```

Expected output (eknn):

```
---SHAP robustness---
Overall correlation coefficient (LIVER_eknn): 0.4966
---Counterfactual robustness---
Average correlation coefficient (LIVER_eknn): -0.8195
```

### Counterfactual explanations dissimilarity
<img src="figures/example/cf_liver_eknn.png" width="500" alt="Counterfactual"/>

### SHAP explanations un-robustness
<img src="figures/example/shap_liver_eknn.png" width="500" alt="SHAP"/>

## 📚 References

- A. Hoarau, B. Quost, S. Destercke, W. Waegeman, *Reducing aleatoric and epistemic uncertainty through multi-modal data acquisition*, 2025. [arXiv:2501.18268](https://arxiv.org/abs/2501.18268)
- A. Hoarau, V. Lemaire, Y. Le Gall, J.-C. Dubois, A. Martin, *Evidential uncertainty sampling strategies for active learning*, Machine Learning, 2024.
- T. Denœux, *A k-nearest neighbor classification rule based on Dempster-Shafer theory*, IEEE Transactions on Systems, Man, and Cybernetics, 1995.
- T. Denœux, *A neural network classifier based on Dempster-Shafer theory*, IEEE Transactions on Systems, Man, and Cybernetics, 2000.
- Z. Tong, P. Xu, T. Denœux, *An evidential classifier based on Dempster-Shafer theory and deep learning*, Neurocomputing, 2021.
- K. He, X. Zhang, S. Ren, J. Sun, *Deep Residual Learning for Image Recognition*, CVPR, 2016.
- Y. Sun, Y. Ming, X. Zhu, Y. Li, *Out-of-Distribution Detection with Deep Nearest Neighbors*, 2022. [arXiv:2204.06507](https://arxiv.org/abs/2204.06507)
- D. Alvarez-Melis, T. S. Jaakkola, *On the robustness of interpretability methods*, 2018. [arXiv:1806.08049](https://arxiv.org/abs/1806.08049)
- M. Pawelczyk, K. Broelemann, G. Kasneci, *Learning model-agnostic counterfactual explanations for tabular data*, WWW '20, pp. 3126–3132, ACM, 2020.
- D. Dua, C. Graff, *UCI Machine Learning Repository*, 2017. [archive.ics.uci.edu](https://archive.ics.uci.edu/)

## ⚠️ Note

The paper associated with this code, **"Robust Explanations Through Uncertainty Decomposition: A Path to Trustworthier AI"**, has been accepted for the ICPR camera-ready stage but has **not yet been formally published**. If you use or reference this code, please make sure to **clearly indicate this status**.
