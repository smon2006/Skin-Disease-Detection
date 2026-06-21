# Skin Disease Detection using EfficientNet-B0

A Deep Learning project using **PyTorch** and **Transfer Learning** to detect 22 different types of skin conditions. This model utilizes the **EfficientNet-B0** architecture, fine-tuned with advanced techniques like Test-Time Augmentation (TTA), Label Smoothing, and Cosine Annealing.
## 🚀 Live Web Application
Try the live interactive dashboard here: **[skin-disease-detection-system.streamlit.app](https://skin-disease-detection-system.streamlit.app/)** 

## Performance Summary
* **Top-1 Accuracy:** 74.90%
* **Top-5 Accuracy:** 95.80%
* **Best F1-Score:** 0.96 (Unknown_Normal) & 0.91 (Vitiligo)

---

## Tech Stack & Methods
-  **Frontend App:** Streamlit (Cloud Deployed)
- **Framework:** PyTorch
- **Base Model:** `EfficientNet_B0` (Pre-trained on ImageNet)
- **Optimizer:** `AdamW` with differential learning rates (features: `1e-7`, classifier: `1e-6`)
- **Scheduler:** `CosineAnnealingLR`
- **Loss Function:** `CrossEntropyLoss` with **Label Smoothing (0.1)** and **Class Weights** to handle dataset imbalance.
- **Augmentations:** Random Horizontal/Vertical Flips, ColorJitter, Random Rotation, and Random Cropping.
- **Weights Pipeline:** Production-optimized stream fetching binaries on-demand from Hugging Face Hub to keep the repository lightweight.

## Dataset
The model was trained on the [Skin Disease Dataset](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset) from Kaggle, containing 13,898 training images and 1,546 test images across 22 classes:

| Index | Class Name | Index | Class Name |
|---|---|---|---|
| 0 | Acne | 11 | Psoriasis |
| 1 | Actinic Keratosis | 12 | Rosacea |
| 2 | Benign Tumors | 13 | Seborrh_Keratoses |
| 3 | Bullous | 14 | Skin Cancer |
| 4 | Candidiasis | 15 | Sun Sunlight Damage |
| 5 | Drug Eruption | 16 | Tinea |
| 6 | Eczema | 17 | Unknown Normal |
| 7 | Infestations Bites | 18 | Vascular Tumors |
| 8 | Lichen | 19 | Vasculitis |
| 9 | Lupus | 20 | Vitiligo |
| 10| Moles | 21 | Warts |

---

## Results Analysis

### Confusion Matrix
The model shows exceptional performance on "Unknown_Normal" and "Vitiligo," while demonstrating common clinical overlaps between inflammatory conditions like Eczema and Psoriasis.

### Classification Report (Sample)
| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Unknown_Normal** | 0.96 | 0.97 | **0.97** |
| **Vitiligo** | 0.90 | 0.93 | **0.92** |
| **Acne** | 0.89 | 0.86 | 0.88 |
| **Skin Cancer** | 0.72 | 0.75 | 0.73 |

---

## How to Use

### 1. Prerequisites
Install the required application ecosystem dependencies:
```bash
pip install streamlit torch torchvision pillow
```
### 2. Run the Application
Launch the unified interface server directly on your local host environment:
```bash
streamlit run streamlit_app.py
```
---

## Disclaimer
*This model is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified dermatologist.*
