# 🧠 Fake News Detection using Multimodal Deep Learning (Text + Image)

### 👨‍💻 Developed by

**Shourya Pratik** (2023UGCS038)  
Department of Computer Science & Engineering  
**National Institute of Technology, Jamshedpur**  
Capstone Project – CS1507 (November 2025)

---

## 📝 Abstract

This project implements a **Fake News Detection** system for Hindi-language news articles using **Multimodal Deep Learning**.  
The model combines:  
* 🧾 **Textual content** → processed using **DistilBERT** (multilingual transformer)  
* 🖼️ **Image features** → extracted using **ResNet-18**  

Both are fused into a unified neural network to classify each news sample as **Real (1)** or **Fake (0)**.

---

## 🧩 Architecture Overview

The system integrates text and image encoders before classification:

```
Text (Hindi News)  ─►  DistilBERT Encoder ─┐
                                           │
Image (from Link) ─►  ResNet18 Encoder  ───┼─► [Feature Fusion + Classifier] ─►  Real/Fake
```

📸 **Outputs in `/outputs/`:**

* `architecture_diagram.png`
* `confusion_matrix.png`
* `roc_curve.png`
* `report_summary.txt`

---

## 📊 Features

✅ Multilingual **DistilBERT** for Hindi & English text  
✅ CNN-based **ResNet18** for image feature extraction  
✅ Data preprocessing with **synthetic fake sample generation**  
✅ Modular and lightweight (runs on **Google Colab**)  
✅ Generates automatic **evaluation report + plots**  

---

## 🧠 Model Summary

| Component         | Description                                    |
| ----------------- | ---------------------------------------------- |
| **Text Encoder**  | DistilBERT (transformer, 768-dim hidden)       |
| **Image Encoder** | ResNet-18 pretrained on ImageNet               |
| **Fusion Layer**  | Concatenates text + image embeddings           |
| **Classifier**    | 2 FC layers + Sigmoid                          |
| **Loss Function** | Binary Cross Entropy (BCEWithLogitsLoss)       |
| **Optimizer**     | Adam (LR = 1e-4)                               |
| **Frameworks**    | PyTorch, HuggingFace Transformers, TorchVision |

---

## 🧾 Dataset

| Field                     | Description                |
| ------------------------- | -------------------------- |
| **Statement**             | Hindi news headline/text   |
| **Label**                 | TRUE (real) / FALSE (fake) |
| **Link**                  | URL of associated image    |
| **Web / Category / Date** | Metadata (optional)        |

> Original source: Hindi news portals (Jagran, Bhaskar, etc.)  
> Synthetic fake samples were created for class balance (~1000 records total).

---

## 🧮 Results

| Metric    | Score    |
| --------- | -------- |
| Accuracy  | **0.88** |
| Precision | **0.87** |
| Recall    | **0.86** |
| F1 Score  | **0.86** |
| ROC-AUC   | **0.91** |

📊 *Saved plots in `outputs/`:*  
* `confusion_matrix.png`  
* `roc_curve.png`

---

## ⚙️ Environment Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/shouryapratikofficial/Fake-News-Detection-using-Multimodal-Deep-Learning-Text-Image-.git
cd Fake-News-Detection-using-Multimodal-Deep-Learning-Text-Image-
```

### 2️⃣ Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers datasets sentencepiece
pip install tqdm pandas scikit-learn matplotlib pillow nltk
```

### 3️⃣ Run the Project

Open the Jupyter/Colab notebook:

```bash
notebooks/Multimodal_FakeNews_Training.ipynb
```

or directly in Google Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🗂️ Folder Structure

```
Capstone/
├── data/
│   ├── hindi dataset.xlsx
│   ├── balanced_with_images.csv
│   ├── train.csv / val.csv / test.csv
├── images/
│   ├── placeholder_0.jpg ...
├── models/
│   ├── best_model.pth
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── architecture_diagram.png
│   ├── report_summary.txt
│   ├── test_predictions.csv
└── notebooks/
    └── Multimodal_FakeNews_Training.ipynb
```

---

## 🧠 How It Works

### 🔹 Text Processing

* Tokenization via `DistilBERTTokenizer`
* Encoding → contextual embeddings (768-dim)

### 🔹 Image Processing

* ResNet18 backbone (frozen)
* Extracts 512-dim image embedding

### 🔹 Fusion

Concatenates `[768 + 512]` feature vectors → Dense(256) → Sigmoid

---

## 🧩 Sample Code

```python
# Forward pass
logits = model(image_tensor, input_ids, attention_mask)
probs = torch.sigmoid(logits)
preds = (probs >= 0.5).int()
```

```python
# Evaluation metrics
from sklearn.metrics import accuracy_score, classification_report
print(classification_report(true_labels, preds))
```

---

## 📈 Visualization (Manual Insert)

| Figure                | File                       | Description                         |
| --------------------- | -------------------------- | ----------------------------------- |
| 🧠 Model Architecture | `architecture_diagram.png` | Visual overview of model            |
| 🔢 Confusion Matrix   | `confusion_matrix.png`     | Correct vs. incorrect predictions   |
| 📊 ROC Curve          | `roc_curve.png`            | Model performance across thresholds |

---

## 🎓 Conclusion

This project demonstrates a **lightweight yet powerful multimodal system** for fake news detection in Hindi media.  
By combining semantic understanding (text) and contextual cues (image), the system achieves robust accuracy with minimal compute resources.  
The modular architecture can be extended to real-world datasets or deployed as a browser plugin for real-time news verification.

---

## 🚀 Future Scope

* Fine-tuning **IndicBERT** for deeper Hindi understanding
* Integrating **real fake-news datasets** for enhanced accuracy
* Deploying model via **Flask/FastAPI** web app
* Building Chrome Extension for real-time news validation

---

## 📚 References

* Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers,” 2018
* He et al., “Deep Residual Learning for Image Recognition,” 2015
* HuggingFace Transformers Documentation
* PyTorch and TorchVision APIs

---

## 🏁 License

This project is developed for **academic and educational use** under the **MIT License**.

---

## 🙌 Acknowledgments

Special thanks to the Department of Computer Science & Engineering, **NIT Jamshedpur**, for guidance and support during this project.

---

## 💬 Contact

📧 **Shourya Pratik** – [2023UGCS038@nitjsr.ac.in](mailto:2023UGCS038@nitjsr.ac.in)