# Bone Fracture Classification Using EfficientNetB4

A deep learning pipeline for automated bone fracture classification from multi-region X-ray images. This project applies transfer learning with EfficientNetB4 to achieve clinical-grade classification performance on a binary fracture detection task.

**Author:** Muhammad Fiaz  
**Live Application:** https://bonescanai.streamlit.app  
**Notebook:** Available in this repository

---

## Project Summary

| Item | Detail |
|---|---|
| Task | Binary classification: Fractured vs Not Fractured |
| Model | EfficientNetB4 with custom classification head |
| Dataset | Bone Fracture Multi-Region X-ray Data (Kaggle) |
| Training Images | 9,246 |
| Validation Images | 829 |
| Test Images | 506 |
| Test Accuracy | 98.02% |
| F1 Score | 0.98 |
| AUC Score | 0.9985 |
| Correct Predictions | 496 out of 506 |

---

## Dataset

The dataset is the [Bone Fracture Multi-Region X-ray Dataset](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data) available on Kaggle. It covers multiple anatomical regions including the wrist, elbow, hand, shoulder, finger, and forearm. Each image is labeled as either fractured or not fractured.

| Split | Fractured | Not Fractured | Total |
|---|---|---|---|
| Train | 4,606 | 4,640 | 9,246 |
| Validation | 337 | 492 | 829 |
| Test | 238 | 268 | 506 |

---

## Model Architecture

The model uses EfficientNetB4 pretrained on ImageNet as the base, with the following classification head added on top:

- Global Average Pooling
- Batch Normalization
- Dropout (rate 0.4)
- Dense layer with 256 units and ReLU activation
- Dropout (rate 0.3)
- Dense output layer with sigmoid activation

Total parameters: 18.14 million  
Trainable parameters (Phase 1): 462,849

---

## Preprocessing

EfficientNetB4 requires its own specific preprocessing function rather than standard pixel rescaling. All images were resized to 380 x 380 pixels and processed using `tensorflow.keras.applications.efficientnet.preprocess_input`. Using standard 0-to-1 rescaling reduced validation accuracy to approximately 65 percent, confirming the critical role of architecture-matched preprocessing.

---

## Training

Training used a single-phase strategy with the base model frozen:

- Optimizer: Adam with learning rate 0.001
- Loss: Binary cross-entropy
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Best validation accuracy: 98.67% at epoch 9

Data augmentation on the training set included horizontal flipping, random rotation up to 10 degrees, zoom variation up to 10 percent, and width and height shifts up to 10 percent.

---

## Results

### Test Set Performance

| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| Fractured | 0.97 | 0.98 | 0.98 |
| Not Fractured | 0.98 | 0.98 | 0.98 |
| Overall | 0.98 | 0.98 | 0.98 |

- Test Accuracy: 98.02%
- Test Loss: 0.0506
- AUC Score: 0.9985

### Confusion Matrix

|  | Predicted Fractured | Predicted Not Fractured |
|---|---|---|
| Actual Fractured | 234 | 4 |
| Actual Not Fractured | 6 | 262 |

---

## Repository Structure

```
Bone-Fracture-Classification-Using-Deep-Learning/
├── model/
│   └── bone_fracture_efficientnetb4.keras
├── .streamlit/
│   └── config.toml
├── app.py
├── requirements.txt
├── Bone_Fracture_Classification_Using_EfficientNetB4.ipynb
├── .gitattributes
└── README.md
```

---

## How to Run the Notebook

1. Open the notebook in Google Colab
2. Go to Runtime and select T4 GPU
3. Run all cells in order
4. When prompted, upload your `kaggle.json` API key file

To get your Kaggle API key: go to kaggle.com, click your profile, go to Settings, scroll to API, and click Create New Token.

---

## How to Run the App Locally

Clone the repository:

```bash
git clone https://github.com/Fiazbhk/Bone-Fracture-Classification-Using-Deep-Learning.git
cd Bone-Fracture-Classification-Using-Deep-Learning
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The model file is tracked using Git LFS. Make sure Git LFS is installed before cloning if you need the model file locally.

---

## Dependencies

- Python 3.x
- TensorFlow 2.20.0
- Keras
- Streamlit
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Key Finding

A critical lesson from this project is that EfficientNetB4 requires its own preprocessing function, not standard rescaling. Using `rescale=1./255` resulted in validation accuracy stalling at 65 percent. Switching to `preprocessing_function=preprocess_input` from `tensorflow.keras.applications.efficientnet` immediately resolved the issue and the model converged to 93 percent accuracy in the very first epoch.

---

## License

This project is for academic and educational purposes.

---

## Author

Muhammad Fiaz  
Information Technology  
GitHub: https://github.com/Fiazbhk
