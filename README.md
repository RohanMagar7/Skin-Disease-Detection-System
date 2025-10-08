# 🧠 Skin Disease Detection System

A research and educational project that detects **7 types of skin diseases** from images using **Deep Learning**. The system uses **MobileNetV2** (pretrained) for classification and provides **Grad-CAM visualizations** for explainability.

---

## 🔹 Project Overview

This project builds an **AI-powered skin disease detection system** using images. The key steps include:

1. **Data Preprocessing**: Resize images, check for blur, remove noise.  
2. **Feature Extraction**: Compute Entropy, Edge Density, Mean & Standard Deviation.  
3. **Deep Learning Model**: Transfer learning using MobileNetV2 for 7-class classification.  
4. **Grad-CAM Visualization**: Highlights areas of the face influencing predictions.  
5. **Streamlit Frontend**: User-friendly interface to upload images and see predictions.

---

## 🔹 Skin Disease Classes

The system predicts the following diseases:

- Chickenpox  
- Monkeypox  
- Herpes  
- Lupus  
- Melanoma  
- Measles  
- Scabies  

---

## 🔹 Project Structure

```
SkinDiseaseProject/
├─ RawImages/                  # Original dataset images
│   ├─ Chickenpox/
│   ├─ Monkeypox/
│   └─ ...
├─ ProcessedImages/            # Preprocessed images
├─ Scripts/                    # Python scripts for preprocessing, training, Grad-CAM
├─ Models/                     # Trained models (.h5)
├─ CSV/                        # Extracted features CSV
├─ GradCAM_Outputs/            # Grad-CAM heatmaps
├─ frontend/                   # Streamlit app code
│   └─ app.py
└─ requirements.txt            # Required Python libraries
```

---

## 🔹 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/SkinDiseaseProject.git
cd FaceSkinDiseaseProject
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

> Make sure Python >= 3.8 is installed. For GPU, install `tensorflow-gpu`.

---

## 🔹 Dataset

You need a dataset with **7 classes of skin diseases**. Place it inside the `RawImages/` folder, keeping subfolders per class.  

Example:

```
RawImages/
├─ Chickenpox/
│   ├─ img1.jpg
│   ├─ img2.jpg
│   └─ ...
├─ Monkeypox/
├─ Herpes/
└─ ...
```

---

## 🔹 Running the Project

### 1️⃣ Preprocessing & Feature Extraction

Run the preprocessing script:

```bash
python Scripts/preprocessing_pipeline.py
```

This will:

- Resize images  
- Check for blur and remove noisy images  
- Remove noise (Gaussian, Salt&Pepper, Speckle)  
- Save processed images in `ProcessedImages/`  
- Extract 4 features and save CSV in `CSV/skin_disease_features.csv`  

---

### 2️⃣ Deep Learning Training

Run the training script:

```bash
python Scripts/train_dl_model.py
```

This will:

- Load processed images  
- Apply data augmentation  
- Train MobileNetV2 for 7-class classification  
- Save the trained model in `Models/FaceSkinDisease_MobileNetV2_DL.h5`  

---

### 3️⃣ Grad-CAM Visualization

Generate heatmaps for all images:

```bash
python Scripts/batch_gradcam.py
```

- Heatmaps will be saved in `GradCAM_Outputs/<ClassName>/`  
- Helps understand which facial areas influence predictions  

---

### 4️⃣ Streamlit Frontend

Run the user interface:

```bash
streamlit run frontend/app.py
```

- Upload a face image  
- Get **predicted disease**, **confidence**, and **class-wise probabilities**  
- Optional: Integrate Grad-CAM to visualize affected areas  

---

## 🔹 Requirements

See `requirements.txt` for the full library list. Main libraries:

- TensorFlow  
- Keras  
- OpenCV  
- Numpy  
- Pandas  
- Scikit-learn  
- Streamlit  
- Matplotlib / Seaborn  

Install with:

```bash
pip install -r requirements.txt
```

---

## 🔹 Notes

- ⚠️ This system is **for educational and research purposes only**. It is **not a medical diagnostic tool**.  
- Images should be **clear and focused** on the face for best predictions.  
- Grad-CAM provides **visual explanations**, but predictions are AI-based.  

---

## 🔹 Future Work / Improvements

- Support **more skin diseases**  
- Improve accuracy with **larger dataset**  
- Add **multi-image batch predictions**  
- Deploy as **web service or mobile app**  
- Integrate **explainability reports** with Grad-CAM  

---

## 🔹 Authors / Contributors

- **Rohan Magar** – Developer & Research  
- Open-source contributors / AI community  

---

## 🔹 License

This project is **open-source** and available under the [MIT License](LICENSE).  
