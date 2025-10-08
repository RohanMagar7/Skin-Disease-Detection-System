import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIG ---
DATA_DIR = "/path/to/FaceSkinDiseases"  # Original dataset folder
SAVE_DIR = "/path/to/ProcessedImages"   # Processed images saved here
IMAGE_SIZE = (224, 224)
BLUR_THRESHOLD = 100  # Laplacian variance threshold for blur detection
FEATURE_CSV = "skin_disease_features.csv"

# --- HELPER FUNCTIONS ---
def is_blurry(image, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

def detect_noise_type(image):
    """Detect noise type: Gaussian, Salt & Pepper, Speckle"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    var = np.var(gray)
    mean = np.mean(gray)
    
    if var < 500:
        return "Gaussian"
    elif np.max(gray) - np.min(gray) > 200:
        return "Salt&Pepper"
    else:
        return "Speckle"

def remove_noise(image, noise_type):
    """Apply denoising based on detected noise type"""
    if noise_type == "Gaussian":
        return cv2.GaussianBlur(image, (3,3), 0)
    elif noise_type == "Salt&Pepper":
        return cv2.medianBlur(image, 3)
    elif noise_type == "Speckle":
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        return image

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 1. Entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    hist_norm = hist.ravel()/hist.sum()
    ent = -np.sum(hist_norm*np.log2(hist_norm + 1e-7))
    # 2. Edge density (Canny)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges>0) / edges.size
    # 3. Mean intensity
    mean_intensity = np.mean(gray)
    # 4. Standard deviation
    std_intensity = np.std(gray)
    return [ent, edge_density, mean_intensity, std_intensity]

# --- PROCESS DATASET ---
processed_features = []
labels = []

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

for disease_class in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, disease_class)
    save_class_path = os.path.join(SAVE_DIR, disease_class)
    Path(save_class_path).mkdir(parents=True, exist_ok=True)
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Resize
            img = cv2.resize(img, IMAGE_SIZE)
            
            # Blur check
            if is_blurry(img):
                print(f"⚠️  Blurry image detected: {img_path}, skipping...")
                continue
            
            # Detect noise type and remove it
            noise_type = detect_noise_type(img)
            img = remove_noise(img, noise_type)
            
            # Save processed image
            save_path = os.path.join(save_class_path, img_name)
            cv2.imwrite(save_path, img)
            
            # Extract features
            features = extract_features(img)
            processed_features.append(features)
            labels.append(disease_class)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# --- SAVE FEATURES TO CSV ---
df = pd.DataFrame(processed_features, columns=["Entropy", "EdgeDensity", "MeanIntensity", "StdIntensity"])
df["Label"] = labels
df.to_csv(FEATURE_CSV, index=False)
print(f"✅ Preprocessing complete! Features saved to {FEATURE_CSV}")
