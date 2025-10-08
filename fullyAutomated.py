import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
RAW_DATA_DIR = "/path/to/FaceSkinDiseases"   # Raw images
PROCESSED_DIR = "/path/to/ProcessedImages"   # Processed images
CSV_FILE = "skin_disease_features.csv"       # Features CSV
MODEL_FILE = "FaceSkinDisease_MobileNetV2.h5"
GRADCAM_DIR = "/path/to/GradCAM_Outputs"
IMAGE_SIZE = (224, 224)
BLUR_THRESHOLD = 100
BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FINE = 20
AUTOTUNE = tf.data.AUTOTUNE
LAST_CONV_LAYER = "Conv_1"  # MobileNetV2 last conv layer

# ---------------- HELPERS ----------------
def is_blurry(img, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

def detect_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = np.var(gray)
    if var < 500:
        return "Gaussian"
    elif np.max(gray) - np.min(gray) > 200:
        return "Salt&Pepper"
    else:
        return "Speckle"

def remove_noise(img, noise_type):
    if noise_type == "Gaussian":
        return cv2.GaussianBlur(img, (3,3), 0)
    elif noise_type == "Salt&Pepper":
        return cv2.medianBlur(img, 3)
    elif noise_type == "Speckle":
        return cv2.fastNlMeansDenoisingColored(img, None, 10,10,7,21)
    return img

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()
    ent = -np.sum(hist_norm*np.log2(hist_norm+1e-7))
    edges = cv2.Canny(gray,100,200)
    edge_density = np.sum(edges>0)/edges.size
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    return [ent, edge_density, mean_intensity, std_intensity]

# ---------------- PREPROCESSING ----------------
features_list = []
labels = []

Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

for cls in os.listdir(RAW_DATA_DIR):
    input_cls_path = os.path.join(RAW_DATA_DIR, cls)
    output_cls_path = os.path.join(PROCESSED_DIR, cls)
    Path(output_cls_path).mkdir(parents=True, exist_ok=True)
    
    for img_name in os.listdir(input_cls_path):
        img_path = os.path.join(input_cls_path, img_name)
        try:
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.resize(img, IMAGE_SIZE)
            if is_blurry(img): continue
            noise_type = detect_noise(img)
            img = remove_noise(img, noise_type)
            cv2.imwrite(os.path.join(output_cls_path, img_name), img)
            features = extract_features(img)
            features_list.append(features)
            labels.append(cls)
        except Exception as e:
            print(f"Error: {img_path} -> {e}")

# Save features CSV
df = pd.DataFrame(features_list, columns=["Entropy","EdgeDensity","MeanIntensity","StdIntensity"])
df["Label"] = labels
df.to_csv(CSV_FILE, index=False)
print(f"✅ Preprocessing & feature extraction done. CSV saved: {CSV_FILE}")

# ---------------- DEEP LEARNING TRAINING ----------------
data_root = Path(PROCESSED_DIR)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_root,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=123
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_root,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=123
)

class_names = train_ds.class_names
num_classes = len(class_names)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.08),
    layers.RandomBrightness(0.08),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
def prepare(ds, augment=False):
    ds = ds.map(lambda x,y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(lambda x,y: (data_augmentation(x,training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(train_ds, augment=True)
val_ds = prepare(val_ds)

base = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE+(3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = layers.Input(shape=IMAGE_SIZE+(3,))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train head
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

# Fine-tune
base.trainable = True
for layer in base.layers[:100]: layer.trainable=False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE)

# Save model
model.save(MODEL_FILE)
print(f"✅ Model saved: {MODEL_FILE}")

# ---------------- BATCH GRAD-CAM ----------------
Path(GRADCAM_DIR).mkdir(parents=True, exist_ok=True)

def grad_cam(img_path, save_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    preds = model.predict(x)
    pred_class = np.argmax(preds[0])
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAST_CONV_LAYER).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, pred_class]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:,:,i] *= pooled_grads[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)
    orig_img = cv2.imread(img_path)
    orig_img = cv2.resize(orig_img, IMAGE_SIZE)
    heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(orig_img,0.6,heatmap_color,0.4,0)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, superimposed_img)

for cls in class_names:
    input_cls = os.path.join(PROCESSED_DIR, cls)
    output_cls = os.path.join(GRADCAM_DIR, cls)
    Path(output_cls).mkdir(parents=True, exist_ok=True)
    for img_name in os.listdir(input_cls):
        try:
            grad_cam(os.path.join(input_cls,img_name), os.path.join(output_cls,img_name))
            print(f"✅ Grad-CAM processed: {img_name}")
        except:
            continue

print("🎯 Full pipeline complete! All outputs saved.")
