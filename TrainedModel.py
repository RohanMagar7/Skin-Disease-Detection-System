import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib
import matplotlib.pyplot as plt

# --- CONFIG ---
DATA_DIR = "ProcessedImages"  # preprocessed images folder
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FINE = 20
AUTOTUNE = tf.data.AUTOTUNE

# --- LOAD DATASET ---
data_root = pathlib.Path(DATA_DIR)
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
print("✅ Classes:", class_names)

# --- DATA AUGMENTATION ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.08),
    layers.RandomBrightness(0.08),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

def prepare(ds, augment=False):
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(train_ds, augment=True)
val_ds = prepare(val_ds)

# --- MODEL ---
base = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = layers.Input(shape=IMAGE_SIZE + (3,))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --- TRAIN HEAD ---
history_head = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

# --- FINE-TUNE ---
base.trainable = True
fine_tune_at = 100
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE)

# --- SAVE MODEL ---
model.save("FaceSkinDisease_MobileNetV2_DL.h5")
print("✅ Deep Learning model saved successfully.")

# --- OPTIONAL: Plot training curves ---
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history_head.history['accuracy'] + history_ft.history['accuracy'], label='train_acc')
plt.plot(history_head.history['val_accuracy'] + history_ft.history['val_accuracy'], label='val_acc')
plt.title("Accuracy")
plt.legend()
plt.subplot(1,2,2)
plt.plot(history_head.history['loss'] + history_ft.history['loss'], label='train_loss')
plt.plot(history_head.history['val_loss'] + history_ft.history['val_loss'], label='val_loss')
plt.title("Loss")
plt.legend()
plt.show()
