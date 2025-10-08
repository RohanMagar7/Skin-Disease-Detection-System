import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIG ---
CSV_FILE = "skin_disease_features.csv"  # generated from preprocessing
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- LOAD DATA ---
df = pd.read_csv(CSV_FILE)
X = df[["Entropy", "EdgeDensity", "MeanIntensity", "StdIntensity"]].values
y = df["Label"].values

# --- ENCODE LABELS ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# --- SCALE FEATURES ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)

# --- TRAIN CLASSIFIER ---
clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

# --- EVALUATE ---
y_pred = clf.predict(X_test)

print("✅ Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
