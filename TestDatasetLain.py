import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns

# 1. PATHS
BASE_DIR      = "/home/lintangarai/Documents/ISIC 2019"
METADATA_PATH = os.path.join(BASE_DIR, "train-metadata.csv")
IMAGE_DIR     = os.path.join(BASE_DIR, "image")

IMG_SIZE   = 224
BATCH_SIZE = 16
AUTOTUNE   = tf.data.AUTOTUNE

# 2. LOAD METADATA
df = pd.read_csv(METADATA_PATH)
print("Columns:", df.columns.tolist())

# 3. BUILD PATH MAP
path_map = {}
for filename in os.listdir(IMAGE_DIR):
    if filename.endswith(".jpg"):
        image_id = filename.replace(".jpg", "")
        path_map[image_id] = os.path.join(IMAGE_DIR, filename)

df       = df[df['isic_id'].isin(path_map)].reset_index(drop=True)
df['path'] = df['isic_id'].map(path_map)

# 4. LABEL
df['label'] = df['target']

print("\nDistribusi label:")
print(df['label'].value_counts())

# 5. SPLIT (group by patient_id)
X      = df['path'].values
y      = df['label'].values
groups = df['patient_id'].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

test_paths = X[test_idx]
y_test     = y[test_idx]  

print(f"\nTest size: {len(test_paths)}")
print(f"Test cancer ratio: {y_test.mean():.2%}")

# 6. DATASET
def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, label

test_ds = (
    tf.data.Dataset.from_tensor_slices((test_paths, y_test))
    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# 7. LOAD MODEL
model = tf.keras.models.load_model(
    "PEKAKU_0.1_model.keras",
    compile=False
)

# 8. PREDICT
y_pred_probs = model.predict(test_ds).flatten()

THRESHOLD = 0.31
y_pred    = (y_pred_probs > THRESHOLD).astype(int)

# 9. EVALUASI
print(f"\n=== EXTERNAL TEST (threshold={THRESHOLD}) ===")
print(classification_report(y_test, y_pred, target_names=["Non-Cancer", "Cancer"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_probs):.4f}")

# 10. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Non-Cancer", "Cancer"],
            yticklabels=["Non-Cancer", "Cancer"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — External Test")
plt.tight_layout()
plt.savefig("confusion_matrix_external.png")
plt.show()

# 11. ROC CURVE
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
roc_auc     = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — External Test")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_external.png")
plt.show()

print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_probs):.4f}")