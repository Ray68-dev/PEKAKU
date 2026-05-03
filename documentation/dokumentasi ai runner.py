import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc, fbeta_score
)
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# 1. PATHS
BASE_DIR      = r"/home/lintangarai/Documents/Skin Cancer DATASET"
METADATA_PATH = os.path.join(BASE_DIR, "HAM10000_metadata.csv")
IMAGE_DIRS    = [
    os.path.join(BASE_DIR, "HAM10000_images_part_1"),
    os.path.join(BASE_DIR, "HAM10000_images_part_2")
]
IMG_SIZE   = 224
BATCH_SIZE = 16
AUTOTUNE   = tf.data.AUTOTUNE

# 2. LOAD METADATA
df = pd.read_csv(METADATA_PATH)
print(f"Metadata loaded: {len(df)} rows")

path_map = {}
for image_dir in IMAGE_DIRS:
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            image_id = filename.replace(".jpg", "")
            path_map[image_id] = os.path.join(image_dir, filename)

df = df[df['image_id'].isin(path_map)].reset_index(drop=True)
df['path'] = df['image_id'].map(path_map)

# 3. BALANCING
df['label'] = df['dx'].apply(
    lambda x: 1 if x in ['mel', 'bcc', 'akiec', 'bkl'] else 0
)

n_cancer  = (df['label'] == 1).sum()
target_nv = min(n_cancer, 3500)

nv_df     = df[df['dx'] == 'nv'].sample(n=target_nv, random_state=42)
non_nv_df = df[df['dx'] != 'nv']
df        = pd.concat([nv_df, non_nv_df]).reset_index(drop=True)

print(f"Cancer    : {(df['label']==1).sum()}")
print(f"Non-Cancer: {(df['label']==0).sum()}")
print(f"Rasio Cancer: {df['label'].mean():.2%}")

# 4. SPLIT
X      = df['path'].values
y      = df['label'].values
groups = df['lesion_id'].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

train_paths, test_paths = X[train_idx], X[test_idx]
y_train,     y_test     = y[train_idx], y[test_idx]

print(f"\nTrain: {len(train_paths)} | Test: {len(test_paths)}")
print(f"Train Cancer ratio: {y_train.mean():.2%}")

# 5. SAMPLE WEIGHTS
neg, pos = np.bincount(y_train)
total    = neg + pos
w_neg    = total / (2 * neg)
w_pos    = total / (2 * pos)
print(f"Sample weights — Non-Cancer: {w_neg:.2f} | Cancer: {w_pos:.2f}")
sample_weights_train = np.where(y_train == 1, w_pos, w_neg).astype(np.float32)

# 6. tf.data PIPELINE
def load_and_preprocess(path, label, weight):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, label, weight

def load_and_preprocess_test(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, label

def augment(img, label, weight):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    return img, label, weight

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, y_train, sample_weights_train))
    .shuffle(buffer_size=1000, seed=42)
    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .map(augment,             num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((test_paths, y_test))
    .map(load_and_preprocess_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# 7. LOSS
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce     = -(y_true * tf.math.log(y_pred) +
                    (1 - y_true) * tf.math.log(1 - y_pred))
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * bce)
    return loss_fn

# 8. BUILD MODEL
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x      = base_model(inputs, training=False)
x      = layers.GlobalAveragePooling2D()(x)
x      = layers.BatchNormalization()(x)
x      = layers.Dense(128, activation='relu')(x)
x      = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=inputs, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)
model.summary()

# 9. CALLBACKS
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5,          # lebih sabar dari sebelumnya
    restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6
)

# 10. PHASE 1 — TRAIN HEAD
print("\n=== Phase 1: Training head (base frozen) ===")
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=15,                              
    callbacks=[early_stop, reduce_lr]
)

# 11. PHASE 2 — FINE-TUNE
print("\n=== Phase 2: Fine-tuning ===")
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

history_fine = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=15,
    callbacks=[early_stop, reduce_lr]
)

# 12. PREDICT
y_pred_probs = model.predict(test_ds).flatten()

print("\n=== Distribusi probabilitas prediksi ===")
print(f"Min : {y_pred_probs.min():.4f}")
print(f"Max : {y_pred_probs.max():.4f}")
print(f"Mean: {y_pred_probs.mean():.4f}")
print(f"Pred > 0.5: {(y_pred_probs > 0.5).sum()} / {len(y_pred_probs)}")

# 13. FIND OPTIMAL THRESHOLD
best_thresh, best_fb = 0.5, 0.0
for thresh in np.arange(0.05, 0.95, 0.01):
    preds = (y_pred_probs > thresh).astype(int)
    fb    = fbeta_score(y_test, preds, beta=2, pos_label=1, zero_division=0)
    if fb > best_fb:
        best_fb     = fb
        best_thresh = thresh

print(f"\nBest threshold (F-beta β=2): {best_thresh:.2f} → F-beta={best_fb:.3f}")
y_pred = (y_pred_probs > best_thresh).astype(int)

# 14. EVALUATION
print("\n" + classification_report(y_test, y_pred, target_names=["Non-Cancer", "Cancer"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_probs):.4f}")

# 15. PLOTS
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Non-Cancer", "Cancer"],
            yticklabels=["Non-Cancer", "Cancer"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

plt.figure()
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='val')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curve.png")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
roc_auc     = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

thresholds_range = np.arange(0.05, 0.95, 0.01)
fb_scores = [
    fbeta_score(y_test, (y_pred_probs > t).astype(int),
                beta=2, pos_label=1, zero_division=0)
    for t in thresholds_range
]
plt.figure()
plt.plot(thresholds_range, fb_scores, label='F-beta (β=2)')
plt.axvline(x=best_thresh, color='red', linestyle='--', label=f'Best={best_thresh:.2f}')
plt.xlabel("Threshold")
plt.ylabel("F-beta Score (Cancer)")
plt.title("F-beta Score vs Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("fbeta_threshold_curve.png")
plt.show()

# 16. SAVE
model.save("skin_cancer_model.keras")

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump({'threshold': float(best_thresh)}, f)

print("\nModel saved  : skin_cancer_model.keras")
print("Threshold    : label_encoder.pkl")
print("Plots saved  : confusion_matrix.png, accuracy_curve.png, roc_curve.png, fbeta_threshold_curve.png")