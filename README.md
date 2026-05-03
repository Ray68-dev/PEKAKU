# 🇮🇩 README (Bahasa Indonesia)

## Deteksi Kanker Kulit dengan Deep Learning

Proyek ini merupakan model *deep learning* berbasis **Convolutional Neural Network (CNN)** menggunakan **EfficientNetB0** untuk mendeteksi kanker kulit dari citra dermatoskopi.

Dataset yang digunakan adalah **HAM10000 (Human Against Machine with 10,000 training images)**.

---

## Fitur Utama

* Klasifikasi **Cancer vs Non-Cancer**
* Menggunakan arsitektur **EfficientNetB0 (transfer learning)**
* Penanganan **class imbalance** (undersampling + focal loss)
* Optimasi threshold berbasis **F-beta score (β=2)**
* Augmentasi data untuk meningkatkan generalisasi
* Evaluasi lengkap:

  * Confusion Matrix
  * ROC Curve & AUC
  * Classification Report

---

## Arsitektur Model

* Backbone: `EfficientNetB0 (ImageNet pretrained)`
* Fully Connected:

  * Global Average Pooling
  * Dense (128)
  * Dropout (0.5)
  * Output Sigmoid (Binary Classification)

---

## Evaluasi Model

Model dievaluasi menggunakan:

* Accuracy
* Precision, Recall, F1-score
* ROC-AUC
* Confusion Matrix

Contoh output:

```
Precision | Recall | F1-score
```

---

## Output

Setelah training:

* `skin_cancer_model.keras` → model terlatih
* `threshold.pkl` → threshold optimal
* `confusion_matrix.png`
* `roc_curve.png`

---

## Catatan

* Dataset tidak seimbang → tangani dengan:

  * Undersampling
  * Focal Loss
* Disarankan menggunakan GPU untuk training lebih cepat
