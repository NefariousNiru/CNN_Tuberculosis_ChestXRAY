# Multi-Class Classification of Pulmonary Diseases

This project explores Chest X-ray classification using both classical machine learning and deep learning techniques. The goal is to classify X-ray images into four categories: **COVID-19**, **Normal**, **Pneumonia**, and **Tuberculosis**. We implemented feature extraction, visualization, and interpretability techniques to build a robust and explainable pipeline.

---

## Techniques Used

1. **Deep Learning Models**:
   - **Custom CNN**: A lightweight model designed for Chest X-ray classification.
   - **DenseNet121**: A pre-trained network for extracting 1024-dimensional feature vectors.
   - **Fine-Tuned Custom CNN**: Enhanced performance via extended architecture.

2. **Classical Machine Learning**:
   - **SVM**: Achieved high accuracy using DenseNet121 features.
   - **Random Forest**: Versatile ensemble method for classification.

3. **Dimensionality Reduction**:
   - **PCA**, **t-SNE**, and **UMAP** for visualizing and understanding feature separability.

4. **Interpretability**:
   - **Grad-CAM** for visualizing model focus areas on X-rays.

---

## How to Run

1. **Get the Dataset here**:
2. **Prepare Dataset**:
   Organize images in the following structure:
   ```plaintext
   dataset/
   ├── train/
   │   ├── COVID19/
   │   ├── Normal/
   │   ├── Pneumonia/
   │   └── Tuberculosis/
   └── test/
       ├── COVID19/
       ├── Normal/
       ├── Pneumonia/
       └── Tuberculosis/
   
3. **Create a Virtual Environment and Install Dependencies:**
  ```bash
  pip install numpy pandas torch torchvision umap-learn scikit-learn seaborn imblearn matplotlib scipy
  ```
  
4. ```Run main.py```