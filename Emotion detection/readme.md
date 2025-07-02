# Emotion Detection Project Report

## Introduction

Facial emotion recognition is a challenging problem in computer vision, with applications in human-computer interaction, security, and psychology. The goal of this project is to classify facial images into discrete emotion categories using a combination of labelled and unlabelled data. Our high-level approach involves leveraging Principal Component Analysis (PCA) for feature extraction—using both labelled and unlabelled images—and training a Support Vector Machine (SVM) classifier for emotion prediction.

---

## Methodology

### Data Preprocessing

- **Unlabelled Data:** We used the LFW (Labeled Faces in the Wild) dataset as a source of unlabelled facial images. Each image was converted to grayscale, resized to 64x64 pixels, and flattened into a 1D array.
- **Labelled Data:** The `labelled.csv` file contains pixel data and emotion labels. Each image was parsed, converted to a numpy array, and split into training and test sets based on the `Usage` column.
- **Scaling:** All image data was standardized using `StandardScaler` to ensure zero mean and unit variance, which is important for PCA and SVM performance.

### PCA Feature Engineering

- **Leveraging Unlabelled Data:** We fit PCA on the scaled unlabelled LFW images to learn a robust set of principal components that capture general facial structure and variation.
- **Dimensionality Reduction:** The optimal number of PCA components was determined by plotting the explained variance ratio for different values of `k`. We selected 100 components, balancing information retention and computational efficiency.
- **Application to Labelled Data:** The same PCA transformation (fit on unlabelled data) was applied to the labelled dataset, ensuring that the features used for classification are informed by a broader distribution of facial features.

### SVM Classification

- **Model Choice:** We used a linear SVM (`SVC(kernel='linear', C=1, gamma='scale')`) for its effectiveness in high-dimensional spaces and interpretability.
- **Training:** The SVM was trained on the PCA-transformed training set and evaluated on the test set.
- **Justification:** The linear kernel is suitable given the high dimensionality and the nature of the PCA features. The regularization parameter `C=1` was chosen as a reasonable default.

---

## Results

### Evaluation Metrics

The model was evaluated using accuracy, precision, recall, F1-score, and the confusion matrix.

```
Classification Report:
              precision    recall  f1-score   support

           0       0.43      0.33      0.38         9
           1       1.00      0.67      0.80        12
           3       1.00      0.93      0.96        14
           5       0.84      0.94      0.89        17
           6       0.92      0.96      0.94       119

    accuracy                           0.90       171
   macro avg       0.84      0.77      0.79       171
weighted avg       0.90      0.90      0.90       171

Confusion Matrix:
[[  3   0   0   0   6]
 [  1   8   0   0   3]
 [  0   0  13   1   0]
 [  0   0   0  16   1]
 [  3   0   0   2 114]]

Accuracy: 90.06%
```

### Visualizations

- **Explained Variance Plot:** Shows how much variance is captured by different numbers of PCA components.
- **Confusion Matrix:** Visualizes the distribution of true vs. predicted labels.

---

## Discussion & Insights

### Insights from SVD on Unlabelled Data

Applying PCA (SVD) to the unlabelled LFW dataset allowed us to extract principal components that represent general facial features, not biased by emotion labels. This improved the robustness of our feature extraction and helped the classifier generalize better.

### Effectiveness of Incorporating Unlabelled Data

Using unlabelled data for PCA provided a richer, more diverse basis for feature extraction, which likely improved classification performance compared to using only labelled data for PCA.

### Challenges and Solutions

- **Data Quality:** Unlabelled images varied in quality and pose. We addressed this by standardizing image size and grayscale conversion.
- **Class Imbalance:** Some emotions were underrepresented. While not fully addressed, future work could include class balancing techniques.
- **Hyperparameter Tuning:** Limited tuning was performed; further optimization could improve results.

### Limitations

- **Model Simplicity:** The linear SVM may not capture complex, non-linear relationships.
- **Feature Limitation:** PCA features, while informative, may not capture subtle emotion cues.
- **Dataset Constraints:** The labelled dataset may not be large or diverse enough for optimal generalization.

### Potential Next Steps

- **Deep Learning:** Explore CNNs or transfer learning for improved feature extraction.
- **Semi-supervised Learning:** Use techniques that directly leverage both labelled and unlabelled data during training.
- **Data Augmentation:** Increase labelled data diversity through augmentation.
- **Hyperparameter Optimization:** Systematic search for optimal SVM and PCA parameters.

---

**Author:**  
Ronit Ranjan