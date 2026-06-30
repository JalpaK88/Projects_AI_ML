# Projects_AI_ML
A collection of Deep Learning, Image Processing, and Pattern Recognition Projects.

## 📁 Repository Structure

Projects_AI_ML/
├── Projects_DeepLearning/
│   └── Driver's Drowsiness Detection using Deep Learning
├──Project_ImagePro_PatternReco/
│   └── Image Processing Projects
│   └── └── Assignment_1_IP.zip
│   └── └── Assignment_2_IP.zip
│   └── └── Assignment_3_IP.zip
│   └── └── Assignment_4_IP.zip
│   └── Pattern Recognition Projects
│   └── └── Assignment_1_All.zip
│   └──└──  Assignment_2_All.zip
│   └──└──  Assignment_3_All.zip
│   └── └── Assignment_4_All.zip



## 1.🚗 Projects_DeepLearning — Driver's Drowsiness Detection

Built a drowsiness detection system using the **MRL Eye Dataset**, comparing **InceptionV3** and **VGG16** architectures across different train-test splits and conditions (illumination, reflection, viewpoint variation).

**Key Findings:**
- InceptionV3 outperformed VGG16 under challenging real-world conditions
- Validated higher accuracy and robustness of InceptionV3 for drowsiness detection

**Contents:**
- `DDD_inception_v3_3_5_60_40.ipynb` / `DDD_inception_v3_3_5_70_30.ipynb` — InceptionV3 model, 60/40 and 70/30 splits
- `DDD_vgg16_3_5_60_40.ipynb` / `DDD_vgg16_3_5_70_30.ipynb` — VGG16 model, 60/40 and 70/30 splits
- `Project_Deep_learning.pdf` — Full Research Paper as per IEEE standard format with methodology and results.

📄 View project details: Projects_AI_ML/Projects_DeepLearning

## 2.🖼️ Project_ImagePro_PatternReco — Image Processing & Pattern Recognition

# 2.1. Project_ImageProcessing

A collection of Projects implementing core digital image processing techniques from scratch using Python (NumPy, PIL, Matplotlib) — focusing on understanding the underlying math rather than relying on built-in library functions.

---

## 📂 Assignment 1 — Image Sampling, Quantization & Histogram Processing

Implements fundamental image transformations from first principles:
- **Sub-sampling** — reducing image resolution by a sub-sampling factor and observing the effect on image quality
- **Quantization** — reducing the number of gray levels in an image to study the trade-off between bit-depth and visual quality
- **Histogram Equalization** — computing image histograms manually, deriving the PDF/CDF, and applying histogram equalization to enhance contrast
- **Histogram Specification (Matching)** — mapping one image's histogram to match a target/specified histogram

📁 `Assignment_1_IP.zip` | Includes a written report (`Assignment_1_report.pdf`)

---

## 📂 Assignment 2 — Spatial Filtering & Pattern Matching

Covers spatial-domain filtering techniques and correlation-based detection:
- **Correlation & Pattern Matching** — using cross-correlation to locate a known pattern within a larger image
- **Smoothing Filters** — averaging and Gaussian smoothing filters at different kernel sizes (7×7, 15×15) to study blurring effects
- **Noise Reduction** — comparing averaging vs. median filtering for removing noise from images (salt-and-pepper style noise)
- **Sharpening & Gradients** — Laplacian-based image sharpening and computing partial-derivative gradients for edge enhancement

📁 `Assignment_2_IP.zip` | Includes a written report (`Assignment_2_report.pdf`)

---

## 📂 Assignment 3 — Frequency Domain Analysis (DFT)

Explores the Discrete Fourier Transform and frequency-domain representation of signals and images:
- **1D DFT/IDFT** — manual implementation of the forward and inverse Discrete Fourier Transform, visualizing real, imaginary, and magnitude components for test signals (cosine wave, rectangular function)
- **2D DFT of Images** — computing the 2D DFT of synthetic images, with magnitude spectrum centering and log transformation for visualization
- **Phase vs. Magnitude Analysis** — reconstructing images using only magnitude information (zero phase) and only phase information (unit magnitude) to demonstrate the relative importance of phase in image content

📁 `Assignment3_IP.zip` | Includes a written report (`Assignment_3_report.pdf`)

---

## 📂 Assignment 4 — Frequency Domain Filtering

Builds on DFT fundamentals to perform filtering directly in the frequency domain:
- **Gaussian Filtering** — implementing a Gaussian filter from scratch and comparing spatial-domain vs. frequency-domain filtering results
- **Frequency Domain Padding** — zero-padding images for FFT-based convolution
- **Homomorphic Filtering** — illuminating/reflectance separation using log transformation + high-pass filtering in the frequency domain, with experiments across different γ_L (low-frequency gain) and γ_H (high-frequency gain) parameters

📁 `Assignment_4.zip` | Includes a written report (`Assignment_4_report.pdf`)

---

## 🛠️ Key Learning:

- **Language:** Python
- **Libraries:** NumPy, PIL (Pillow), Matplotlib
- **Approach:** Most operations (DFT, filtering, histogram equalization) are implemented manually rather than using high-level library calls, to demonstrate understanding of the underlying algorithms

📄 View project details: Projects_AI_ML/Project_ImagePro_PatternReco/Projects_ImageProcessing.zip

# 2.2. Projects_PatternRecognition


A collection of Projects implementing core statistical pattern recognition algorithms from scratch in Python — covering Bayesian decision theory, parameter estimation, skin-color detection, Eigenfaces, and classifier comparison (SVM vs. Bayes).

---

## 📂 Assignment 1 — Bayesian Decision Theory & Classifiers

Implements Bayes classification on synthetically generated 2D Gaussian data:
- **Synthetic Data Generation** — generating 2D Gaussian-distributed samples manually using the Box-Muller transform
- **Bayes Classifier (Case I & Case III)** — deriving and applying discriminant functions under different covariance-matrix assumptions, plotting decision boundaries
- **Error Analysis** — computing theoretical error bounds and empirical misclassification rates
- **Euclidean Distance Classifier** — classifying samples using minimum Euclidean distance to class means, for comparison against the Bayes classifier

📁 `Assignment_1_All.zip` | Includes a written report (`Assignment_1_report.pdf`)

---

## 📂 Assignment 2 — Parameter Estimation & Skin Color Detection

Extends Assignment 1 with parameter estimation and a real-world classification task:
- **Maximum Likelihood Estimation** — estimating Gaussian parameters (mean, covariance) from varying amounts of training data and analyzing how estimation accuracy changes with sample size
- **Skin Color Detection (Chromatic Color Space)** — building a Gaussian skin-color model in RGB-normalized chromatic space, with ROC curve analysis and Equal Error Rate (EER) threshold selection
- **Skin Color Detection (YCbCr Color Space)** — repeating the skin detection task in YCbCr color space and comparing classification performance against the chromatic-space model

📁 `Assignment_2_All.zip` | Includes a written report (`Assignment_2_report.pdf`)

---

## 📂 Assignment 3 — Eigenfaces & Face Recognition (PCA)

A face recognition system built using Principal Component Analysis (Eigenfaces), evaluated on high- and low-resolution face datasets:
- **Eigenface Computation** — computing the average face, mean-subtracted face space, and eigenvectors/eigenvalues of the covariance matrix to build the eigenface basis
- **Dimensionality Selection (CMC Curves)** — evaluating recognition accuracy (Cumulative Match Characteristic curves) while retaining different proportions of variance (80%, 90%, 95%) to select the optimal number of eigenfaces
- **High vs. Low Resolution Comparison** — repeating recognition experiments at full (48×60) and reduced (16×20) image resolutions to study the accuracy/efficiency trade-off
- **Intruder (Open-Set) Detection** — testing the system's ability to reject unknown/unseen identities not present in the training gallery

📁 `Submission_Assignment_3.zip` | Includes a written report (`Assignment_3_report.pdf`)

---

## 📂 Assignment 4 — Gender Classification: SVM vs. Bayes

Compares two classifiers for gender classification using PCA-reduced face features (eigen-coefficients):
- **Experiment 1 — SVM Classification** — training Support Vector Machines with RBF and Polynomial kernels, tuning hyperparameters (C, gamma) via grid search, evaluated at high and low resolution
- **Experiment 2 — Bayes Classification** — training a Bayes classifier using Mahalanobis distance with class-specific mean and covariance estimates, evaluated at high and low resolution

📁 `Assignment_4_all.zip` | Includes a written report (`Assignment_4_report.pdf`)

---

## 🛠️ Key Learning:

- **Language:** Python
- **Libraries:** NumPy, Matplotlib, PIL, LIBSVM (`svm_train`/`svm_predict`)
- **Core Techniques:** Bayesian decision theory, Maximum Likelihood Estimation, PCA/Eigenfaces, SVM, Mahalanobis distance, ROC/EER analysis

📄 View project details: Projects_AI_ML/Project_ImagePro_PatternReco/Project_PatternRecognition.zip

## 🛠️ Overall Tech Stack from AI/ML Projects

- Programming Skills: Python (Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn)
- Data Science & Statistical Modeling: TensorFlow, PyTorch, Keras,CNN architectures, Supervised/Unsupervised Learning,
Feature Engineering, Correlation, Cross-Validation, Confusion Matrix, Precision, Recall, F1-score, ROC Analysis, Anomaly Detection, Predictive Modeling, A/B Testing.

## 📫 Contact

Feel free to reach out if you have questions about any of these projects.

