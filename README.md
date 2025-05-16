
# ML Pattern Classification Pipeline

An end-to-end machine learning pipeline for pattern classification, featuring comprehensive stages including preprocessing, feature transformation, dimensionality reduction, model training, and evaluation. This project demonstrates experimentation with various preprocessing techniques, classifiers, and feature reduction approaches to evaluate model performance under different scenarios.

---

## 📁 Project Structure

- `pattern_classification.ipynb` – Jupyter Notebook implementing the full ML workflow from preprocessing to model evaluation.
- `PS4_GamesSales.csv` – Raw dataset used for training and testing models.
- `Report.pdf` – Project report explaining the motivation, methodology, experiments, results, and conclusions.


---

## 📊 Dataset

The pipeline uses the `PS4_GamesSales.csv` dataset, which contains information on various PS4 games, including:

- Game Name
- Genre
- Publisher
- Global and regional sales
- Critic and user scores
- Release year

The target for classification is based on derived patterns or categories formed from one or more attributes (genre/sales/etc.).

---

## 🧪 Pipeline Steps

### 1. Data Preprocessing
- Removal of duplicates
- Handling of missing or invalid entries
- Label Encoding for ordinal variables
- One-Hot Encoding for nominal variables
- Normalization:
  - Min-Max Scaling
  - Z-Score Normalization

### 2. Feature Transformation
- Encoded categorical variables into numeric form
- Transformed original features using PCA
- Feature selection using SFFS (Sequential Floating Forward Selection)

### 3. Dimensionality Reduction
- **PCA**: Custom implementation to reduce feature space while preserving maximum variance
- **SFFS**: Wrapper-based feature selection to iteratively select an optimal subset of features

### 4. Model Training
Trained and evaluated the following classifiers on each dataset variation (original, normalized, PCA-reduced, SFFS-reduced):
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Bayesian Classifiers:
  - Naive Bayes
  - Multivariate Gaussian Bayes
  - Non-parametric Bayes (Parzen)

All models were evaluated using cross-validation for reliability.

### 5. Model Evaluation
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- Classification Report (per class and averaged)
- Performance comparison across preprocessing and reduction strategies

---

## 🚀 Getting Started

### Requirements

Ensure the following tools are installed:

- Python 3.6+
- Jupyter Notebook

Install dependencies using:

```bash
pip install -r requirements.txt
```

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running the Project

1. Clone the repository:
```bash
git clone https://github.com/patel-ab/ML-pattern-classification-pipeline.git
cd ML-pattern-classification-pipeline
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open `pattern_classification.ipynb` and execute the cells in sequence to run the pipeline end-to-end.
   
4. Make sure to update the location of the dataset.

---

## 📄 Documentation

For a detailed explanation of the methodology, results, and experimental observations, refer to the [Report.pdf](./Report.pdf) included in this repository.

---

## 📌 Key Highlights

- Custom PCA and SFFS implementations demonstrate deeper control over dimensionality reduction.
- Comparative study of model performance across different transformations and classifiers.
- Includes both parametric and non-parametric Bayesian classifiers for advanced pattern recognition.
- Easy-to-read and modular codebase for further extension or research.

---

## 🤝 Contributing

Contributions and suggestions are welcome! Feel free to:
- Fork the repository
- Create a new branch
- Make your changes
- Submit a pull request for review

