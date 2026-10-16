# Gamma vs. Hadron Classification (MAGIC Telescope Data)

Classifying high-energy particle events detected by the MAGIC gamma telescope as
**gamma rays (signal)** or **hadrons (background)**, using classical ML models in scikit-learn.

## Dataset
- [MAGIC Gamma Telescope dataset](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope) (UCI ML Repository)
- 19,020 samples, 10 numeric features describing the shower image
  (`fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, `fDist`)
- Target: `g` (gamma) → 1, `h` (hadron) → 0
- Classes are imbalanced (gamma is the majority class)

## Approach
1. Load data and encode labels
2. Explore feature distributions per class (histograms)
3. Split into train / validation / test (60 / 20 / 20)
4. Standardize features with `StandardScaler`
5. Balance the training set with `RandomOverSampler`
6. Train and compare models, evaluated on the held-out test set

## Results (test set, 3,804 samples)

| Model               | Accuracy | Macro F1 |
|---------------------|----------|----------|
| SVM (RBF)           | 0.86     | 0.84     |
| k-NN (k=3)          | 0.81     | 0.79     |
| Logistic Regression | 0.77     | 0.75     |
| Gaussian Naive Bayes| 0.72     | 0.66     |

SVM performed best. Naive Bayes struggled mainly with recall on the hadron class (0.41),
likely because its feature-independence assumption doesn't hold for these correlated features.

## Tech Stack
Python, pandas, NumPy, Matplotlib, scikit-learn, imbalanced-learn

## How to Run
```bash
git clone https://github.com/<Samarth-shukla-23>/<Cosmic-ray-ml>.git
cd <Cosic-ray-ml>
pip install -r requirements.txt
jupyter notebook radiation.ipynb
```
Download `magic04.data` from the UCI link above and place it in a `data/` folder.

## Future Work
- Hyperparameter tuning (GridSearchCV) for SVM and k-NN
- Train and evaluate the neural network (TensorFlow/Keras)
- Try tree-based models (Random Forest, Gradient Boosting)
- Use the validation set for model selection and add ROC-AUC
