# 🩺 Diabetes Prediction — ML Web Application

> A full end-to-end data science and machine learning project developed as a group assignment at the **Superior University Lahore**. The application trains multiple classifiers on 100,000 patient records and serves live diabetes risk predictions through a Flask web interface.

## 📁 Project Structure

```
diabetes-prediction/
│
├── app.py                     # Flask application (routes + prediction logic)
├── requirements.txt           # Python dependencies
├── feature_columns.json       # Ordered feature list saved during training
├── best_model.pkl             # Trained best-performing model (Random Forest)
├── scaler.pkl                 # Fitted StandardScaler
│
├── NOTEBOOK_EXPORT_CELL.py    # Notebook cell to export all ML assets
│
├── templates/
│   ├── index.html             # Analysis report / dashboard page
│   ├── predict.html           # Prediction input form
│   └── result.html            # Prediction result display
│
├── static/
│   ├── css/
│   │   └── style.css          # Global stylesheet
│   └── images/                # EDA and model evaluation plots (auto-generated)
│       ├── eda_histograms.png
│       ├── eda_boxplots.png
│       ├── eda_correlation_heatmap.png
│       ├── eda_gender_diabetes.png
│       ├── eda_smoking_diabetes.png
│       ├── m2_smote_balance.png
│       ├── m3_model_comparison.png
│       └── m3_confusion_matrices.png
│
└── DiabetesDataSet.csv        # Source dataset (100,000 records)
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) |
| Records | 100,000 |
| Features | 9 (8 input + 1 target) |
| Diabetic cases | ≈ 8.5 % |
| Missing values | 0 |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `gender` | Categorical | Male / Female / Other |
| `age` | Numeric | Patient age in years |
| `hypertension` | Binary | 1 = has hypertension |
| `heart_disease` | Binary | 1 = has heart disease |
| `smoking_history` | Categorical | never / former / current / ever / not current / No Info |
| `bmi` | Numeric | Body Mass Index (kg/m²) |
| `HbA1c_level` | Numeric | Glycated haemoglobin (%) |
| `blood_glucose_level` | Numeric | Fasting blood glucose (mg/dL) |
| `diabetes` | Binary | **Target** — 1 = Diabetic |

---

## 🔬 Methodology

### Member 1 — Exploratory Data Analysis
- Histograms for `age`, `bmi`, `blood_glucose_level`, `HbA1c_level` — right-skewed distributions observed
- Box plots for outlier detection — significant outliers found in BMI and blood glucose
- Correlation heatmap — blood glucose (*r* ≈ 0.42) and HbA1c (*r* ≈ 0.40) are strongest predictors
- Countplots for gender vs diabetes and smoking history vs diabetes

### Member 2 — Pre-processing Pipeline
1. **Label Encoding** — Gender: Female=0, Male=1, Other=2
2. **One-Hot Encoding** — Smoking history expanded into 6 binary columns via `pd.get_dummies`
3. **IQR Outlier Capping** — BMI and blood glucose clipped to [Q1 − 1.5·IQR, Q3 + 1.5·IQR]
4. **Feature Selection** — Retained features with |correlation with target| > 0.05
5. **SMOTE** — Synthetic Minority Over-sampling applied to training set to address the ~91.5% class imbalance

### Member 3 — Model Training & Evaluation
Four classifiers trained across **three dataset versions** (unscaled, standardised, normalised):

| Model | Notes |
|-------|-------|
| Logistic Regression | `max_iter=1000`, `random_state=42` |
| K-Nearest Neighbours | `n_neighbors=5` |
| Decision Tree | `max_depth=6`, `random_state=42` |
| **Random Forest** ✅ | `n_estimators=100`, `max_depth=8`, `random_state=42` |

**Primary metric: Recall** — minimising false negatives (missed diabetic patients) was the priority.

**Best model: Random Forest on the standardised dataset** — highest recall for the diabetic class with strong overall accuracy and F1-score.

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd diabetes-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Export ML assets from the notebook

Open `Diabetes.ipynb` in Jupyter, scroll to the bottom, paste the contents of `NOTEBOOK_EXPORT_CELL.py` as a new cell, and run it. This will generate:

```
best_model.pkl
scaler.pkl
feature_columns.json
static/images/*.png   (all EDA and evaluation plots)
```

### 4. Run the Flask app

```bash
python app.py
```

Then open your browser at **http://127.0.0.1:5000**

---

## 🌐 Application Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Analysis Report | Full EDA report with all visualisations, pipeline summary, and model results |
| `/predict` (GET) | Prediction Form | Clinical input form (demographics, medical history, measurements) |
| `/predict` (POST) | Prediction Result | Risk label, confidence score, risk level badge, recommendations |

---

## 🧠 Prediction Logic (`app.py`)

When a form is submitted, `build_input_vector()`:

1. Reads raw form values (gender, age, hypertension, heart disease, smoking history, BMI, HbA1c, blood glucose)
2. Label-encodes gender (Female=0, Male=1, Other=2)
3. One-hot encodes smoking history into the 6 OHE columns
4. Assembles values in the exact column order stored in `feature_columns.json`
5. Scales the vector using the loaded `scaler.pkl`
6. Runs `model.predict()` and `model.predict_proba()` to get prediction + confidence

Risk levels are assigned as:

| Condition | Risk Level |
|-----------|-----------|
| Diabetic + confidence ≥ 80% | High Risk |
| Diabetic + confidence < 80% | Moderate Risk |
| Not Diabetic + confidence ≥ 80% | Low Risk |
| Not Diabetic + confidence < 80% | Borderline |

---

## 📦 Dependencies

```
flask>=3.0
joblib>=1.3
numpy>=1.26
scikit-learn>=1.4
imbalanced-learn>=0.12
pandas>=2.2
```

---

## ⚠️ Disclaimer

This application is built **for educational purposes only** as part of a university course project. It is **not a medical diagnostic tool** and does not constitute medical advice. Always consult a qualified healthcare professional for any health concerns.

---

## 📄 License

Academic project —Superior University Lahore, Data Science Course, 2026.
