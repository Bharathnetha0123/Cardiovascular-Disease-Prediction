# Cardiovascular Disease Prediction System

A complete Machine Learning web application that predicts the risk of cardiovascular
(heart) disease from clinical parameters. Built with **Flask**, **Scikit-learn**,
**Pandas/NumPy**, **Matplotlib/Seaborn** and a responsive **Bootstrap 5** front-end.

---

## Features

1. **Home Page** – attractive, responsive landing page with project overview.
2. **Dataset Upload** – upload a CSV and inspect shape, first 10 rows, data types,
   missing values and a statistical summary.
3. **Data Preprocessing** – missing-value handling, duplicate removal, encoding and
   feature scaling.
4. **Data Visualization** – 15+ plots: count plot, pie chart, histogram, box plot,
   pair plot, scatter plot, heatmap, correlation matrix, feature/target distributions,
   age/gender/chest-pain/blood-pressure/cholesterol analyses.
5. **Correlation Matrix** – heatmap plus positive / negative / highly-correlated insights.
6. **Machine Learning Models** – Logistic Regression, Decision Tree, Random Forest, SVM
   and KNN trained and evaluated (accuracy, precision, recall, F1, confusion matrix,
   classification report) with a comparison table and best-model highlight.
7. **Prediction Page** – professional form with all 13 clinical inputs returning
   *Healthy* or *Heart Disease Detected* plus a probability score.
8. **Dashboard** – dataset size, training/testing accuracy, best model, charts and
   recent predictions.
9. **About Page** – explains ML, heart-disease prediction, the dataset and algorithms.
10. **Contact Page** – simple contact form (stored in SQLite).

---

## Tech Stack

| Layer            | Technology                                   |
|------------------|----------------------------------------------|
| Frontend         | HTML5, CSS3, JavaScript, Bootstrap 5         |
| Backend          | Python, Flask                                |
| Machine Learning | Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn |
| Database         | SQLite (prediction & contact history)        |

---

## Project Structure

```
HeartDiseasePrediction/
├── app.py                # Flask web server & routes
├── train_model.py        # ML pipeline: clean → EDA → train → evaluate → save
├── requirements.txt      # Python dependencies
├── README.md
├── heart.csv             # Dataset (UCI heart-disease schema, 13 features + target)
├── model.pkl             # Best trained model (created by train_model.py)
├── scaler.pkl            # Fitted StandardScaler (created by train_model.py)
├── metrics.json          # Metrics + comparison table (created by train_model.py)
├── history.db            # SQLite DB (auto-created on first run)
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── dashboard.html
│   ├── prediction.html
│   ├── visualization.html
│   ├── about.html
│   └── contact.html
│
└── static/
    ├── css/style.css
    ├── js/script.js
    ├── images/
    └── plots/            # All generated graphs (created by train_model.py)
```

---

## Getting Started (Visual Studio Code)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models (generates model.pkl, scaler.pkl, metrics.json and all plots)
```bash
python train_model.py
```

### 3. Run the web application
```bash
python app.py
```

### 4. Open in your browser
```
http://127.0.0.1:5000
```

---

## Machine Learning Workflow

```
Load Dataset → Data Cleaning → EDA → Visualization → Train-Test Split
→ Feature Scaling → Train Models → Evaluate Models → Save Best Model (Pickle) → Prediction
```

---

## Dataset Features

| Feature   | Description                          |
|-----------|--------------------------------------|
| age       | Age in years                         |
| sex       | 1 = male, 0 = female                 |
| cp        | Chest pain type (0–3)                |
| trestbps  | Resting blood pressure (mm Hg)       |
| chol      | Serum cholesterol (mg/dl)            |
| fbs       | Fasting blood sugar > 120 mg/dl      |
| restecg   | Resting ECG results (0–2)            |
| thalach   | Maximum heart rate achieved          |
| exang     | Exercise-induced angina (1/0)        |
| oldpeak   | ST depression                        |
| slope     | Slope of peak exercise ST segment    |
| ca        | Number of major vessels (0–4)        |
| thal      | Thalassemia (0–3)                    |
| target    | 1 = disease, 0 = healthy             |

---

## Notes

- The included `heart.csv` follows the standard UCI heart-disease schema. You can
  replace it with your own dataset that uses the same columns and re-run
  `python train_model.py`.
- This application is for **educational / academic purposes only** and must not be
  used as a substitute for professional medical advice or diagnosis.
```
