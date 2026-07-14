"""
train_model.py
===============
Machine Learning training pipeline for the Cardiovascular Disease Prediction System.

Workflow:
    Load Dataset -> Data Cleaning -> EDA -> Visualization -> Train-Test Split
    -> Feature Scaling -> Train Models -> Evaluate Models -> Save Best Model (Pickle)

Run this ONCE before starting the Flask app:
    python train_model.py

Outputs:
    model.pkl        -> the best performing trained model
    scaler.pkl       -> the fitted StandardScaler
    static/plots/*   -> all generated visualizations (PNG)
    metrics.json     -> metrics + comparison table consumed by the dashboard
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd

# Use a non-interactive backend so plots render without a display server.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_PATH = "heart.csv"
PLOTS_DIR = os.path.join("static", "plots")
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
METRICS_PATH = "metrics.json"

# Human readable labels for each raw column (used in plots / reports).
FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Rest ECG",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Angina",
    "oldpeak": "ST Depression",
    "slope": "Slope",
    "ca": "Major Vessels",
    "thal": "Thalassemia",
    "target": "Heart Disease",
}

os.makedirs(PLOTS_DIR, exist_ok=True)


def save_fig(fig, name):
    """Helper to save a matplotlib figure into the plots directory."""
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


# ---------------------------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------------------------
def load_data():
    print("[1/8] Loading dataset ...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2. Data Cleaning
# ---------------------------------------------------------------------------
def clean_data(df):
    print("[2/8] Cleaning data ...")
    # Handle missing values by filling numeric columns with the median.
    missing_before = int(df.isnull().sum().sum())
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    # Remove duplicate rows.
    dup_before = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  missing values handled: {missing_before}")
    print(f"  duplicates removed: {dup_before}")
    return df


# ---------------------------------------------------------------------------
# 3 & 4. EDA + Visualization
# ---------------------------------------------------------------------------
def generate_visualizations(df):
    print("[3/8] Generating visualizations ...")
    palette = ["#2563eb", "#dc2626"]

    # Count Plot -- target class balance
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="target", data=df, palette=palette, ax=ax)
    ax.set_title("Target Count Plot")
    ax.set_xticklabels(["Healthy", "Disease"])
    save_fig(fig, "count_plot.png")

    # Pie Chart -- target distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df["target"].value_counts().sort_index()
    ax.pie(counts, labels=["Healthy", "Disease"], autopct="%1.1f%%",
           colors=palette, startangle=90, wedgeprops={"edgecolor": "white"})
    ax.set_title("Target Distribution (Pie Chart)")
    save_fig(fig, "pie_chart.png")

    # Histogram -- age distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["age"], bins=20, kde=True, color="#2563eb", ax=ax)
    ax.set_title("Age Histogram")
    save_fig(fig, "histogram.png")

    # Box Plot -- cholesterol by target
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x="target", y="chol", data=df, palette=palette, ax=ax)
    ax.set_title("Cholesterol Box Plot by Target")
    ax.set_xticklabels(["Healthy", "Disease"])
    save_fig(fig, "box_plot.png")

    # Pair Plot -- a subset of key numeric features
    subset = df[["age", "trestbps", "chol", "thalach", "oldpeak", "target"]]
    pair = sns.pairplot(subset, hue="target", palette=palette, corner=True)
    pair.fig.suptitle("Pair Plot of Key Features", y=1.02)
    pair.fig.savefig(os.path.join(PLOTS_DIR, "pair_plot.png"), dpi=100, bbox_inches="tight")
    plt.close(pair.fig)
    print(f"  saved -> {os.path.join(PLOTS_DIR, 'pair_plot.png')}")

    # Scatter Plot -- age vs max heart rate
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x="age", y="thalach", hue="target", data=df,
                    palette=palette, ax=ax)
    ax.set_title("Age vs Max Heart Rate")
    ax.legend(title="Target", labels=["Healthy", "Disease"])
    save_fig(fig, "scatter_plot.png")

    # Heatmap / Correlation Matrix
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix Heatmap")
    save_fig(fig, "heatmap.png")
    # Same figure reused conceptually as the correlation matrix image.
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest",
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix")
    save_fig(fig, "correlation_matrix.png")

    # Feature Distribution -- KDE grid of all numeric features
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.ravel()
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], kde=True, color="#2563eb", ax=axes[i])
        axes[i].set_title(FEATURE_LABELS.get(col, col))
    for j in range(len(df.columns), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Feature Distributions", y=1.01, fontsize=16)
    save_fig(fig, "feature_distribution.png")

    # Target Distribution (bar with counts)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="target", data=df, palette=palette, ax=ax)
    for p in ax.patches:
        ax.annotate(int(p.get_height()),
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Target Distribution")
    ax.set_xticklabels(["Healthy", "Disease"])
    save_fig(fig, "target_distribution.png")

    # Age vs Heart Disease
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(data=df, x="age", hue="target", multiple="stack",
                 palette=palette, bins=20, ax=ax)
    ax.set_title("Age vs Heart Disease")
    save_fig(fig, "age_vs_disease.png")

    # Gender vs Heart Disease
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(x="sex", hue="target", data=df, palette=palette, ax=ax)
    ax.set_title("Gender vs Heart Disease")
    ax.set_xticklabels(["Female", "Male"])
    ax.legend(title="Target", labels=["Healthy", "Disease"])
    save_fig(fig, "gender_vs_disease.png")

    # Chest Pain Analysis
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(x="cp", hue="target", data=df, palette=palette, ax=ax)
    ax.set_title("Chest Pain Type Analysis")
    ax.set_xlabel("Chest Pain Type")
    ax.legend(title="Target", labels=["Healthy", "Disease"])
    save_fig(fig, "chest_pain_analysis.png")

    # Blood Pressure Analysis
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(data=df, x="trestbps", hue="target", fill=True,
                palette=palette, ax=ax)
    ax.set_title("Resting Blood Pressure Analysis")
    save_fig(fig, "blood_pressure_analysis.png")

    # Cholesterol Analysis
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(data=df, x="chol", hue="target", fill=True,
                palette=palette, ax=ax)
    ax.set_title("Cholesterol Analysis")
    save_fig(fig, "cholesterol_analysis.png")

    # Build correlation insight text for the correlation page.
    target_corr = corr["target"].drop("target").sort_values(ascending=False)
    insights = {
        "positive": [
            {"feature": FEATURE_LABELS.get(k, k), "value": round(float(v), 3)}
            for k, v in target_corr[target_corr > 0].items()
        ],
        "negative": [
            {"feature": FEATURE_LABELS.get(k, k), "value": round(float(v), 3)}
            for k, v in target_corr[target_corr < 0].items()
        ],
    }
    return insights


# ---------------------------------------------------------------------------
# 5, 6, 7. Split -> Scale -> Train
# ---------------------------------------------------------------------------
def train_models(df):
    print("[4/8] Splitting data (train/test) ...")
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[5/8] Scaling features ...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("[6/8] Training models ...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "K-Nearest Neighbor": KNeighborsClassifier(n_neighbors=11),
    }

    results = []
    trained = {}
    print("[7/8] Evaluating models ...")
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                       target_names=["Healthy", "Disease"],
                                       zero_division=0)
        train_acc = accuracy_score(y_train, model.predict(X_train_s))

        # Save a confusion matrix plot per model.
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Healthy", "Disease"],
                    yticklabels=["Healthy", "Disease"], ax=ax)
        ax.set_title(f"Confusion Matrix\n{name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fname = "cm_" + name.lower().replace(" ", "_") + ".png"
        save_fig(fig, fname)

        results.append({
            "model": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "train_accuracy": round(train_acc, 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "cm_plot": "plots/" + fname,
        })
        trained[name] = model
        print(f"  {name:<24} acc={acc:.3f} f1={f1:.3f}")

    # Comparison bar chart of accuracy.
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r["model"] for r in results]
    accs = [r["accuracy"] for r in results]
    bars = ax.bar(names, accs, color="#2563eb")
    ax.set_ylim(0, 1)
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=20, ha="right")
    for b, a in zip(bars, accs):
        ax.annotate(f"{a:.2f}", (b.get_x() + b.get_width() / 2, a),
                    ha="center", va="bottom")
    save_fig(fig, "model_comparison.png")

    # Determine the best model by F1 score (tie-broken by accuracy).
    best = max(results, key=lambda r: (r["f1"], r["accuracy"]))
    best_model = trained[best["model"]]
    print(f"  BEST MODEL -> {best['model']}")

    return best_model, scaler, results, best, len(X_train), len(X_test)


# ---------------------------------------------------------------------------
# 8. Save artifacts
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    df = clean_data(df)
    insights = generate_visualizations(df)
    best_model, scaler, results, best, n_train, n_test = train_models(df)

    print("[8/8] Saving model, scaler and metrics ...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    metrics = {
        "dataset_size": int(len(df)),
        "train_size": int(n_train),
        "test_size": int(n_test),
        "n_features": int(df.shape[1] - 1),
        "best_model": best["model"],
        "best_train_accuracy": best["train_accuracy"],
        "best_test_accuracy": best["accuracy"],
        "results": results,
        "correlation_insights": insights,
        "columns": list(df.columns),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone! Artifacts created:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {SCALER_PATH}")
    print(f"  - {METRICS_PATH}")
    print(f"  - {PLOTS_DIR}/ (all plots)")
    print("\nNow run:  python app.py")


if __name__ == "__main__":
    main()
