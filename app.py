"""
app.py
======
Flask web server for the Cardiovascular Disease Prediction System.

Routes:
    /               Home / landing page
    /upload         Upload a CSV dataset and inspect it (shape, types, nulls, stats)
    /visualization  Gallery of all generated EDA plots + correlation matrix
    /dashboard      Model comparison table, best model, key metrics
    /prediction     Prediction form + result (probability %)
    /about          Project / ML explanation
    /contact        Contact form (stored in SQLite)

Before running, execute:  python train_model.py
Then start the server:     python app.py
Open:                      http://127.0.0.1:5000
"""

import os
import json
import pickle
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)

app = Flask(__name__)
app.secret_key = "cardio-disease-prediction-secret-key"

# Limit uploads to 5 MB.
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
METRICS_PATH = "metrics.json"
DB_PATH = "history.db"

# Order of features expected by the trained model (must match train_model.py).
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


# ---------------------------------------------------------------------------
# Helpers: load ML artifacts + metrics
# ---------------------------------------------------------------------------
def load_artifacts():
    """Load the pickled model and scaler if they exist."""
    model = scaler = None
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    return model, scaler


def load_metrics():
    """Load metrics.json produced by the training script."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Database (SQLite) for prediction + contact history
# ---------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                age INTEGER, sex INTEGER, cp INTEGER, trestbps INTEGER,
                chol INTEGER, fbs INTEGER, restecg INTEGER, thalach INTEGER,
                exang INTEGER, oldpeak REAL, slope INTEGER, ca INTEGER,
                thal INTEGER, result TEXT, probability REAL
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT, name TEXT, email TEXT, message TEXT
        )"""
    )
    conn.commit()
    conn.close()


def save_prediction(values, result, probability):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO predictions
           (created_at, age, sex, cp, trestbps, chol, fbs, restecg, thalach,
            exang, oldpeak, slope, ca, thal, result, probability)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + values + [result, probability],
    )
    conn.commit()
    conn.close()


def recent_predictions(limit=8):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def save_contact(name, email, message):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO contacts (created_at, name, email, message) VALUES (?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, email, message),
    )
    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    metrics = load_metrics()
    return render_template("index.html", metrics=metrics)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    context = {"info": None, "error": None}
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or file.filename == "":
            context["error"] = "Please choose a CSV file to upload."
            return render_template("upload.html", **context)
        if not file.filename.lower().endswith(".csv"):
            context["error"] = "Only .csv files are supported."
            return render_template("upload.html", **context)
        try:
            df = pd.read_csv(file)
        except Exception as exc:  # noqa: BLE001
            context["error"] = f"Could not read CSV: {exc}"
            return render_template("upload.html", **context)

        # Build dataset summary information for display.
        dtypes = pd.DataFrame(
            {"Column": df.columns, "Type": [str(t) for t in df.dtypes]}
        )
        missing = pd.DataFrame(
            {"Column": df.columns, "Missing": df.isnull().sum().values}
        )
        describe = df.describe().round(2).reset_index()

        context["info"] = {
            "filename": file.filename,
            "shape": df.shape,
            "rows": df.shape[0],
            "cols": df.shape[1],
            "head": df.head(10).to_html(
                classes="table table-striped table-sm mb-0", index=False, border=0
            ),
            "dtypes": dtypes.to_html(
                classes="table table-sm mb-0", index=False, border=0
            ),
            "missing": missing.to_html(
                classes="table table-sm mb-0", index=False, border=0
            ),
            "total_missing": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "describe": describe.to_html(
                classes="table table-striped table-sm mb-0", index=False, border=0
            ),
        }
    return render_template("upload.html", **context)


@app.route("/visualization")
def visualization():
    metrics = load_metrics()
    # Discover which plot files actually exist so the page never shows broken images.
    plots = [
        ("Count Plot", "count_plot.png"),
        ("Pie Chart", "pie_chart.png"),
        ("Histogram", "histogram.png"),
        ("Box Plot", "box_plot.png"),
        ("Pair Plot", "pair_plot.png"),
        ("Scatter Plot", "scatter_plot.png"),
        ("Feature Distribution", "feature_distribution.png"),
        ("Target Distribution", "target_distribution.png"),
        ("Age vs Heart Disease", "age_vs_disease.png"),
        ("Gender vs Heart Disease", "gender_vs_disease.png"),
        ("Chest Pain Analysis", "chest_pain_analysis.png"),
        ("Blood Pressure Analysis", "blood_pressure_analysis.png"),
        ("Cholesterol Analysis", "cholesterol_analysis.png"),
    ]
    available = [
        (title, f"plots/{fname}")
        for title, fname in plots
        if os.path.exists(os.path.join("static", "plots", fname))
    ]
    has_corr = os.path.exists(os.path.join("static", "plots", "correlation_matrix.png"))
    return render_template(
        "visualization.html",
        plots=available,
        has_corr=has_corr,
        metrics=metrics,
    )


@app.route("/dashboard")
def dashboard():
    metrics = load_metrics()
    return render_template(
        "dashboard.html",
        metrics=metrics,
        predictions=recent_predictions(),
    )


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    metrics = load_metrics()
    result = None
    if request.method == "POST":
        model, scaler = load_artifacts()
        if model is None or scaler is None:
            flash("Model not found. Please run 'python train_model.py' first.", "danger")
            return render_template("prediction.html", result=None, metrics=metrics)

        try:
            values = [
                int(request.form["age"]),
                int(request.form["sex"]),
                int(request.form["cp"]),
                int(request.form["trestbps"]),
                int(request.form["chol"]),
                int(request.form["fbs"]),
                int(request.form["restecg"]),
                int(request.form["thalach"]),
                int(request.form["exang"]),
                float(request.form["oldpeak"]),
                int(request.form["slope"]),
                int(request.form["ca"]),
                int(request.form["thal"]),
            ]
        except (ValueError, KeyError):
            flash("Please fill in all fields with valid values.", "danger")
            return render_template("prediction.html", result=None, metrics=metrics)

        arr = np.array(values).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        pred = int(model.predict(arr_scaled)[0])

        # Probability of the positive (disease) class.
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(arr_scaled)[0][1]) * 100
        else:
            proba = 100.0 if pred == 1 else 0.0

        label = "Heart Disease Detected" if pred == 1 else "Healthy"
        save_prediction(values, label, round(proba, 2))
        result = {
            "label": label,
            "positive": pred == 1,
            "probability": round(proba, 2),
            "healthy_prob": round(100 - proba, 2),
        }
    return render_template("prediction.html", result=result, metrics=metrics)


@app.route("/about")
def about():
    metrics = load_metrics()
    return render_template("about.html", metrics=metrics)


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()
        if name and email and message:
            save_contact(name, email, message)
            flash("Thank you! Your message has been received.", "success")
            return redirect(url_for("contact"))
        flash("Please complete all fields before submitting.", "danger")
    return render_template("contact.html")


if __name__ == "__main__":
    # host=0.0.0.0 so the preview environment can detect the port.
    app.run(host="0.0.0.0", port=5000, debug=True)
