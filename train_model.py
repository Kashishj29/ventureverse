import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

DATA_FILE = "global_startup_success_dataset.csv"
MODEL_FILE = "ventureverse_model.joblib"

def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip(),
        errors='coerce'
    )

def map_tech_stack(text):
    s = str(text).lower().strip()
    if not s or s == "nan" or s == "none": return "Unknown"
    if "s'ai" in s or "ml" in s or "python" in s: return "AI/ML"
    if "cloud" in s or "aws" in s: return "Cloud"
    if "data" in s or "sql" in s: return "Data"
    if "web" in s or "app" in s or "mobile" in s: return "Mobile/Web"
    return "Other"

def main():
    print(f"📥 Loading dataset: {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # --- 1. CLEANING ---
    numeric_cols = ["Total Funding ($M)", "Number of Employees", "Valuation ($B)", 
                    "Social Media Followers", "Founded Year"]
    
    for col in numeric_cols:
        df[col] = clean_numeric(df[col]).fillna(0)

    df["Tech Stack"] = df["Tech Stack"].apply(map_tech_stack)

    # --- 2. TARGET (Strict Success: IPO or Acquired) ---
    df["target_success"] = 0
    if "IPO?" in df.columns:
        df.loc[df["IPO?"].astype(str).str.lower().str.contains("yes"), "target_success"] = 1
    if "Acquired?" in df.columns:
        df.loc[df["Acquired?"].astype(str).str.lower().str.contains("yes"), "target_success"] = 1

    print("\n📊 Target Distribution (0=Fail, 1=Success):")
    print(df["target_success"].value_counts())

    # --- 3. BENCHMARKS ---
    success_df = df[df["target_success"] == 1].copy()
    success_df["cap_per_emp"] = success_df["Total Funding ($M)"] / success_df["Number of Employees"]
    success_df["val_mult"] = (success_df["Valuation ($B)"] * 1000) / success_df["Total Funding ($M)"]
    
    benchmarks = {
        "avg_funding": success_df["Total Funding ($M)"].median(),
        "avg_employees": success_df["Number of Employees"].median(),
        "avg_valuation": success_df["Valuation ($B)"].median(),
        "avg_followers": success_df["Social Media Followers"].median(),
        "avg_cap_per_emp": success_df["cap_per_emp"].replace([np.inf, -np.inf], 0).median(),
        "avg_val_mult": success_df["val_mult"].replace([np.inf, -np.inf], 0).median()
    }

    # --- 4. TOURNAMENT (Advanced Metrics) ---
    X = df[numeric_cols + ["Country", "Industry", "Funding Stage", "Tech Stack"]]
    y = df["target_success"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["Country", "Industry", "Funding Stage", "Tech Stack"]),
    ])

    print("\n⚔️  Model Tournament (Selection Metric: F1 Score)...")
    models = [
        {"name": "Logistic Regression", "model": LogisticRegression(max_iter=1000)},
        {"name": "Random Forest", "model": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)},
        {"name": "Gradient Boosting", "model": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)}
    ]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    best_f1 = -1
    best_pipeline = None
    best_name = ""
    best_metrics = {}

    # Print Table Header
    print(f"{'Model':<20} | {'F1':<6} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'AUC':<6}")
    print("-" * 65)

    for m in models:
        pipe = Pipeline(steps=[("pre", preprocessor), ("clf", m["model"])])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]

        # Calculate Indicators
        f1 = f1_score(y_test, preds)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        print(f"{m['name']:<20} | {f1:.3f}  | {acc:.3f}  | {prec:.3f}  | {rec:.3f}  | {auc:.3f}")

        # WINNER SELECTION: Strictly based on F1 Score
        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = pipe
            best_name = m["name"]
            best_metrics = {"f1": f1, "acc": acc, "auc": auc}

    print("-" * 65)
    print(f"🏆 WINNER: {best_name} (F1: {best_f1:.3f})")

    # --- 5. SAVE ---
    best_pipeline.fit(X, y) # Retrain on full data
    probs = best_pipeline.predict_proba(X)[:, 1]
    
    final_object = {
        "model": best_pipeline,
        "min_prob": np.percentile(probs, 5),
        "max_prob": np.percentile(probs, 95),
        "benchmarks": benchmarks,
        "model_metrics": best_metrics # Saving stats to display in App
    }
    
    joblib.dump(final_object, MODEL_FILE)
    print(f"✅ Model Saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()