"""
VentureVerse - Startup Success Prediction Platform
===================================================
A machine learning web application that predicts startup success probability
using pre-launch indicators derived from publicly available data.

Author: Kashish Jadhav (w2035589)
University of Westminster - BSc Computer Science Final Year Project
Supervisor: Ebad Majeed
"""

import os
import re
import json
import sqlite3
import numpy as np
import pandas as pd
import joblib

from flask import Flask, render_template, request, redirect, url_for, session, flash

# Try importing password hashing - use fallback if not available
try:
    from werkzeug.security import generate_password_hash, check_password_hash
except ImportError:
    import hashlib
    def generate_password_hash(password):
        return hashlib.sha256(password.encode()).hexdigest()
    def check_password_hash(stored, provided):
        return stored == hashlib.sha256(provided.encode()).hexdigest()


# =============================================================================
# APP CONFIGURATION
# =============================================================================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "ventureverse-dev-key-change-in-production")

# File paths
MODEL_PATH = "ventureverse_model.joblib"
DATA_FILE = "global_startup_success_dataset.csv"
DB_PATH = "users.db"
CURRENT_YEAR = 2026

# =============================================================================
# LOAD ML MODEL & REFERENCE DATA
# =============================================================================
print("🚀 Loading VentureVerse ML Model...")

try:
    model_data = joblib.load(MODEL_PATH)
    # Handle both old format (just pipeline) and new format (dict with metadata)
    if isinstance(model_data, dict):
        model = model_data.get("model")
        BENCHMARKS = model_data.get("benchmarks", {})
        MODEL_METRICS = model_data.get("model_metrics", {})
    else:
        model = model_data
        BENCHMARKS = {}
        MODEL_METRICS = {}
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Model loading error: {e}")
    model = None
    BENCHMARKS = {}
    MODEL_METRICS = {}

# Load reference data for dropdowns
try:
    df_ref = pd.read_csv(DATA_FILE)
    COUNTRIES = sorted(df_ref["Country"].dropna().astype(str).unique().tolist())
    INDUSTRIES = sorted(df_ref["Industry"].dropna().astype(str).unique().tolist())
    STAGES = sorted(df_ref["Funding Stage"].dropna().astype(str).unique().tolist())
    print(f"✅ Loaded {len(df_ref)} records from dataset")
except Exception as e:
    print(f"⚠️ Dataset loading error: {e}")
    COUNTRIES = ["USA", "UK", "Germany", "India", "China", "Canada", "Australia"]
    INDUSTRIES = ["Technology", "Healthcare", "Finance", "E-commerce", "AI/ML", "SaaS"]
    STAGES = ["Pre-Seed", "Seed", "Series A", "Series B", "Series C", "Series D+"]

# Tech stack options for the form
TECH_OPTIONS = [
    "AI/ML", "Cloud (AWS/Azure/GCP)", "Mobile (iOS/Android)", 
    "Web Development", "Blockchain/Web3", "Data Analytics",
    "DevOps/CI-CD", "Cybersecurity", "IoT", "Other"
]


# =============================================================================
# DATABASE HELPERS
# =============================================================================
def init_db():
    """Initialize SQLite database with users table"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Create predictions history table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            prediction_score REAL,
            input_data TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


def get_user_by_email(email: str):
    """Retrieve user by email address"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),))
    user = cur.fetchone()
    conn.close()
    return user


def create_user(full_name: str, email: str, password: str):
    """Create new user account"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
        (full_name.strip(), email.lower().strip(), generate_password_hash(password))
    )
    conn.commit()
    conn.close()


def save_prediction(user_id: int, score: float, input_data: dict):
    """Save prediction to history"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (user_id, prediction_score, input_data) VALUES (?, ?, ?)",
        (user_id, score, json.dumps(input_data))
    )
    conn.commit()
    conn.close()


# =============================================================================
# AUTHENTICATION HELPERS
# =============================================================================
def login_required():
    """Check if user is logged in"""
    return "user_id" in session


def get_current_user_name():
    """Get current logged in user's name"""
    return session.get("user_name", "User")


# =============================================================================
# FEATURE ENGINEERING HELPERS
# =============================================================================
def split_stack(text: str) -> list:
    """Split tech stack string into list"""
    parts = re.split(r"[,\|/;\n]+", str(text).lower())
    return [p.strip() for p in parts if p.strip()]


def compute_stack_flags(stack_text: str) -> dict:
    """Extract binary flags for tech stack categories"""
    s = str(stack_text).lower()
    
    keywords = {
        "has_ai_ml": ["ai", "ml", "machine learning", "tensorflow", "pytorch", "sklearn", "llm", "nlp"],
        "has_cloud": ["aws", "azure", "gcp", "google cloud", "cloud", "lambda", "s3", "kubernetes"],
        "has_mobile": ["android", "ios", "flutter", "react native", "mobile"],
        "has_blockchain": ["blockchain", "web3", "ethereum", "solana", "smart contract"],
        "has_data": ["data", "spark", "hadoop", "postgres", "mysql", "mongodb", "snowflake", "bigquery"],
        "has_security": ["security", "auth", "oauth", "jwt", "encryption", "iam", "cyber"],
        "has_devops": ["docker", "k8s", "kubernetes", "ci/cd", "github actions", "jenkins", "terraform"],
    }
    
    out = {}
    for col, words in keywords.items():
        out[col] = int(any(w in s for w in words))
    out["stack_count"] = len(split_stack(stack_text))
    return out


def map_tech_stack(text: str) -> str:
    """Map tech stack to category for model"""
    s = str(text).lower().strip()
    if not s or s == "nan" or s == "none":
        return "Unknown"
    if "ai" in s or "ml" in s or "machine" in s:
        return "AI/ML"
    if "cloud" in s or "aws" in s or "azure" in s or "gcp" in s:
        return "Cloud"
    if "data" in s or "sql" in s or "analytics" in s:
        return "Data"
    if "web" in s or "app" in s or "mobile" in s or "ios" in s or "android" in s:
        return "Mobile/Web"
    if "blockchain" in s or "web3" in s or "crypto" in s:
        return "Blockchain"
    if "security" in s or "cyber" in s:
        return "Security"
    return "Other"


def build_feature_row(form) -> pd.DataFrame:
    """Build feature DataFrame from form input for model prediction"""
    # Extract form values
    founded_year = int(form.get("founded_year", 0) or 0)
    total_funding = float(form.get("total_funding", 0) or 0)
    employees = int(form.get("employees", 0) or 0)
    valuation = float(form.get("valuation", 0) or 0)
    followers = float(form.get("social_followers", 0) or 0)
    
    country = form.get("country", "Unknown")
    industry = form.get("industry", "Unknown")
    stage = form.get("funding_stage", "Unknown")
    
    # Handle tech stack (could be list from multi-select or string)
    tech_stack_raw = form.getlist("tech_stack") if hasattr(form, 'getlist') else form.get("tech_stack", "")
    if isinstance(tech_stack_raw, list):
        tech_stack = ", ".join(tech_stack_raw)
    else:
        tech_stack = str(tech_stack_raw)
    
    # Map tech stack to category
    tech_stack_mapped = map_tech_stack(tech_stack)
    
    # Build the row matching training features
    row = {
        "Founded Year": founded_year,
        "Total Funding ($M)": total_funding,
        "Number of Employees": employees,
        "Valuation ($B)": valuation,
        "Social Media Followers": followers,
        "Country": country,
        "Industry": industry,
        "Funding Stage": stage,
        "Tech Stack": tech_stack_mapped,
    }
    
    return pd.DataFrame([row])


def get_classification(score: float) -> tuple:
    """Get classification label and color based on score"""
    if score >= 75:
        return "High Potential", "#34d399"  # Green
    elif score >= 50:
        return "Moderate Potential", "#fbbf24"  # Yellow/Amber
    elif score >= 25:
        return "Needs Improvement", "#f97316"  # Orange
    else:
        return "High Risk", "#f87171"  # Red


def generate_insights(form_data: dict, score: float) -> list:
    """Generate actionable insights based on input data"""
    insights = []
    
    funding = float(form_data.get("total_funding", 0) or 0)
    employees = int(form_data.get("employees", 0) or 0)
    valuation = float(form_data.get("valuation", 0) or 0)
    followers = int(form_data.get("social_followers", 0) or 0)
    stage = form_data.get("funding_stage", "")
    founded_year = int(form_data.get("founded_year", 2020) or 2020)
    
    company_age = CURRENT_YEAR - founded_year
    
    # Benchmarks (use loaded or defaults)
    avg_funding = BENCHMARKS.get("avg_funding", 15)
    avg_employees = BENCHMARKS.get("avg_employees", 50)
    avg_valuation = BENCHMARKS.get("avg_valuation", 0.1)
    avg_followers = BENCHMARKS.get("avg_followers", 10000)
    
    # Funding analysis
    if funding >= avg_funding:
        insights.append({"type": "pos", "text": f"Strong funding of ${funding}M exceeds typical successful startups (${avg_funding:.1f}M median)"})
    else:
        insights.append({"type": "neg", "text": f"Funding of ${funding}M is below successful startup median (${avg_funding:.1f}M)"})
    
    # Team size analysis
    if employees >= avg_employees * 0.7:
        insights.append({"type": "pos", "text": f"Team size of {employees} indicates good operational capacity"})
    elif employees < 10:
        insights.append({"type": "neg", "text": f"Small team ({employees}) may limit execution speed - consider strategic hiring"})
    
    # Valuation analysis
    if valuation > 0 and funding > 0:
        val_multiple = (valuation * 1000) / funding  # Convert B to M for ratio
        if val_multiple >= 10:
            insights.append({"type": "pos", "text": f"Valuation multiple of {val_multiple:.1f}x shows strong investor confidence"})
        elif val_multiple < 3:
            insights.append({"type": "neg", "text": f"Low valuation multiple ({val_multiple:.1f}x) - may need stronger value proposition"})
    
    # Social presence
    if followers >= avg_followers:
        insights.append({"type": "pos", "text": f"Strong social presence ({followers:,} followers) aids customer acquisition"})
    elif followers < 1000:
        insights.append({"type": "neg", "text": f"Limited social reach ({followers:,}) - invest in digital marketing"})
    
    # Stage-specific insights
    if stage in ["Series C", "Series D+"] and funding < 50:
        insights.append({"type": "neg", "text": f"Late-stage ({stage}) typically requires higher funding levels"})
    elif stage in ["Pre-Seed", "Seed"] and funding > 10:
        insights.append({"type": "pos", "text": f"Strong early funding (${funding}M at {stage}) indicates high potential"})
    
    # Company age
    if company_age <= 2 and score >= 50:
        insights.append({"type": "pos", "text": f"Young company ({company_age} years) showing strong early indicators"})
    elif company_age > 7 and stage in ["Pre-Seed", "Seed"]:
        insights.append({"type": "neg", "text": f"Company age ({company_age} years) vs stage ({stage}) may indicate slow progress"})
    
    return insights[:6]  # Return top 6 insights


# =============================================================================
# ROUTES: AUTHENTICATION
# =============================================================================
@app.route("/signup", methods=["GET", "POST"])
def signup():
    """User registration page"""
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        
        # Validation
        if not full_name or not email or not password:
            flash("Please fill all fields.", "error")
            return render_template("signup.html")
        
        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")
        
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("signup.html")
        
        if get_user_by_email(email):
            flash("Email already registered. Please login.", "error")
            return redirect(url_for("login"))
        
        # Create account
        create_user(full_name, email, password)
        flash("Account created successfully! Please login.", "success")
        return redirect(url_for("login"))
    
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """User login page"""
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        
        user = get_user_by_email(email)
        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")
        
        # Set session
        session["user_id"] = user["id"]
        session["user_name"] = user["full_name"]
        flash(f"Welcome back, {user['full_name']}!", "success")
        return redirect(url_for("home"))
    
    return render_template("login.html")


@app.route("/logout")
def logout():
    """User logout"""
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


# =============================================================================
# ROUTES: MAIN APPLICATION
# =============================================================================
@app.route("/", methods=["GET"])
def home():
    """Main prediction page"""
    if not login_required():
        return redirect(url_for("login"))
    
    return render_template(
        "index.html",
        countries=COUNTRIES,
        industries=INDUSTRIES,
        stages=STAGES,
        tech_options=TECH_OPTIONS,
        prediction=None,
        ring_color="#60a5fa",
        classification=None,
        error=None,
        form_data={},
        user_name=get_current_user_name()
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction request"""
    if not login_required():
        return redirect(url_for("login"))
    
    form_data = request.form.to_dict()
    # Handle multi-select for tech stack
    form_data["tech_stack"] = request.form.getlist("tech_stack")
    
    try:
        if model is None:
            raise Exception("Model not loaded. Please contact administrator.")
        
        # Build features and predict
        X = build_feature_row(request.form)
        proba = float(model.predict_proba(X)[0][1])
        prediction = round(proba * 100, 2)
        
        # Get classification and color
        classification, ring_color = get_classification(prediction)
        
        # Generate insights for the insights page
        insights = generate_insights(form_data, prediction)
        
        # Store in session for insights page
        session["last_prediction"] = {
            "score": prediction,
            "classification": classification,
            "color": ring_color,
            "form_data": form_data,
            "insights": insights
        }
        
        # Save to database
        if "user_id" in session:
            save_prediction(session["user_id"], prediction, form_data)
        
        return render_template(
            "index.html",
            countries=COUNTRIES,
            industries=INDUSTRIES,
            stages=STAGES,
            tech_options=TECH_OPTIONS,
            prediction=prediction,
            ring_color=ring_color,
            classification=classification,
            error=None,
            form_data=form_data,
            user_name=get_current_user_name()
        )
    
    except Exception as e:
        return render_template(
            "index.html",
            countries=COUNTRIES,
            industries=INDUSTRIES,
            stages=STAGES,
            tech_options=TECH_OPTIONS,
            prediction=None,
            ring_color="#60a5fa",
            classification=None,
            error=str(e),
            form_data=form_data,
            user_name=get_current_user_name()
        )


@app.route("/insights")
def insights():
    """Analytics dashboard with detailed insights"""
    if not login_required():
        return redirect(url_for("login"))
    
    # Get last prediction from session
    last_pred = session.get("last_prediction")
    
    if not last_pred:
        flash("Please make a prediction first.", "error")
        return redirect(url_for("home"))
    
    form_data = last_pred.get("form_data", {})
    
    # Extract values for charts
    funding = float(form_data.get("total_funding", 0) or 0)
    valuation = float(form_data.get("valuation", 0) or 0)
    employees = int(form_data.get("employees", 0) or 0)
    founded_year = int(form_data.get("founded_year", 2020) or 2020)
    
    company_age = max(CURRENT_YEAR - founded_year, 1)
    employees_safe = max(employees, 1)
    
    # Calculate ratios
    cap_per_emp = funding / employees_safe
    val_mult = (valuation * 1000) / max(funding, 0.001)  # Valuation in $B, funding in $M
    funding_velocity = funding / company_age
    
    # Benchmark values
    bench_funding = BENCHMARKS.get("avg_funding", 15)
    bench_valuation = BENCHMARKS.get("avg_valuation", 0.1)
    bench_cap_per_emp = BENCHMARKS.get("avg_cap_per_emp", 0.5)
    bench_val_mult = BENCHMARKS.get("avg_val_mult", 10)
    
    # Prepare chart data
    financial_json = json.dumps({
        "labels": ["Total Funding ($M)", "Valuation ($B × 100)"],
        "user": [funding, valuation * 100],
        "benchmark": [bench_funding, bench_valuation * 100]
    })
    
    ratio_json = json.dumps({
        "labels": ["Capital/Employee ($M)", "Valuation Multiple"],
        "user": [round(cap_per_emp, 2), round(val_mult, 1)],
        "benchmark": [round(bench_cap_per_emp, 2), round(bench_val_mult, 1)]
    })
    
    velocity_json = json.dumps({
        "labels": ["Funding Velocity"],
        "user": [round(funding_velocity, 2)],
        "benchmark": [round(bench_funding / 3, 2)]  # Assume 3 year average
    })
    
    data = {
        "score": last_pred.get("score", 0),
        "classification": last_pred.get("classification", "Unknown"),
        "color": last_pred.get("color", "#60a5fa"),
        "insights": last_pred.get("insights", []),
        "financial_json": financial_json,
        "ratio_json": ratio_json,
        "velocity_json": velocity_json,
    }
    
    return render_template(
        "insights.html",
        data=data,
        user_name=get_current_user_name()
    )


@app.route("/about")
def about():
    """About page"""
    if not login_required():
        return redirect(url_for("login"))
    
    return render_template("about.html", user_name=get_current_user_name())


# =============================================================================
# ERROR HANDLERS
# =============================================================================
@app.errorhandler(404)
def not_found(e):
    return render_template("login.html"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("login.html"), 500


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 VentureVerse - Startup Success Predictor")
    print("=" * 50)
    init_db()
    print("✅ Database initialized")
    print("🌐 Starting server on http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
