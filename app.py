"""
VentureVerse – Flask Web Application (v5)
==========================================
Pages: Login, Signup, Predict, Charts, Insights, About
Database: SQLite for auth + prediction history
"""

from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import json
import numpy as np
import pandas as pd
import sqlite3
import hashlib
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ────────────────────────────────────────────────────────────────
# DATABASE
# ────────────────────────────────────────────────────────────────
DB_FILE = "ventureverse.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        prediction_score REAL,
        pred_label TEXT,
        input_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""")
    conn.commit()
    conn.close()


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


init_db()

# ────────────────────────────────────────────────────────────────
# LOAD MODEL & RESULTS
# ────────────────────────────────────────────────────────────────
MODEL_FILE = "ventureverse_model.joblib"
RESULTS_FILE = "model_results_summary.json"

model = joblib.load(MODEL_FILE)

try:
    with open(RESULTS_FILE, "r") as f:
        model_results = json.load(f)
except FileNotFoundError:
    model_results = None

# ────────────────────────────────────────────────────────────────
# OPTIONS
# ────────────────────────────────────────────────────────────────
CATEGORIES = ["biotech", "consulting", "ecommerce", "enterprise",
              "games_video", "mobile", "software", "web", "advertising", "other"]
STATES = ["CA", "NY", "MA", "TX", "WA", "CO", "IL", "FL", "other"]


# ────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────
def build_input_df(form: dict) -> pd.DataFrame:
    funding_total    = float(form.get("funding_total_usd", 0))
    funding_rounds   = int(form.get("funding_rounds", 1))
    relationships    = int(form.get("relationships", 0))
    milestones       = int(form.get("milestones", 0))
    avg_participants = float(form.get("avg_participants", 1.0))
    age_first_fund   = float(form.get("age_first_funding_year", 0))
    age_last_fund    = float(form.get("age_last_funding_year", 0))
    age_first_mile   = float(form.get("age_first_milestone_year", 0)) or np.nan
    age_last_mile    = float(form.get("age_last_milestone_year", 0)) or np.nan
    has_vc    = int(form.get("has_VC", 0))
    has_angel = int(form.get("has_angel", 0))
    has_rA    = int(form.get("has_roundA", 0))
    has_rB    = int(form.get("has_roundB", 0))
    has_rC    = int(form.get("has_roundC", 0))
    has_rD    = int(form.get("has_roundD", 0))
    is_top    = int(form.get("is_top500", 0))
    category  = form.get("category_code", "other")
    state     = form.get("state_code", "other")

    funding_duration      = max(age_last_fund - age_first_fund, 0)
    safe_rounds           = funding_rounds if funding_rounds > 0 else 1
    avg_funding_per_round = funding_total / safe_rounds
    log_funding           = np.log1p(funding_total)

    row = {
        "age_first_funding_year": age_first_fund, "age_last_funding_year": age_last_fund,
        "age_first_milestone_year": age_first_mile, "age_last_milestone_year": age_last_mile,
        "relationships": relationships, "funding_rounds": funding_rounds,
        "funding_total_usd": funding_total, "milestones": milestones,
        "avg_participants": avg_participants, "funding_duration": funding_duration,
        "avg_funding_per_round": avg_funding_per_round, "log_funding": log_funding,
        "has_VC": has_vc, "has_angel": has_angel,
        "has_roundA": has_rA, "has_roundB": has_rB,
        "has_roundC": has_rC, "has_roundD": has_rD,
        "is_top500": is_top, "category_code": category, "state_code": state,
    }
    return pd.DataFrame([row])


def compute_risk_breakdown(form: dict) -> list:
    factors = []
    funding = float(form.get("funding_total_usd", 0))
    if funding >= 10_000_000:   factors.append({"factor": "Total Funding", "score": 90, "status": "strong"})
    elif funding >= 2_000_000:  factors.append({"factor": "Total Funding", "score": 65, "status": "moderate"})
    elif funding >= 500_000:    factors.append({"factor": "Total Funding", "score": 40, "status": "moderate"})
    else:                       factors.append({"factor": "Total Funding", "score": 15, "status": "weak"})

    rounds = int(form.get("funding_rounds", 0))
    if rounds >= 4:   factors.append({"factor": "Funding Rounds", "score": 85, "status": "strong"})
    elif rounds >= 2: factors.append({"factor": "Funding Rounds", "score": 55, "status": "moderate"})
    else:             factors.append({"factor": "Funding Rounds", "score": 20, "status": "weak"})

    has_vc = int(form.get("has_VC", 0))
    is_top = int(form.get("is_top500", 0))
    if is_top:                        factors.append({"factor": "Investor Quality", "score": 95, "status": "strong"})
    elif has_vc:                      factors.append({"factor": "Investor Quality", "score": 70, "status": "strong"})
    elif int(form.get("has_angel",0)):factors.append({"factor": "Investor Quality", "score": 45, "status": "moderate"})
    else:                             factors.append({"factor": "Investor Quality", "score": 10, "status": "weak"})

    rels = int(form.get("relationships", 0))
    if rels >= 10:  factors.append({"factor": "Network Strength", "score": 85, "status": "strong"})
    elif rels >= 4: factors.append({"factor": "Network Strength", "score": 55, "status": "moderate"})
    else:           factors.append({"factor": "Network Strength", "score": 20, "status": "weak"})

    miles = int(form.get("milestones", 0))
    if miles >= 3:  factors.append({"factor": "Early Traction", "score": 80, "status": "strong"})
    elif miles >= 1:factors.append({"factor": "Early Traction", "score": 50, "status": "moderate"})
    else:           factors.append({"factor": "Early Traction", "score": 10, "status": "weak"})

    state = form.get("state_code", "other")
    if state == "CA":                  factors.append({"factor": "Location", "score": 85, "status": "strong"})
    elif state in ("NY", "MA", "WA"):  factors.append({"factor": "Location", "score": 65, "status": "moderate"})
    else:                              factors.append({"factor": "Location", "score": 35, "status": "moderate"})

    return factors


def generate_insights(form: dict, prediction: float, pred_label: str, risk_factors: list) -> list:
    """Generate human-readable insight cards based on the prediction."""
    insights = []

    # Overall verdict
    if prediction >= 75:
        insights.append({
            "title": "Strong Success Indicators",
            "icon": "&#9733;",
            "type": "positive",
            "text": f"This startup profile shows a {prediction}% success probability, significantly above the 50% threshold. The combination of factors suggests strong market readiness."
        })
    elif prediction >= 50:
        insights.append({
            "title": "Moderate Success Potential",
            "icon": "&#9888;",
            "type": "neutral",
            "text": f"At {prediction}%, this startup is above the success threshold but has room for improvement. Strengthening weaker factors could push probability higher."
        })
    else:
        insights.append({
            "title": "High Risk Profile",
            "icon": "&#9888;",
            "type": "negative",
            "text": f"With {prediction}% success probability, this profile indicates elevated failure risk. Key areas need attention before proceeding."
        })

    # Funding insight
    funding = float(form.get("funding_total_usd", 0))
    rounds = int(form.get("funding_rounds", 0))
    if funding >= 5_000_000 and rounds >= 3:
        insights.append({
            "title": "Funding Strength",
            "icon": "&#128176;",
            "type": "positive",
            "text": f"Total funding of ${funding:,.0f} across {rounds} rounds shows strong investor confidence. Multiple rounds indicate repeated due diligence validation."
        })
    elif funding < 500_000:
        insights.append({
            "title": "Funding Gap",
            "icon": "&#128176;",
            "type": "negative",
            "text": f"Total funding of ${funding:,.0f} is below the typical threshold for acquired startups. Historically, startups with under $500K in funding have significantly higher failure rates."
        })
    else:
        insights.append({
            "title": "Moderate Funding",
            "icon": "&#128176;",
            "type": "neutral",
            "text": f"Total funding of ${funding:,.0f} across {rounds} round(s) is moderate. Consider pursuing additional rounds to strengthen the funding profile."
        })

    # Investor type insight
    has_vc = int(form.get("has_VC", 0))
    has_angel = int(form.get("has_angel", 0))
    is_top = int(form.get("is_top500", 0))
    if is_top:
        insights.append({
            "title": "Elite Investor Backing",
            "icon": "&#127942;",
            "type": "positive",
            "text": "Backed by a Top-500 ranked VC firm. This is one of the strongest success signals — elite VCs provide not just capital but strategic guidance, network access, and credibility."
        })
    elif has_vc and has_angel:
        insights.append({
            "title": "Diversified Investor Base",
            "icon": "&#127942;",
            "type": "positive",
            "text": "Having both VC and angel investors provides a balanced funding structure. Angel investors offer early validation while VCs bring institutional support."
        })
    elif not has_vc and not has_angel:
        insights.append({
            "title": "No Institutional Backing",
            "icon": "&#127942;",
            "type": "negative",
            "text": "No VC or angel investor backing detected. Startups without institutional investors historically face lower acquisition rates. Consider seeking angel or VC funding."
        })

    # Network insight
    rels = int(form.get("relationships", 0))
    if rels >= 8:
        insights.append({
            "title": "Strong Network",
            "icon": "&#128101;",
            "type": "positive",
            "text": f"With {rels} key connections, this startup has a robust network of advisors, board members, and co-founders. Strong networks correlate with better mentorship and deal flow."
        })
    elif rels <= 2:
        insights.append({
            "title": "Limited Network",
            "icon": "&#128101;",
            "type": "negative",
            "text": f"Only {rels} key connection(s) linked to this startup. Building a stronger advisory board and expanding co-founder relationships could significantly improve outcomes."
        })

    # Milestone insight
    miles = int(form.get("milestones", 0))
    if miles >= 3:
        insights.append({
            "title": "Strong Early Traction",
            "icon": "&#127937;",
            "type": "positive",
            "text": f"{miles} milestones achieved — product launches, partnerships, or press coverage. Multiple milestones demonstrate execution capability and market validation."
        })
    elif miles == 0:
        insights.append({
            "title": "No Milestones Recorded",
            "icon": "&#127937;",
            "type": "negative",
            "text": "No milestones achieved yet. Milestones like product demos, partnerships, or media coverage are important early traction signals that investors look for."
        })

    # Location insight
    state = form.get("state_code", "other")
    state_names = {"CA": "California", "NY": "New York", "MA": "Massachusetts", "TX": "Texas", "WA": "Washington"}
    if state == "CA":
        insights.append({
            "title": "Silicon Valley Advantage",
            "icon": "&#128205;",
            "type": "positive",
            "text": "Based in California — home to the world's largest startup ecosystem. CA-based startups benefit from proximity to investors, talent, and potential acquirers."
        })
    elif state in ("NY", "MA", "WA"):
        insights.append({
            "title": f"{state_names.get(state, state)} Ecosystem",
            "icon": "&#128205;",
            "type": "neutral",
            "text": f"Located in {state_names.get(state, state)}, a strong secondary startup hub. While not as dense as California, this location offers solid ecosystem support."
        })
    elif state == "other":
        insights.append({
            "title": "Non-Hub Location",
            "icon": "&#128205;",
            "type": "negative",
            "text": "Located outside major US startup hubs. While not a dealbreaker, startups outside CA/NY/MA/WA historically face slightly lower acquisition rates due to reduced network effects."
        })

    # Recommendations
    weak_factors = [rf for rf in risk_factors if rf["status"] == "weak"]
    if weak_factors:
        weak_names = ", ".join([wf["factor"] for wf in weak_factors])
        insights.append({
            "title": "Key Recommendations",
            "icon": "&#128161;",
            "type": "action",
            "text": f"Priority areas for improvement: {weak_names}. Addressing these weak factors could materially increase success probability. Focus on the lowest-scoring factor first for maximum impact."
        })

    return insights


def get_model_comparison() -> dict:
    if not model_results or "all_model_results" not in model_results:
        return None
    names, roc_aucs, accuracies, f1s = [], [], [], []
    for m in model_results["all_model_results"]:
        names.append(m["name"])
        roc_aucs.append(round(m.get("cv_roc_auc_mean", 0) * 100, 1))
        accuracies.append(round(m.get("cv_accuracy_mean", 0) * 100, 1))
        f1s.append(round(m.get("cv_f1_mean", 0) * 100, 1))
    return {"names": names, "roc_aucs": roc_aucs, "accuracies": accuracies, "f1s": f1s,
            "winner": model_results.get("winner", "")}


def get_prediction_history(user_id: int) -> list:
    """Get last 10 predictions for the insights page."""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT prediction_score, pred_label, input_data, created_at FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 10", (user_id,))
        rows = c.fetchall()
        conn.close()
        return [{"score": r[0], "label": r[1], "data": json.loads(r[2]) if r[2] else {}, "date": r[3]} for r in rows]
    except Exception:
        return []


# ────────────────────────────────────────────────────────────────
# AUTH ROUTES
# ────────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, full_name, password_hash FROM users WHERE email=?", (email,))
        user = c.fetchone()
        conn.close()
        if user and user[2] == hash_password(password):
            session["user_id"] = user[0]
            session["user_name"] = user[1]
            session["user_email"] = email
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid email or password.")
    return render_template("login.html", error=None)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if not full_name or not email or not password:
            return render_template("signup.html", error="All fields are required.")
        if password != confirm:
            return render_template("signup.html", error="Passwords do not match.")
        if len(password) < 6:
            return render_template("signup.html", error="Password must be at least 6 characters.")
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("INSERT INTO users (full_name, email, password_hash) VALUES (?,?,?)",
                      (full_name, email, hash_password(password)))
            conn.commit()
            uid = c.lastrowid
            conn.close()
            session["user_id"] = uid
            session["user_name"] = full_name
            session["user_email"] = email
            return redirect(url_for("home"))
        except sqlite3.IntegrityError:
            return render_template("signup.html", error="Email already registered.")
    return render_template("signup.html", error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ────────────────────────────────────────────────────────────────
# PAGE 1: PREDICT (home)
# ────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("index.html",
        prediction=None, pred_label=None, error=None,
        categories=CATEGORIES, states=STATES, form_data={},
        risk_factors=None, user_name=session.get("user_name", ""),
        active_page="predict")


@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return redirect(url_for("login"))
    try:
        form_data = {k: request.form.get(k, "") for k in request.form}
        input_df = build_input_df(form_data)
        proba = model.predict_proba(input_df)[0][1]
        prediction_percent = round(proba * 100, 2)
        pred_label = "Success" if proba >= 0.5 else "Failure"
        risk_factors = compute_risk_breakdown(form_data)

        # Store in session for charts/insights pages
        session["last_prediction"] = prediction_percent
        session["last_pred_label"] = pred_label
        session["last_form_data"] = form_data
        session["last_risk_factors"] = risk_factors

        # Save to DB
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("INSERT INTO predictions (user_id, prediction_score, pred_label, input_data) VALUES (?,?,?,?)",
                      (session["user_id"], prediction_percent, pred_label, json.dumps(form_data)))
            conn.commit()
            conn.close()
        except Exception:
            pass

        return render_template("index.html",
            prediction=prediction_percent, pred_label=pred_label, error=None,
            categories=CATEGORIES, states=STATES, form_data=form_data,
            risk_factors=risk_factors, user_name=session.get("user_name", ""),
            active_page="predict")

    except Exception as e:
        form_data = {k: request.form.get(k, "") for k in request.form}
        return render_template("index.html",
            prediction=None, pred_label=None, error=str(e),
            categories=CATEGORIES, states=STATES, form_data=form_data,
            risk_factors=None, user_name=session.get("user_name", ""),
            active_page="predict")


# ────────────────────────────────────────────────────────────────
# PAGE 2: CHARTS (graphs only)
# ────────────────────────────────────────────────────────────────
@app.route("/charts")
def charts():
    if "user_id" not in session:
        return redirect(url_for("login"))

    prediction = session.get("last_prediction")
    pred_label = session.get("last_pred_label")
    risk_factors = session.get("last_risk_factors")
    model_comparison = get_model_comparison()

    return render_template("charts.html",
        prediction=prediction, pred_label=pred_label,
        risk_factors=risk_factors, model_comparison=model_comparison,
        user_name=session.get("user_name", ""),
        active_page="charts")


# ────────────────────────────────────────────────────────────────
# PAGE 3: INSIGHTS
# ────────────────────────────────────────────────────────────────
@app.route("/insights")
def insights():
    if "user_id" not in session:
        return redirect(url_for("login"))

    prediction = session.get("last_prediction")
    pred_label = session.get("last_pred_label")
    form_data = session.get("last_form_data", {})
    risk_factors = session.get("last_risk_factors", [])

    insight_cards = []
    if prediction is not None:
        insight_cards = generate_insights(form_data, prediction, pred_label, risk_factors)

    history = get_prediction_history(session["user_id"])

    return render_template("insights.html",
        prediction=prediction, pred_label=pred_label,
        insights=insight_cards, history=history,
        user_name=session.get("user_name", ""),
        active_page="insights")


# ────────────────────────────────────────────────────────────────
# PAGE 4: ABOUT
# ────────────────────────────────────────────────────────────────
@app.route("/about")
def about():
    return render_template("about.html",
        user_name=session.get("user_name", ""),
        active_page="about")


if __name__ == "__main__":
    app.run(debug=True)
