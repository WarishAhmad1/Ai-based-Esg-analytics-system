#!/usr/bin/env python3

import os
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

from IPython.display import Markdown, display
from contextlib import contextmanager

df = pd.read_csv("company_esg_financial_dataset.csv")

class NotebookColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def metric(self, label, value):
        display(Markdown(f"**{label}:** {value}"))

    def markdown(self, text):
        display(Markdown(text))

    def write(self, text):
        display(text)

    def plotly_chart(self, fig, use_container_width=True):
        fig.show()


class NotebookST:
    def set_page_config(self, **kwargs):
        return None

    def title(self, text):
        display(Markdown(f"## {text}"))

    def subheader(self, text):
        display(Markdown(f"### {text}"))

    def caption(self, text):
        display(Markdown(f"*{text}*"))

    def markdown(self, text):
        display(Markdown(text))

    def write(self, text):
        display(text)

    def error(self, text):
        raise RuntimeError(text)

    def stop(self):
        raise SystemExit

    def plotly_chart(self, fig, use_container_width=True):
        fig.show()

    def columns(self, spec):
        if isinstance(spec, int):
            count = spec
        elif isinstance(spec, (list, tuple)):
            count = len(spec)
        else:
            count = 1
        return [NotebookColumn() for _ in range(max(1, count))]

    @contextmanager
    def expander(self, label):
        display(Markdown(f"#### {label}"))
        yield

    def dataframe(self, df, use_container_width=True):
        display(df)


st = NotebookST()



st.set_page_config(page_title="Risk and Alert Analysis", layout="wide")
st.title("Risk and Alert Analysis Dashboard")


def load_data() -> pd.DataFrame:
    current_file = Path(__file__).resolve() if '__file__' in globals() else (Path.cwd() / 'Risk_alert.ipynb').resolve()
    csv_candidates = [
        current_file.parents[2] / "company_esg_financial_dataset.csv",
        current_file.parents[1] / "company_esg_financial_dataset.csv",
        current_file.parent / "company_esg_financial_dataset.csv",
    ]

    for csv_path in csv_candidates:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            needed = [
                "CompanyName",
                "Year",
                "ESG_Overall",
                "CarbonEmissions",
                "Revenue",
                "ProfitMargin",
            ]
            df = df.dropna(subset=needed).copy()
            for col in ["Year", "ESG_Overall", "CarbonEmissions", "Revenue", "ProfitMargin"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["Year", "ESG_Overall", "CarbonEmissions", "Revenue", "ProfitMargin"])
            return df

    raise FileNotFoundError("company_esg_financial_dataset.csv not found.")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "Flask" / "templates"
DB_DIR = BASE_DIR / "instance"
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "users.db"
DATASET_PATH = BASE_DIR / "company_esg_financial_dataset.csv"

_DATA_CACHE = None
_DATA_MTIME = None

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH.as_posix()}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    company = db.Column(db.String(150), nullable=True)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


def ensure_database_schema():
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(150) NOT NULL UNIQUE,
                email VARCHAR(150) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL
            )
            """
        )

        cursor.execute("PRAGMA table_info(user)")
        columns = {row[1] for row in cursor.fetchall()}

        if "company" not in columns:
            cursor.execute("ALTER TABLE user ADD COLUMN company VARCHAR(150)")

        if "created_at" not in columns:
            cursor.execute("ALTER TABLE user ADD COLUMN created_at DATETIME")
            cursor.execute(
                "UPDATE user SET created_at = ? WHERE created_at IS NULL",
                (datetime.utcnow().isoformat(timespec="seconds"),),
            )

        connection.commit()


with app.app_context():
    ensure_database_schema()
    db.create_all()


def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return db.session.get(User, user_id)


def require_login():
    if not get_current_user():
        flash("Please log in to access your dashboard.", "danger")
        return False
    return True


def validate_password(password):
    if len(password) < 6:
        return "Password must be at least 6 characters long."
    return None


def categorize_esg(score):
    if score >= 75:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def categorize_risk(score):
    if score >= 70:
        return "High Risk"
    if score >= 40:
        return "Medium Risk"
    return "Low Risk"


def governance_risk_level(score):
    if score >= 75:
        return "Low Risk"
    if score >= 55:
        return "Medium Risk"
    return "High Risk"


def _theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font=dict(color="#e5e7eb"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def plot_html(fig):
    return _theme(fig).to_html(full_html=False, include_plotlyjs=False)


def load_dataset():
    global _DATA_CACHE, _DATA_MTIME

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    mtime = DATASET_PATH.stat().st_mtime
    if _DATA_CACHE is not None and _DATA_MTIME == mtime:
        return _DATA_CACHE.copy()

    df = pd.read_csv(DATASET_PATH)
    numeric_columns = [
        "Year",
        "Revenue",
        "ProfitMargin",
        "MarketCap",
        "GrowthRate",
        "ESG_Overall",
        "ESG_Environmental",
        "ESG_Social",
        "ESG_Governance",
        "CarbonEmissions",
        "WaterUsage",
        "EnergyConsumption",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["CompanyName", "Year", "ESG_Overall"]).copy()
    _DATA_CACHE = df
    _DATA_MTIME = mtime
    return df.copy()


def latest_company_snapshot(df):
    latest = (
        df.sort_values(["CompanyName", "Year"]).groupby("CompanyName", as_index=False).tail(1)
    )
    return latest.copy()


# --------------------------
# Dashboard Chart Functions
# --------------------------
def overall_esg_dashboard(df):
    latest = latest_company_snapshot(df)
    perf = latest[["CompanyName", "ESG_Overall"]].dropna().copy()
    perf["Category"] = perf["ESG_Overall"].apply(categorize_esg)
    perf = perf.sort_values("ESG_Overall", ascending=False)

    fig1 = px.bar(
        perf,
        x="CompanyName",
        y="ESG_Overall",
        color="Category",
        color_discrete_map={"High": "#22c55e", "Medium": "#f59e0b", "Low": "#ef4444"},
        title="Company-wise ESG Score",
    )

    category_dist = perf["Category"].value_counts().rename_axis("Category").reset_index(name="Count")
    fig2 = px.pie(
        category_dist,
        names="Category",
        values="Count",
        hole=0.45,
        color="Category",
        color_discrete_map={"High": "#22c55e", "Medium": "#f59e0b", "Low": "#ef4444"},
        title="ESG Category Distribution",
    )

    avg_score = perf["ESG_Overall"].mean()
    top_company = perf.iloc[0]
    low_company = perf.iloc[-1]

    fig3 = go.Figure()
    fig3.add_trace(
        go.Indicator(
            mode="number",
            value=avg_score,
            title={"text": "Average ESG Score"},
            number={"valueformat": ".2f"},
            domain={"x": [0.0, 0.33], "y": [0.0, 1.0]},
        )
    )
    fig3.add_trace(
        go.Indicator(
            mode="number",
            value=top_company["ESG_Overall"],
            title={"text": f"Top: {top_company['CompanyName']}"},
            number={"valueformat": ".1f"},
            domain={"x": [0.34, 0.66], "y": [0.0, 1.0]},
        )
    )
    fig3.add_trace(
        go.Indicator(
            mode="number",
            value=low_company["ESG_Overall"],
            title={"text": f"Lower: {low_company['CompanyName']}"},
            number={"valueformat": ".1f"},
            domain={"x": [0.67, 1.0], "y": [0.0, 1.0]},
        )
    )
    fig3.update_layout(title="KPI Overview")

    return [
        {"title": "A. Company-wise ESG Score", "graph": plot_html(fig1)},
        {"title": "B. ESG Category Distribution", "graph": plot_html(fig2)},
        {"title": "C. KPI Overview", "graph": plot_html(fig3)},
    ]


def environmental_dashboard(df):
    env_df = df.dropna(subset=["CarbonEmissions", "EnergyConsumption", "ESG_Overall"]).copy()

    trend = env_df.groupby("Year", as_index=False)["CarbonEmissions"].mean().sort_values("Year")
    fig1 = px.line(
        trend,
        x="Year",
        y="CarbonEmissions",
        markers=True,
        title="Emission Trend Over Time",
    )

    latest = latest_company_snapshot(env_df)
    compare = latest[["CompanyName", "CarbonEmissions", "EnergyConsumption"]].copy()
    fig2 = px.bar(
        compare,
        x="CompanyName",
        y=["CarbonEmissions", "EnergyConsumption"],
        barmode="group",
        title="Company-wise Emission / Energy Use",
    )

    scatter = latest[["CompanyName", "CarbonEmissions", "ESG_Overall", "EnergyConsumption"]].copy()
    fig3 = px.scatter(
        scatter,
        x="CarbonEmissions",
        y="ESG_Overall",
        size="EnergyConsumption",
        color="CompanyName",
        title="Emission vs ESG Score",
    )

    return [
        {"title": "A. Emission Trend Over Time", "graph": plot_html(fig1)},
        {"title": "B. Company-wise Emission / Energy Use", "graph": plot_html(fig2)},
        {"title": "C. Emission vs ESG Score", "graph": plot_html(fig3)},
    ]


def social_dashboard(df):
    social_df = df.dropna(subset=["ESG_Social", "ESG_Governance"]).copy()
    latest = latest_company_snapshot(social_df)

    metrics = latest[["CompanyName", "ESG_Social", "ESG_Governance"]].copy()
    metrics["Customer Satisfaction"] = metrics["ESG_Social"].round(1)
    metrics["Engagement"] = (metrics["ESG_Social"] * 0.7 + metrics["ESG_Governance"] * 0.3).round(1)
    metrics["Employee Turnover (%)"] = (35 - metrics["ESG_Social"] * 0.3).clip(lower=5, upper=35).round(1)
    metrics["Diversity (%)"] = (metrics["ESG_Social"] * 0.6).clip(lower=20, upper=70).round(1)
    metrics["Inclusion (%)"] = (metrics["ESG_Governance"] * 0.5).clip(lower=20, upper=70).round(1)

    fig1 = px.bar(
        metrics,
        x="CompanyName",
        y=["Customer Satisfaction", "Engagement"],
        barmode="group",
        title="Customer Satisfaction / Engagement",
    )

    fig2 = px.bar(
        metrics,
        x="CompanyName",
        y="Employee Turnover (%)",
        color="Employee Turnover (%)",
        color_continuous_scale="Reds",
        title="Employee Turnover Distribution",
    )

    diversity_melt = metrics.melt(
        id_vars="CompanyName",
        value_vars=["Diversity (%)", "Inclusion (%)"],
        var_name="Metric",
        value_name="Percentage",
    )
    fig3 = px.bar(
        diversity_melt,
        x="CompanyName",
        y="Percentage",
        color="Metric",
        barmode="stack",
        title="Diversity / Inclusion Metrics",
    )

    return [
        {"title": "A. Customer Satisfaction / Engagement", "graph": plot_html(fig1)},
        {"title": "B. Employee Turnover Distribution", "graph": plot_html(fig2)},
        {"title": "C. Diversity / Inclusion Metrics", "graph": plot_html(fig3)},
    ]


def governance_dashboard(df):
    gov_df = df.dropna(subset=["ESG_Governance", "ESG_Overall", "ProfitMargin"]).copy()
    latest = latest_company_snapshot(gov_df)

    comp = latest[["CompanyName", "ESG_Governance", "ESG_Overall", "ProfitMargin"]].copy()
    comp["Board Compliance"] = (comp["ESG_Governance"] * 0.9 + comp["ESG_Overall"] * 0.1).clip(0, 100).round(1)
    comp["Audit Compliance"] = (comp["ESG_Governance"] * 0.75 + comp["ESG_Overall"] * 0.25).clip(0, 100).round(1)
    comp["Regulatory Compliance"] = (
        comp["ESG_Governance"] * 0.7 + (comp["ProfitMargin"] + 20) * 1.0
    ).clip(0, 100).round(1)
    comp["Risk Level"] = comp["ESG_Governance"].apply(governance_risk_level)

    fig1 = px.bar(
        comp.sort_values("ESG_Governance", ascending=False),
        x="CompanyName",
        y="ESG_Governance",
        color="Risk Level",
        color_discrete_map={"Low Risk": "#22c55e", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"},
        title="Governance Score Comparison",
    )

    heatmap_df = comp.set_index("CompanyName")[["Board Compliance", "Audit Compliance", "Regulatory Compliance"]]
    fig2 = px.imshow(
        heatmap_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title="Compliance vs Companies",
    )

    risk_dist = comp["Risk Level"].value_counts().rename_axis("Risk Level").reset_index(name="Count")
    fig3 = px.pie(
        risk_dist,
        names="Risk Level",
        values="Count",
        title="Governance Risk Levels",
        color="Risk Level",
        color_discrete_map={"Low Risk": "#22c55e", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"},
    )

    return [
        {"title": "A. Governance Score Comparison", "graph": plot_html(fig1)},
        {"title": "B. Compliance vs Companies", "graph": plot_html(fig2)},
        {"title": "C. Governance Risk Levels", "graph": plot_html(fig3)},
    ]


def risk_dashboard(df):
    risk_df = df.dropna(subset=["ESG_Overall", "CarbonEmissions", "Revenue", "ProfitMargin"]).copy()
    latest = latest_company_snapshot(risk_df)

    metrics = latest[["CompanyName", "ESG_Overall", "CarbonEmissions", "Revenue", "ProfitMargin"]].copy()
    metrics["EmissionIntensity"] = metrics["CarbonEmissions"] / metrics["Revenue"].replace(0, pd.NA)
    metrics["EmissionIntensity"] = metrics["EmissionIntensity"].fillna(metrics["EmissionIntensity"].median())

    metrics["ESG_Risk_Component"] = 100 - metrics["ESG_Overall"]
    metrics["Emission_Risk_Component"] = 100 * (
        (metrics["EmissionIntensity"] - metrics["EmissionIntensity"].min())
        / (metrics["EmissionIntensity"].max() - metrics["EmissionIntensity"].min() + 1e-9)
    )
    metrics["Profit_Risk_Component"] = (20 - metrics["ProfitMargin"]).clip(lower=0, upper=40) * 2.5
    metrics["Risk Score"] = (
        0.5 * metrics["ESG_Risk_Component"]
        + 0.35 * metrics["Emission_Risk_Component"]
        + 0.15 * metrics["Profit_Risk_Component"]
    ).clip(0, 100).round(1)
    metrics["Risk Category"] = metrics["Risk Score"].apply(categorize_risk)

    fig1 = px.bar(
        metrics.sort_values("Risk Score", ascending=False),
        x="CompanyName",
        y="Risk Score",
        color="Risk Category",
        color_discrete_map={"Low Risk": "#22c55e", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"},
        title="Risk Score by Company",
    )

    fig2 = px.scatter(
        metrics,
        x="Risk Score",
        y="ESG_Overall",
        size="EmissionIntensity",
        color="Risk Category",
        color_discrete_map={"Low Risk": "#22c55e", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"},
        hover_name="CompanyName",
        title="Risk vs ESG Score",
    )

    risk_counts = metrics["Risk Category"].value_counts().rename_axis("Risk Category").reset_index(name="Count")
    fig3 = px.pie(
        risk_counts,
        names="Risk Category",
        values="Count",
        hole=0.5,
        color="Risk Category",
        color_discrete_map={"Low Risk": "#22c55e", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"},
        title="Risk Categories",
    )

    return [
        {"title": "A. Risk Score by Company", "graph": plot_html(fig1)},
        {"title": "B. Risk vs ESG Score", "graph": plot_html(fig2)},
        {"title": "C. Risk Categories", "graph": plot_html(fig3)},
    ]


def trend_time_dashboard(df):
    trend_df = df.dropna(subset=["ESG_Overall"]).copy()

    overall = trend_df.groupby("Year", as_index=False)["ESG_Overall"].mean().sort_values("Year")
    fig1 = px.line(
        overall,
        x="Year",
        y="ESG_Overall",
        markers=True,
        title="ESG Score Trend Over Time",
    )

    company_trend = trend_df.groupby(["Year", "CompanyName"], as_index=False)["ESG_Overall"].mean()
    latest_ranking = (
        company_trend.sort_values(["CompanyName", "Year"]).groupby("CompanyName", as_index=False).tail(1)
    )
    top_companies = latest_ranking.nlargest(8, "ESG_Overall")["CompanyName"]
    company_top = company_trend[company_trend["CompanyName"].isin(top_companies)]

    fig2 = px.line(
        company_top,
        x="Year",
        y="ESG_Overall",
        color="CompanyName",
        markers=True,
        title="Company Comparison",
    )

    fig3 = px.area(
        company_top,
        x="Year",
        y="ESG_Overall",
        color="CompanyName",
        title="Growth Pattern",
    )

    return [
        {"title": "A. ESG Score Trend Over Time", "graph": plot_html(fig1)},
        {"title": "B. Company Comparison", "graph": plot_html(fig2)},
        {"title": "C. Growth Pattern", "graph": plot_html(fig3)},
    ]


DASHBOARD_BUILDERS = {
    "performance": ("Overall ESG Performance Dashboard", overall_esg_dashboard),
    "environmental": ("Environmental Analysis Dashboard", environmental_dashboard),
    "social": ("Social Analysis Dashboard", social_dashboard),
    "governance": ("Governance Analysis Dashboard", governance_dashboard),
    "risk": ("Risk & Alert Analysis Dashboard", risk_dashboard),
    "trend": ("Trend & Time Analysis Dashboard", trend_time_dashboard),
}


@app.route("/")
def home():
    active_modal = request.args.get("auth", "")
    user = get_current_user()
    platform_stats = {"companies": 500, "accuracy": 95, "reports": 1200, "watchlists": 42}
    return render_template("index.html", active_modal=active_modal, user=user, platform_stats=platform_stats)


@app.route("/about")
def about_page():
    user = get_current_user()
    return render_template("about.html", user=user)


@app.route("/features")
def features_page():
    user = get_current_user()
    return render_template("features.html", user=user)


@app.route("/contact")
def contact_page():
    user = get_current_user()
    return render_template("contact.html", user=user)


@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("username", "").strip()
    company = request.form.get("company", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not username or not email or not password or not confirm_password:
        flash("All registration fields are required.", "danger")
        return redirect(url_for("home", auth="register"))

    password_error = validate_password(password)
    if password_error:
        flash(password_error, "danger")
        return redirect(url_for("home", auth="register"))

    if password != confirm_password:
        flash("Passwords do not match.", "danger")
        return redirect(url_for("home", auth="register"))

    new_user = User(
        username=username,
        company=company or None,
        email=email,
        password=generate_password_hash(password),
    )

    try:
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("home", auth="login"))
    except IntegrityError:
        db.session.rollback()
        if User.query.filter_by(email=email).first():
            flash("This email is already registered.", "danger")
        else:
            flash("This username is already taken.", "danger")
        return redirect(url_for("home", auth="register"))
    except Exception:
        db.session.rollback()
        flash("Registration failed. Please try again.", "danger")
        return redirect(url_for("home", auth="register"))


@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    if not email or not password:
        flash("Email and password are required.", "danger")
        return redirect(url_for("home", auth="login"))

    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.password, password):
        session["user_id"] = user.id
        flash("Logged in successfully.", "success")
        return redirect(url_for("dashboard"))

    flash("Invalid email or password.", "danger")
    return redirect(url_for("home", auth="login"))


@app.route("/forgot-password", methods=["POST"])
def forgot_password():
    email = request.form.get("email", "").strip().lower()
    new_password = request.form.get("new_password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not email or not new_password or not confirm_password:
        flash("Please complete all password reset fields.", "danger")
        return redirect(url_for("home", auth="forgot"))

    password_error = validate_password(new_password)
    if password_error:
        flash(password_error, "danger")
        return redirect(url_for("home", auth="forgot"))

    if new_password != confirm_password:
        flash("Passwords do not match.", "danger")
        return redirect(url_for("home", auth="forgot"))

    user = User.query.filter_by(email=email).first()
    if not user:
        flash("No account found for that email.", "danger")
        return redirect(url_for("home", auth="forgot"))

    user.password = generate_password_hash(new_password)
    db.session.commit()
    flash("Password updated successfully. Please log in.", "success")
    return redirect(url_for("home", auth="login"))


@app.route("/dashboard")
def dashboard():
    if not require_login():
        return redirect(url_for("home", auth="login"))

    user = get_current_user()
    analytics_dashboards = [
        {"key": "performance", "label": "Overall ESG"},
        {"key": "environmental", "label": "Environmental"},
        {"key": "social", "label": "Social"},
        {"key": "governance", "label": "Governance"},
        {"key": "risk", "label": "Risk & Alerts"},
        {"key": "trend", "label": "Trend & Time"},
    ]

    esg_snapshot = {"environment": 82, "social": 74, "governance": 88}
    yearly_trend = [61, 67, 72, 78, 81]
    alerts = [
        "Carbon intensity improved 12% quarter-over-quarter.",
        "Supplier policy review is due this month.",
        "Board governance metrics remain above target.",
    ]

    return render_template(
        "dashboard.html",
        user=user,
        user_name=user.username,
        esg_snapshot=esg_snapshot,
        yearly_trend=yearly_trend,
        alerts=alerts,
        analytics_dashboards=analytics_dashboards,
    )


@app.route("/analytics/<dashboard_key>")
def open_analytics_dashboard(dashboard_key):
    if not require_login():
        return redirect(url_for("home", auth="login"))

    user = get_current_user()
    config = DASHBOARD_BUILDERS.get(dashboard_key)

    if not config:
        flash("Invalid dashboard selected.", "danger")
        return redirect(url_for("dashboard"))

    title, builder = config

    try:
        df = load_dataset()
        charts = builder(df)
    except Exception as error:
        flash(f"Dashboard load failed: {error}", "danger")
        return redirect(url_for("dashboard"))

    return render_template(
        "analytics_dashboard.html",
        user=user,
        dashboard_title=title,
        charts=charts,
    )


@app.route("/overview_dashboard")
def overview_dashboard():
    return redirect(url_for("open_analytics_dashboard", dashboard_key="performance"))


@app.route("/environment_dashboard")
def environment_dashboard_route():
    return redirect(url_for("open_analytics_dashboard", dashboard_key="environmental"))


@app.route("/social_dashboard")
def social_dashboard_route():
    return redirect(url_for("open_analytics_dashboard", dashboard_key="social"))


@app.route("/governance_dashboard")
def governance_dashboard_route():
    return redirect(url_for("open_analytics_dashboard", dashboard_key="governance"))


@app.route("/risk_alert_dashboard")
def risk_alert_dashboard_route():
    return redirect(url_for("open_analytics_dashboard", dashboard_key="risk"))


@app.route("/trend_time_dashboard")
def trend_time_dashboard_route():
    return redirect(url_for("open_analytics_dashboard", dashboard_key="trend"))


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Logged out successfully.", "success")
    return redirect(url_for("home"))

#graphs1 🔴 1. Risk Score by Company (Bar Chart)
def Risk_Score_by_Company():

    df = df.dropna(subset=["CompanyName","Year","ESG_Overall","CarbonEmissions","Revenue","ProfitMargin"])

    latest = df.sort_values(["CompanyName","Year"]).groupby("CompanyName").tail(1)

    latest["EmissionIntensity"] = latest["CarbonEmissions"] / latest["Revenue"]

    latest["Risk Score"] = (
        0.5*(100-latest["ESG_Overall"]) +
        0.35*((latest["EmissionIntensity"]-latest["EmissionIntensity"].min()) /
        (latest["EmissionIntensity"].max()-latest["EmissionIntensity"].min()+1e-9))*100 +
        0.15*((20-latest["ProfitMargin"]).clip(0,40)*2.5)
    ).clip(0,100)

    fig1 = px.bar(
        latest.sort_values("Risk Score", ascending=False),
        x="CompanyName",
        y="Risk Score",
        title="Risk Score by Company"
    )
    graph1_html= pio.to_html(fig1 ,full_html=False)
    return graph1_html

def Risk_vs_ESG():

    df = pd.read_csv("company_esg_financial_dataset.csv")

    df = df.dropna(subset=["CompanyName","Year","ESG_Overall","CarbonEmissions","Revenue","ProfitMargin"])

    latest = df.sort_values(["CompanyName","Year"]).groupby("CompanyName").tail(1)

    latest["EmissionIntensity"] = latest["CarbonEmissions"] / latest["Revenue"]

    latest["Risk Score"] = (
        0.5*(100-latest["ESG_Overall"]) +
        0.35*((latest["EmissionIntensity"]-latest["EmissionIntensity"].min()) /
        (latest["EmissionIntensity"].max()-latest["EmissionIntensity"].min()+1e-9))*100 +
        0.15*((20-latest["ProfitMargin"]).clip(0,40)*2.5)
    ).clip(0,100)

    fig2 = px.scatter(
        latest,
        x="Risk Score",
        y="ESG_Overall",
        size="EmissionIntensity",
        hover_name="CompanyName",
        title="Risk vs ESG Score"
    )

    graph2_html = pio.to_html(fig2, full_html=False)
    return graph2_html

def Risk_Category_Distribution():
    df = pd.read_csv("company_esg_financial_dataset.csv")

    df = df.dropna(subset=["CompanyName","Year","ESG_Overall","CarbonEmissions","Revenue","ProfitMargin"])

    latest = df.sort_values(["CompanyName","Year"]).groupby("CompanyName").tail(1)

    latest["EmissionIntensity"] = latest["CarbonEmissions"] / latest["Revenue"]

    latest["Risk Score"] = (
        0.5*(100-latest["ESG_Overall"]) +
        0.35*((latest["EmissionIntensity"]-latest["EmissionIntensity"].min()) /
        (latest["EmissionIntensity"].max()-latest["EmissionIntensity"].min()+1e-9))*100 +
        0.15*((20-latest["ProfitMargin"]).clip(0,40)*2.5)
    ).clip(0,100)

    def categorize(score):
        if score >= 70:
            return "High Risk"
        elif score >= 40:
            return "Medium Risk"
        else:
            return "Low Risk"

    latest["Risk Category"] = latest["Risk Score"].apply(categorize)

    counts = latest["Risk Category"].value_counts().reset_index()
    counts.columns = ["Risk Category", "Count"]

    fig3= px.pie(
        counts,
        names="Risk Category",
        values="Count",
        hole=0.5,
        title="Risk Category Distribution"
    )
    graph3_html = pio.to_html(fig3, full_html=False)
    return graph3_html

#analysis graphs routes for risk and alert dashboard
@app.route("/risk_alert_graphs")
def risk_alert_graphs():
    graph1 = Risk_Score_by_Company()
    graph2 = Risk_vs_ESG()
    graph3 = Risk_Category_Distribution()

    return render_template("risk_alert.html", graph1=graph1, graph2=graph2, graph3=graph3)

#graphs2 🔴 2. Emission Trend Over Time
def Emission_Trend_Over_Time():
    df = pd.read_csv("company_esg_financial_dataset.csv")

    df = df.dropna(subset=["Year", "CarbonEmissions"])

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["CarbonEmissions"] = pd.to_numeric(df["CarbonEmissions"], errors="coerce")

    df = df.dropna()

    emission_trend = df.groupby("Year", as_index=False)["CarbonEmissions"].mean()

    fig1 = px.line(
    emission_trend,
    x="Year",
    y="CarbonEmissions",
    markers=True,
    title="Average Carbon Emission Trend"
    )
    graph1_html = pio.to_html(fig1, full_html=False)
    return graph1_html

def Company_wise_Emission_vs_Energy():
    latest = df.sort_values(["CompanyName","Year"]).groupby("CompanyName").tail(1)

    fig2 = px.bar(
    latest,
    x="CompanyName",
    y=["CarbonEmissions", "EnergyConsumption"],
    barmode="group",
    title="Company-wise Emission and Energy Use"
    )
    graph2_html = pio.to_html(fig2, full_html=False)
    return graph2_html

def Emission_vs_ESG_Score():
    latest = df.sort_values(["CompanyName","Year"]).groupby("CompanyName").tail(1)

    fig3 = px.scatter(
    latest,
    x="CarbonEmissions",
    y="ESG_Overall",
    size="EnergyConsumption",
    hover_name="CompanyName",
    title="Emission vs ESG Score"
    )
    graph3_html = pio.to_html(fig3, full_html=False)
    return graph3_html

#analysis graphs routes for environmental dashboard
@app.route("/environment_graphs")
def environment_graphs():
    graph1 = Emission_Trend_Over_Time()
    graph2 = Company_wise_Emission_vs_Energy()
    graph3 = Emission_vs_ESG_Score()

    return render_template("environment.html", graph1=graph1, graph2=graph2, graph3=graph3)

#graphs3 Governance Analysis Page
def Governance_Score_by_Company():
    df = pd.read_csv("company_esg_financial_dataset.csv")

    df = df.dropna(subset=["CompanyName","Year","ESG_Governance"])

    df["ESG_Governance"] = pd.to_numeric(df["ESG_Governance"], errors="coerce")

    latest = df.sort_values(["CompanyName","Year"]).groupby("CompanyName").tail(1)

def risk_level(score):
    if score >= 75:
        return "Low Risk"
    elif score >= 55:
        return "Medium Risk"
    else:
        return "High Risk"

    latest["Risk Level"] = latest["ESG_Governance"].apply(risk_level)

    fig1 = px.bar(
    latest.sort_values("ESG_Governance", ascending=False),
    x="CompanyName",
    y="ESG_Governance",
    color="Risk Level",
    title="Governance Score by Company"
)
    graph1_html = pio.to_html(fig, full_html=False)
    return graph1_html

def Compliance_Heatmap():
    latest = df.sort_values(["CompanyName","Year"]).groupby("CompanyName").tail(1)

    latest["Board Compliance"] = (latest["ESG_Governance"] * 0.9 + latest["ESG_Overall"] * 0.1).clip(0, 100)
    latest["Audit Compliance"] = (latest["ESG_Governance"] * 0.75 + latest["ESG_Overall"] * 0.25).clip(0, 100)
    latest["Regulatory Compliance"] = (latest["ESG_Governance"] * 0.7 + (latest["ProfitMargin"] + 20) * 1.0).clip(0, 100)

    heatmap_data = latest.set_index("CompanyName")[["Board Compliance", "Audit Compliance", "Regulatory Compliance"]]

    fig2 = px.imshow(
    heatmap_data,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdYlGn",
    title="Compliance Heatmap"
)
    graph2_html = pio.to_html(fig2, full_html=False)
    return graph2_html

def Governance_Risk_Level_Distribution():
    latest = df.sort_values(["CompanyName","Year"]).groupby("CompanyName").tail(1)

    latest["Risk Level"] = latest["ESG_Governance"].apply(risk_level)

    risk_counts = latest["Risk Level"].value_counts().rename_axis("Risk Level").reset_index(name="Count")

    fig3 = px.pie(
    risk_counts,
    names="Risk Level",
    values="Count",
    title="Governance Risk Level Distribution"
)
    graph3_html = pio.to_html(fig3, full_html=False)
    return graph3_html

#analysis graphs routes for governance dashboard
@app.route("/governance_graphs")
def governance_graphs():
    graph1 = Governance_Score_by_Company()
    graph2 = Compliance_Heatmap()
    graph3 = Governance_Risk_Level_Distribution()

    return render_template("governance.html", graph1=graph1, graph2=graph2, graph3=graph3)

#graphs4 Social Analysis Page
def Customer_Satisfaction_and_Engagement():
    df = pd.read_csv("company_esg_financial_dataset.csv")

    df = df.dropna(subset=["CompanyName","Year","ESG_Social","ESG_Governance"])

    latest = df.sort_values(["CompanyName","Year"]).groupby("CompanyName").tail(1)

    latest["Customer Satisfaction"] = latest["ESG_Social"].round(1)
    latest["Engagement"] = (latest["ESG_Social"] * 0.7 + latest["ESG_Governance"] * 0.3).round(1)

    fig1 = px.bar(
    latest,
    x="CompanyName",
    y=["Customer Satisfaction", "Engagement"],
    barmode="group",
    title="Customer Satisfaction and Engagement"
)
    graph1_html = pio.to_html(fig1, full_html=False)
    return graph1_html

def Employee_Turnover_Distribution():
    df = pd.read_csv("company_esg_financial_dataset.csv")

    df = df.dropna(subset=["CompanyName","Year","ESG_Social"])

    latest = df.sort_values(["CompanyName","Year"]).groupby("CompanyName").tail(1)

    latest["Employee Turnover (%)"] = (35 - latest["ESG_Social"] * 0.3).clip(lower=5, upper=35).round(1)

    fig2 = px.bar(
    latest,
    x="CompanyName",
    y="Employee Turnover (%)",
    color="Employee Turnover (%)",
    color_continuous_scale="Reds",
    title="Employee Turnover Distribution"
)
    graph2_html = pio.to_html(fig2, full_html=False)
    return graph2_html

def Diversity_and_Inclusion():
    df = pd.read_csv("company_esg_financial_dataset.csv")

    df = df.dropna(subset=["CompanyName","Year","ESG_Social","ESG_Governance"])

    latest = df.sort_values(["CompanyName","Year"]).groupby("CompanyName").tail(1)

    latest["Diversity (%)"] = (latest["ESG_Social"] * 0.6).clip(lower=20, upper=70).round(1)
    latest["Inclusion (%)"] = (latest["ESG_Governance"] * 0.5).clip(lower=20, upper=70).round(1)

    diversity_data = latest.melt(
        id_vars="CompanyName",
        value_vars=["Diversity (%)", "Inclusion (%)"],
        var_name="Metric",
        value_name="Percentage"
    )

    fig3 = px.bar(
    diversity_data,
    x="CompanyName",
    y="Percentage",
    color="Metric",
    barmode="stack",
    title="Diversity and Inclusion Metrics"
)
    graph3_html = pio.to_html(fig3, full_html=False)
    return graph3_html

#analysis graphs routes for social dashboard
@app.route("/social_graphs")
def social_graphs():
    graph1 = Customer_Satisfaction_and_Engagement()
    graph2 = Employee_Turnover_Distribution()
    graph3 = Diversity_and_Inclusion()

    return render_template("social.html", graph1=graph1, graph2=graph2, graph3=graph3)

#graphs5 Trend & Time Analysis Page

def Trend_of_ESG_Score_Over_Time():
    df = pd.read_csv("company_esg_financial_dataset.csv")

    df = df.dropna(subset=["Year","ESG_Overall"])

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["ESG_Overall"] = pd.to_numeric(df["ESG_Overall"], errors="coerce")

    df = df.dropna()

    esg_trend = df.groupby("Year", as_index=False)["ESG_Overall"].mean()

    fig1 = px.line(
    esg_trend,
    x="Year",
    y="ESG_Overall",
    markers=True,
    title="Trend of ESG Score Over Time"
)
    graph1_html = pio.to_html(fig1, full_html=False)
    return graph1_html

def Company_Comparison():
    df = pd.read_csv("company_esg_financial_dataset.csv")

    df = df.dropna(subset=["Year","CompanyName","ESG_Overall"])

    company_trend = df.groupby(["Year", "CompanyName"], as_index=False)["ESG_Overall"].mean()

    latest_ranking = company_trend.sort_values(["CompanyName","Year"]).groupby("CompanyName").tail(1)

    top_companies = latest_ranking.nlargest(8, "ESG_Overall")["CompanyName"]

    company_top = company_trend[company_trend["CompanyName"].isin(top_companies)]

    fig2 = px.line(
    company_top,
    x="Year",
    y="ESG_Overall",
    color="CompanyName",
    markers=True,
    title="Company Comparison of ESG Score"
)
    graph2_html = pio.to_html(fig2, full_html=False)
    return graph2_html

def Growth_Pattern():
    df = pd.read_csv("company_esg_financial_dataset.csv")

    df = df.dropna(subset=["Year","CompanyName","ESG_Overall"])

    company_trend = df.groupby(["Year", "CompanyName"], as_index=False)["ESG_Overall"].mean()

    latest_ranking = company_trend.sort_values(["CompanyName","Year"]).groupby("CompanyName").tail(1)

    top_companies = latest_ranking.nlargest(8, "ESG_Overall")["CompanyName"]

    company_top = company_trend[company_trend["CompanyName"].isin(top_companies)]

    fig3 = px.area(
    company_top,
    x="Year",
    y="ESG_Overall",
    color="CompanyName",
    title="Growth Pattern of ESG Score"
)
    graph3_html = pio.to_html(fig3, full_html=False)
    return graph3_html

#analysis graphs routes for trend and time dashboard
@app.route("/trend_time_graphs")
def trend_time_graphs():
    graph1 = Trend_of_ESG_Score_Over_Time()
    graph2 = Company_Comparison()
    graph3 = Growth_Pattern()

    return render_template("trend_time.html", graph1=graph1, graph2=graph2, graph3=graph3)

#graphs6 Overall ESG Performance Page

def KPI_Overview():
    df = pd.read_csv("company_esg_financial_dataset.csv")

    df = df.dropna(subset=["CompanyName","Year","ESG_Overall"])

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["ESG_Overall"] = pd.to_numeric(df["ESG_Overall"], errors="coerce")

    df = df.dropna()

    latest = df.sort_values(["CompanyName","Year"]).groupby("CompanyName").tail(1)

# KPI calculations
    avg_score = latest["ESG_Overall"].mean()
    top_company = latest.sort_values("ESG_Overall", ascending=False).iloc[0]
    low_company = latest.sort_values("ESG_Overall", ascending=True).iloc[0]

# Print KPIs
    print("Average ESG Score:", round(avg_score,2))
    print("Top Company:", top_company["CompanyName"], top_company["ESG_Overall"])
    print("Lowest Company:", low_company["CompanyName"], low_company["ESG_Overall"])


# 🔥 Graph added (KPI Comparison)
    kpi_df = pd.DataFrame({
        "Metric": ["Average ESG", "Top ESG", "Lowest ESG"],
        "Score": [
            avg_score,
            top_company["ESG_Overall"],
            low_company["ESG_Overall"]
    ]
})

    fig1 = px.bar(
        kpi_df,
        x="Metric",
        y="Score",
        title="KPI Comparison",
        text="Score"
)

    fig1.update_traces(textposition="outside")
    graph1_html = pio.to_html(fig1, full_html=False)
    return graph1_html

def company_wise_esg_score():
    df = pd.read_csv("company_esg_financial_dataset.csv")

    df = df.dropna(subset=["CompanyName","Year","ESG_Overall"])

    latest = df.sort_values(["CompanyName","Year"]).groupby("CompanyName").tail(1)

    fig2 = px.bar(
        latest,
        x="CompanyName",
        y="ESG_Overall",
        title="Company-wise ESG Score"
)
    graph2_html = pio.to_html(fig2, full_html=False)
    return graph2_html

def esg_category_distribution():
    df = pd.read_csv("company_esg_financial_dataset.csv")

    df = df.dropna(subset=["CompanyName","Year","ESG_Environmental","ESG_Social","ESG_Governance"])

    latest = df.sort_values(["CompanyName","Year"]).groupby("CompanyName").tail(1)

    category_df = latest.melt(
        id_vars="CompanyName",
        value_vars=["ESG_Environmental", "ESG_Social", "ESG_Governance"],
        var_name="Category",
        value_name="Score"
)

    fig3 = px.bar(
        category_df,
        x="CompanyName",
        y="Score",
        color="Category",
        barmode="group",
        title="ESG Category Distribution"
)
    graph3_html = pio.to_html(fig3, full_html=False)
    return graph3_html

#analysis graphs routes for overall esg performance dashboard
@app.route("/overall_esg_graphs")
def overall_esg_graphs():
    graph1 = KPI_Overview()
    graph2 = company_wise_esg_score()
    graph3 = esg_category_distribution()

    return render_template("esg_performance.html", graph1=graph1, graph2=graph2, graph3=graph3)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
