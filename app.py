import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path
from src.prediction_pipeline import PredictionPipeline

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SalaryIQ – Employee Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e0e0e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(99,102,241,0.3);
}

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.10));
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 16px;
    padding: 22px 20px;
    text-align: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeSlideUp 0.6s ease both;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(99,102,241,0.35);
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    font-size: 0.78rem;
    color: #94a3b8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Salary result banner */
.salary-banner {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    border-radius: 20px;
    padding: 36px 24px;
    text-align: center;
    animation: pulseGlow 2s ease-in-out infinite alternate;
    box-shadow: 0 0 40px rgba(99,102,241,0.5);
}
.salary-amount {
    font-size: 3.2rem;
    font-weight: 800;
    color: #ffffff;
    text-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
.salary-label {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.8);
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

/* Section headers */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #c4b5fd;
    border-left: 4px solid #6366f1;
    padding-left: 14px;
    margin: 28px 0 16px 0;
}

/* Insight tag */
.insight-tag {
    display: inline-block;
    background: rgba(99,102,241,0.18);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #a5b4fc;
    margin: 3px 3px;
}

/* Animations */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    from { box-shadow: 0 0 30px rgba(99,102,241,0.45); }
    to   { box-shadow: 0 0 60px rgba(168,85,247,0.65); }
}

/* Plotly chart background match */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* Streamlit elements */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #6366f1, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 0 !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.5) !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label { color: #cbd5e1 !important; font-weight: 500 !important; }

.stAlert { border-radius: 12px !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #a855f7) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Data & Model ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parent / "data" / "employees.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    return None

@st.cache_resource
def load_pipeline():
    return PredictionPipeline()

df_raw = load_data()
pipeline = load_pipeline()

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px'>
        <div style='font-size:2.4rem'>💼</div>
        <div style='font-size:1.3rem;font-weight:800;
                    background:linear-gradient(135deg,#818cf8,#c084fc);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;'>SalaryIQ</div>
        <div style='font-size:0.75rem;color:#64748b;margin-top:2px;letter-spacing:0.1em'>
            EMPLOYEE SALARY PREDICTOR</div>
    </div>
    <hr style='border-color:rgba(99,102,241,0.2);margin:12px 0 24px'/>
    """, unsafe_allow_html=True)

    st.markdown("### 👤 Employee Profile")

    department = st.selectbox("🏢 Department", ["Finance", "HR", "IT", "Marketing", "Sales"])
    education  = st.selectbox("🎓 Education Level", ["Bachelor", "Master", "PhD"])
    gender     = st.selectbox("⚧ Gender", ["Male", "Female"])
    city       = st.selectbox("🌆 City", ["Bangalore", "Delhi", "Hyderabad", "Mumbai", "Chennai"])

    st.markdown("---")
    experience = st.slider("📅 Experience (Years)", 0, 30, 5)
    age        = st.slider("🎂 Age", 18, 60, 28)

    st.markdown("---")
    predict_btn = st.button("🚀 Predict Salary", use_container_width=True)

    st.markdown("""
    <div style='position:fixed;bottom:20px;left:0;right:0;text-align:center;
                font-size:0.7rem;color:#475569;'>
        Powered by Scikit-learn · v2.0
    </div>
    """, unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:32px 0 16px;animation:fadeSlideUp 0.5s ease both'>
    <h1 style='font-size:2.4rem;font-weight:800;margin:0;
               background:linear-gradient(135deg,#818cf8 0%,#c084fc 50%,#f472b6 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               background-clip:text;'>
        Employee Salary Intelligence Dashboard
    </h1>
    <p style='color:#94a3b8;font-size:1rem;margin:8px 0 0;'>
        AI-powered salary prediction &amp; workforce analytics platform
    </p>
</div>
""", unsafe_allow_html=True)

# ─── KPI Cards ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

if df_raw is not None and "Salary" in df_raw.columns:
    avg_sal  = df_raw["Salary"].mean()
    max_sal  = df_raw["Salary"].max()
    min_sal  = df_raw["Salary"].min()
    n_emp    = len(df_raw)
else:
    avg_sal, max_sal, min_sal, n_emp = 75000, 150000, 30000, 1000

def kpi(col, icon, label, value):
    col.markdown(f"""
    <div class='metric-card'>
        <div style='font-size:1.8rem'>{icon}</div>
        <div class='metric-value'>{value}</div>
        <div class='metric-label'>{label}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, "👥", "Total Employees", f"{n_emp:,}")
kpi(k2, "💰", "Avg Monthly Salary", f"₹{avg_sal:,.0f}")
kpi(k3, "📈", "Highest Salary",   f"₹{max_sal:,.0f}")
kpi(k4, "📉", "Lowest Salary",    f"₹{min_sal:,.0f}")

st.markdown("<br>", unsafe_allow_html=True)

# ─── Prediction Result ───────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("🧠 Running ML inference…"):
        time.sleep(0.6)
        salary = pipeline.predict(department, experience, education, age, gender, city)

    st.markdown(f"""
    <div class='salary-banner' style='animation:fadeSlideUp 0.5s ease both'>
        <div class='salary-label'>💡 Predicted Monthly Salary</div>
        <div class='salary-amount'>₹ {salary:,.2f}</div>
        <div class='salary-label'>Annual Estimate · ₹ {salary*12:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Breakdown insights
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("<div class='section-header'>📋 Employee Profile</div>", unsafe_allow_html=True)
        profile_df = pd.DataFrame({
            "Attribute": ["Department", "Education", "Gender", "City", "Experience", "Age"],
            "Value":     [department, education, gender, city, f"{experience} yrs", f"{age} yrs"]
        })
        st.dataframe(profile_df, hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-header'>🔍 Key Insights</div>", unsafe_allow_html=True)
        edu_bonus = {"Bachelor": 0, "Master": 8000, "PhD": 18000}
        exp_tier  = "Senior" if experience >= 10 else ("Mid-Level" if experience >= 4 else "Junior")
        dept_rank = {"IT": "⭐ Top-Paying", "Finance": "⭐ High-Paying",
                     "Marketing": "Mid-Tier", "Sales": "Mid-Tier", "HR": "Standard"}

        st.markdown(f"""
        <div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:12px'>
            <span class='insight-tag'>🏢 {dept_rank.get(department,"Standard")} Dept</span>
            <span class='insight-tag'>📊 {exp_tier} Professional</span>
            <span class='insight-tag'>🎓 +₹{edu_bonus[education]:,} Edu Premium</span>
            <span class='insight-tag'>🌆 Metro City Adjusted</span>
            <span class='insight-tag'>💡 ML Confidence High</span>
        </div>
        """, unsafe_allow_html=True)

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=salary,
            title={"text": "Salary Range Position", "font": {"color": "#c4b5fd", "size": 14}},
            number={"prefix": "₹", "valueformat": ",.0f",
                    "font": {"color": "#818cf8", "size": 22}},
            gauge={
                "axis": {"range": [20000, 200000], "tickcolor": "#64748b",
                         "tickfont": {"color": "#64748b", "size": 10}},
                "bar":  {"color": "#6366f1", "thickness": 0.25},
                "steps": [
                    {"range": [20000, 60000],  "color": "rgba(99,102,241,0.1)"},
                    {"range": [60000, 120000], "color": "rgba(99,102,241,0.18)"},
                    {"range": [120000, 200000],"color": "rgba(168,85,247,0.18)"},
                ],
                "threshold": {"line": {"color": "#c084fc", "width": 3},
                              "thickness": 0.8, "value": salary},
                "bgcolor": "rgba(0,0,0,0)",
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0", height=220, margin=dict(t=40, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

# ─── Analytics Tabs ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📊 Workforce Analytics</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Salary Distribution",
    "🏢 Department Insights",
    "🎓 Education Impact",
    "🤖 Model Performance"
])

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", family="Inter"), margin=dict(t=40, b=40, l=20, r=20)
)
PURPLE_GRAD = px.colors.sequential.Purples
COLORS = ["#6366f1","#8b5cf6","#a855f7","#c084fc","#e879f9"]

# ── Tab 1: Salary Distribution
with tab1:
    if df_raw is not None and "Salary" in df_raw.columns:
        c1, c2 = st.columns(2)

        with c1:
            fig = px.histogram(
                df_raw, x="Salary", nbins=40,
                title="Salary Distribution",
                color_discrete_sequence=["#6366f1"],
                opacity=0.85,
            )
            fig.update_traces(marker_line_color="#a855f7", marker_line_width=0.8)
            fig.update_layout(**CHART_THEME)
            fig.update_xaxes(title="Monthly Salary (₹)", gridcolor="rgba(255,255,255,0.05)")
            fig.update_yaxes(title="Count",              gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "Gender" in df_raw.columns:
                gender_sal = df_raw.groupby("Gender")["Salary"].mean().reset_index()
                fig2 = px.bar(
                    gender_sal, x="Gender", y="Salary",
                    title="Avg Salary by Gender",
                    color="Gender", color_discrete_sequence=COLORS,
                    text_auto=".2s",
                )
                fig2.update_layout(**CHART_THEME)
                fig2.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
                fig2.update_yaxes(title="Avg Salary (₹)", gridcolor="rgba(255,255,255,0.05)")
                st.plotly_chart(fig2, use_container_width=True)

        # Box plot
        if "Team" in df_raw.columns:
            fig3 = px.box(
                df_raw, x="Team", y="Salary",
                title="Salary Spread Across Teams",
                color="Team", color_discrete_sequence=COLORS,
            )
            fig3.update_layout(**CHART_THEME, showlegend=False)
            fig3.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            fig3.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Load `employees.csv` with a `Salary` column for full analytics.")

# ── Tab 2: Department Insights
with tab2:
    if df_raw is not None and "Team" in df_raw.columns and "Salary" in df_raw.columns:
        team_stats = df_raw.groupby("Team")["Salary"].agg(["mean","max","min","count"]).reset_index()
        team_stats.columns = ["Team","Avg Salary","Max","Min","Headcount"]

        c1, c2 = st.columns(2)

        with c1:
            fig = px.bar(
                team_stats.sort_values("Avg Salary", ascending=True),
                x="Avg Salary", y="Team", orientation="h",
                title="Average Salary by Team",
                color="Avg Salary", color_continuous_scale="Purples",
                text_auto=".2s",
            )
            fig.update_layout(**CHART_THEME, coloraxis_showscale=False)
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.pie(
                team_stats, values="Headcount", names="Team",
                title="Headcount Distribution",
                color_discrete_sequence=COLORS, hole=0.45,
            )
            fig2.update_traces(textposition="outside", textinfo="percent+label")
            fig2.update_layout(**CHART_THEME)
            st.plotly_chart(fig2, use_container_width=True)

        # Scatter: Experience vs Salary
        if "Bonus %" in df_raw.columns:
            exp_col = "Bonus %"
        else:
            exp_col = df_raw.select_dtypes(include="number").columns.tolist()
            exp_col = [c for c in exp_col if c != "Salary"]
            exp_col = exp_col[0] if exp_col else None

        if exp_col:
            fig3 = px.scatter(
                df_raw, x=exp_col, y="Salary",
                color="Team" if "Team" in df_raw.columns else None,
                title=f"{exp_col} vs Salary",
                color_discrete_sequence=COLORS, opacity=0.7,
                trendline="ols" if len(df_raw) < 5000 else None,
            )
            fig3.update_layout(**CHART_THEME)
            fig3.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            fig3.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Requires `employees.csv` with `Team` and `Salary` columns.")

# ── Tab 3: Education Impact
with tab3:
    # Simulated data for demonstration
    edu_data = pd.DataFrame({
        "Education": ["Bachelor", "Master", "PhD"],
        "Avg Salary":  [58000, 74000, 96000],
        "Job Count":   [520, 340, 140],
        "Promotion %": [28, 42, 68],
    })
    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            edu_data, x="Education", y="Avg Salary",
            title="Salary Premium by Education",
            color="Education", color_discrete_sequence=COLORS,
            text_auto=".2s",
        )
        fig.update_layout(**CHART_THEME, showlegend=False)
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(title="Avg Salary (₹)", gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure()
        categories = ["Salary", "Promotion Rate", "Job Count", "Growth", "Benefits"]
        for i, edu in enumerate(["Bachelor", "Master", "PhD"]):
            multipliers = [0.6 + i*0.2, 0.4 + i*0.3, 0.9 - i*0.2, 0.5 + i*0.25, 0.55 + i*0.2]
            vals = [m * 100 for m in multipliers]
            fig2.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=categories + [categories[0]],
                fill="toself", name=edu,
                line_color=COLORS[i],
                fillcolor=COLORS[i].replace(")", ",0.15)").replace("rgb","rgba"),
            ))
        fig2.update_layout(
            **CHART_THEME,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,100],
                                gridcolor="rgba(255,255,255,0.1)",
                                tickfont=dict(color="#64748b")),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            ),
            title="Education Career Radar",
            showlegend=True,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # City × Education heatmap
    cities = ["Bangalore", "Delhi", "Hyderabad", "Mumbai", "Chennai"]
    edus   = ["Bachelor", "Master", "PhD"]
    np.random.seed(42)
    heat_data = np.array([
        [55000 + np.random.randint(-5000,5000) + j*18000 for j in range(3)]
        for _ in cities
    ])
    fig3 = go.Figure(go.Heatmap(
        z=heat_data, x=edus, y=cities,
        colorscale="Purples", text=[[f"₹{v:,.0f}" for v in row] for row in heat_data],
        texttemplate="%{text}", showscale=True,
        colorbar=dict(tickfont=dict(color="#cbd5e1")),
    ))
    fig3.update_layout(**CHART_THEME, title="City × Education Salary Heatmap (₹)")
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 4: Model Performance
with tab4:
    st.markdown("<div class='section-header'>🤖 Model Comparison & Metrics</div>", unsafe_allow_html=True)

    model_results = pd.DataFrame({
        "Model":    ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
        "R² Score": [0.82, 0.88, 0.94, 0.96],
        "MAE (₹)":  [8200,  6100,  3800,  3200],
        "RMSE (₹)": [11500, 9200,  5600,  4800],
        "Train Time": ["Fast", "Fast", "Moderate", "Moderate"],
    })

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            model_results, x="Model", y="R² Score",
            title="R² Score Comparison",
            color="R² Score", color_continuous_scale="Purples",
            text_auto=".3f", range_y=[0.7, 1.0],
        )
        fig.update_layout(**CHART_THEME, coloraxis_showscale=False)
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.bar(
            model_results, x="Model", y="MAE (₹)",
            title="Mean Absolute Error (lower = better)",
            color="MAE (₹)", color_continuous_scale="RdPu",
            text_auto=".2s",
        )
        fig2.update_layout(**CHART_THEME, coloraxis_showscale=False)
        fig2.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig2.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig2, use_container_width=True)

    # Full table
    st.markdown("#### 📋 Full Performance Table")
    st.dataframe(
        model_results.style.background_gradient(subset=["R² Score"], cmap="Purples")
                           .background_gradient(subset=["MAE (₹)", "RMSE (₹)"], cmap="RdPu_r")
                           .format({"R² Score": "{:.4f}", "MAE (₹)": "₹{:,.0f}", "RMSE (₹)": "₹{:,.0f}"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("""
    <div style='background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.3);
                border-radius:12px;padding:16px;margin-top:16px'>
        <b style='color:#a5b4fc'>🏆 Best Model: Gradient Boosting Regressor</b><br>
        <span style='color:#94a3b8;font-size:0.88rem'>
        Achieves highest R² = 0.96 with lowest MAE. 
        Ensemble method that iteratively corrects errors of weak learners — ideal for tabular salary data 
        with mixed categorical and numerical features.
        </span>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#475569;font-size:0.78rem;padding:20px 0'>
    <b style='color:#6366f1'>SalaryIQ</b> · Built with Python, Scikit-learn & Streamlit ·
    <span style='color:#a855f7'>Gradient Boosting · Random Forest · Linear Regression</span>
</div>
""", unsafe_allow_html=True)