import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PROFESSIONAL THEME
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }

    [data-testid="stSidebar"] * {
        color: #f9fafb !important;
    }

    .app-brand {
        padding: 8px 0 20px 0;
    }

    .app-brand h1 {
        font-size: 24px;
        margin: 0;
        font-weight: 800;
        color: #ffffff;
    }

    .app-brand p {
        margin: 5px 0 0 0;
        font-size: 12px;
        color: #9ca3af !important;
    }

    .hero {
        background: linear-gradient(135deg, #111827 0%, #1e3a5f 100%);
        padding: 30px 34px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(17, 24, 39, 0.12);
    }

    .hero h1 {
        margin: 0;
        font-size: 32px;
        font-weight: 800;
        color: white;
    }

    .hero p {
        margin: 8px 0 0 0;
        color: #dbeafe;
        font-size: 14px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 750;
        color: #111827;
        margin: 8px 0 14px 0;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 13px;
        margin-top: -8px;
        margin-bottom: 16px;
    }

    .kpi-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px;
        min-height: 115px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
    }

    .kpi-label {
        color: #6b7280;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .04em;
    }

    .kpi-value {
        color: #111827;
        font-size: 26px;
        font-weight: 800;
        margin-top: 8px;
    }

    .kpi-note {
        color: #6b7280;
        font-size: 11px;
        margin-top: 5px;
    }

    .risk-high {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        color: #9f1239;
        border-radius: 14px;
        padding: 18px;
    }

    .risk-medium {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
        border-radius: 14px;
        padding: 18px;
    }

    .risk-low {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        border-radius: 14px;
        padding: 18px;
    }

    .insight-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px;
        height: 100%;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.04);
    }

    .insight-card h4 {
        margin: 0 0 8px 0;
        color: #111827;
    }

    .insight-card p {
        margin: 0;
        color: #4b5563;
        font-size: 13px;
        line-height: 1.55;
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 11px;
        padding: 24px 0 8px 0;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 16px;
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.04);
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 44px;
    }

    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    [data-testid="stDataFrame"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONSTANTS
# =========================================================
COLOR_CHURN = "#EF553B"
COLOR_STAY = "#00CC96"
COLOR_PRIMARY = "#2563EB"
COLOR_DARK = "#111827"
PLOT_TEMPLATE = "plotly_white"


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def money(value):
    return f"${value:,.0f}"


def pct(value):
    return f"{value:.1f}%"


def add_chart_layout(fig, title=None, height=380):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=55 if title else 20, b=10),
        font=dict(family="Inter, sans-serif", color="#374151"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white"),
    )
    return fig


def kpi_card(label, value, note=""):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """


def page_header(title, subtitle, icon="📊"):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{icon} {title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_band(probability):
    if probability >= 0.60:
        return "HIGH", "risk-high"
    if probability >= 0.30:
        return "MEDIUM", "risk-medium"
    return "LOW", "risk-low"


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load("churn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        model_columns = joblib.load("model_columns.pkl")
        return model, scaler, model_columns, None
    except Exception as exc:
        return None, None, None, str(exc)


@st.cache_data
def load_dataset():
    data = pd.read_csv("Telco-Customer-Churn.csv")
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"], errors="coerce"
        )
    data = data.dropna(subset=["TotalCharges"]).copy()
    return data


model, scaler, model_columns, model_error = load_model_artifacts()

try:
    df = load_dataset()
    data_error = None
except Exception as exc:
    df = pd.DataFrame()
    data_error = str(exc)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="app-brand">
            <h1>📊 ChurnIQ</h1>
            <p>Customer Retention Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Executive Dashboard",
            "📊 Churn Analysis",
            "🔮 Churn Prediction",
            "💡 Business Insights",
        ],
    )

    st.divider()

    if not df.empty:
        st.caption("DATASET")
        st.write(f"**{len(df):,}** customers")
        st.write(f"**{df.shape[1]}** variables")

    st.caption("MODEL")
    st.write("**XGBoost Classifier**")
    st.caption("Preprocessing: One-hot encoding + scaling + SMOTE")

    st.divider()
    st.caption("Customer Churn Intelligence • v2.0")

# =========================================================
# GLOBAL ERROR CHECK
# =========================================================
if data_error:
    st.error(
        "Dataset could not be loaded. Make sure "
        "`Telco-Customer-Churn.csv` is in the same folder as `app.py`."
    )
    st.code(data_error)
    st.stop()

if model_error:
    st.warning(
        "The dashboard and analysis pages can still be viewed, "
        "but prediction is unavailable because the model artifacts "
        "could not be loaded."
    )
    with st.expander("Model loading details"):
        st.code(model_error)

# =========================================================
# COMMON METRICS
# =========================================================
total_customers = len(df)
churned_customers = int((df["Churn"] == "Yes").sum())
retained_customers = int((df["Churn"] == "No").sum())
churn_rate = churned_customers / total_customers * 100
retention_rate = retained_customers / total_customers * 100
avg_monthly_charges = df["MonthlyCharges"].mean()
avg_tenure = df["tenure"].mean()
monthly_revenue = df["MonthlyCharges"].sum()

# =========================================================
# PAGE 1 — EXECUTIVE DASHBOARD
# =========================================================
if page == "🏠 Executive Dashboard":
    page_header(
        "Customer Churn Intelligence",
        "Executive view of customer retention, churn exposure, and revenue risk.",
        "📊",
    )

    # KPI ROW
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            kpi_card(
                "Total Customers",
                f"{total_customers:,}",
                "Customers in the analysis dataset",
            ),
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            kpi_card(
                "Churn Rate",
                pct(churn_rate),
                f"{churned_customers:,} customers churned",
            ),
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            kpi_card(
                "Avg. Monthly Charge",
                f"${avg_monthly_charges:,.2f}",
                "Average customer monthly bill",
            ),
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            kpi_card(
                "Avg. Tenure",
                f"{avg_tenure:.1f} mo.",
                "Average customer relationship length",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # MAIN VISUALS
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="section-title">Churn Overview</div>', unsafe_allow_html=True)

        churn_counts = (
            df["Churn"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Customers")
        )

        fig = px.pie(
            churn_counts,
            names="Status",
            values="Customers",
            hole=0.62,
            color="Status",
            color_discrete_map={"Yes": COLOR_CHURN, "No": COLOR_STAY},
        )
        fig.update_traces(
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>%{percent}<extra></extra>",
        )
        fig.add_annotation(
            text=f"<b>{churn_rate:.1f}%</b><br>Churn",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=18, color=COLOR_DARK),
        )
        add_chart_layout(fig, height=390)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Churn by Contract</div>', unsafe_allow_html=True)

        contract_rate = (
            df.groupby("Contract")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Churn Rate")
            .sort_values("Churn Rate", ascending=True)
        )

        fig = px.bar(
            contract_rate,
            x="Churn Rate",
            y="Contract",
            orientation="h",
            text="Churn Rate",
        )
        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            marker_color=COLOR_CHURN,
        )
        fig.update_xaxes(title="Churn Rate (%)", range=[0, max(100, contract_rate["Churn Rate"].max() + 15)])
        fig.update_yaxes(title="")
        add_chart_layout(fig, height=390)
        st.plotly_chart(fig, use_container_width=True)

    # SECOND ROW
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">Tenure vs. Churn</div>', unsafe_allow_html=True)
        tenure_bins = pd.cut(
            df["tenure"],
            bins=[-1, 6, 12, 24, 48, 72, 1000],
            labels=["0–6", "7–12", "13–24", "25–48", "49–72", "73+"],
        )

        tenure_churn = (
            df.assign(TenureGroup=tenure_bins)
            .groupby("TenureGroup", observed=False)["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Churn Rate")
        )

        fig = px.line(
            tenure_churn,
            x="TenureGroup",
            y="Churn Rate",
            markers=True,
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=9))
        fig.update_yaxes(title="Churn Rate (%)")
        fig.update_xaxes(title="Tenure (months)")
        add_chart_layout(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Payment Method Risk</div>', unsafe_allow_html=True)

        payment_rate = (
            df.groupby("PaymentMethod")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Churn Rate")
            .sort_values("Churn Rate", ascending=False)
        )

        fig = px.bar(
            payment_rate,
            x="Churn Rate",
            y="PaymentMethod",
            orientation="h",
            text="Churn Rate",
        )
        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            marker_color=COLOR_PRIMARY,
        )
        fig.update_xaxes(title="Churn Rate (%)")
        fig.update_yaxes(title="")
        add_chart_layout(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)

    # EXECUTIVE TAKEAWAYS
    st.markdown('<div class="section-title">Executive Takeaways</div>', unsafe_allow_html=True)

    highest_contract = contract_rate.sort_values("Churn Rate", ascending=False).iloc[0]
    highest_payment = payment_rate.iloc[0]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>🎯 Highest Contract Risk</h4>
                <p><b>{highest_contract['Contract']}</b> customers have the highest
                observed churn rate at <b>{highest_contract['Churn Rate']:.1f}%</b>.
                This segment deserves focused retention attention.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>💳 Payment Risk</h4>
                <p><b>{highest_payment['PaymentMethod']}</b> has the highest observed
                churn rate at <b>{highest_payment['Churn Rate']:.1f}%</b>.
                Payment experience may be worth investigating.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>💰 Revenue Exposure</h4>
                <p>The dataset represents approximately <b>${monthly_revenue:,.0f}</b>
                in monthly customer charges. Reducing churn can protect recurring
                revenue and customer lifetime value.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# PAGE 2 — CHURN ANALYSIS
# =========================================================
elif page == "📊 Churn Analysis":
    page_header(
        "Churn Analysis",
        "Explore customer segments and identify the strongest observable churn patterns.",
        "📊",
    )

    # Filters
    st.markdown('<div class="section-title">Interactive Filters</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)

    with f1:
        selected_contracts = st.multiselect(
            "Contract",
            sorted(df["Contract"].dropna().unique()),
            default=sorted(df["Contract"].dropna().unique()),
        )

    with f2:
        selected_internet = st.multiselect(
            "Internet Service",
            sorted(df["InternetService"].dropna().unique()),
            default=sorted(df["InternetService"].dropna().unique()),
        )

    with f3:
        selected_gender = st.multiselect(
            "Gender",
            sorted(df["gender"].dropna().unique()),
            default=sorted(df["gender"].dropna().unique()),
        )

    filtered_df = df[
        df["Contract"].isin(selected_contracts)
        & df["InternetService"].isin(selected_internet)
        & df["gender"].isin(selected_gender)
    ].copy()

    if filtered_df.empty:
        st.warning("No customers match the selected filters.")
        st.stop()

    fc1, fc2, fc3, fc4 = st.columns(4)

    filtered_churn = (filtered_df["Churn"] == "Yes").sum()

    with fc1:
        st.metric("Filtered Customers", f"{len(filtered_df):,}")

    with fc2:
        st.metric(
            "Filtered Churn Rate",
            f"{filtered_churn / len(filtered_df) * 100:.1f}%",
        )

    with fc3:
        st.metric(
            "Avg. Monthly Charges",
            f"${filtered_df['MonthlyCharges'].mean():,.2f}",
        )

    with fc4:
        st.metric(
            "Avg. Tenure",
            f"{filtered_df['tenure'].mean():.1f} mo.",
        )

    st.divider()

    # Churn by selected dimensions
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Churn Distribution</div>', unsafe_allow_html=True)

        counts = (
            filtered_df["Churn"]
            .value_counts()
            .rename_axis("Churn")
            .reset_index(name="Customers")
        )

        fig = px.bar(
            counts,
            x="Churn",
            y="Customers",
            color="Churn",
            color_discrete_map={"Yes": COLOR_CHURN, "No": COLOR_STAY},
            text="Customers",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        add_chart_layout(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Churn by Contract</div>', unsafe_allow_html=True)

        contract_cross = pd.crosstab(
            filtered_df["Contract"],
            filtered_df["Churn"],
        ).reset_index()

        fig = px.bar(
            contract_cross,
            x="Contract",
            y=["No", "Yes"],
            barmode="group",
            color_discrete_map={"No": COLOR_STAY, "Yes": COLOR_CHURN},
        )
        add_chart_layout(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Internet Service Risk</div>', unsafe_allow_html=True)

        internet_rate = (
            filtered_df.groupby("InternetService")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Churn Rate")
            .sort_values("Churn Rate", ascending=False)
        )

        fig = px.bar(
            internet_rate,
            x="InternetService",
            y="Churn Rate",
            text="Churn Rate",
        )
        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            marker_color=COLOR_PRIMARY,
        )
        fig.update_yaxes(title="Churn Rate (%)")
        add_chart_layout(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Payment Method Risk</div>', unsafe_allow_html=True)

        payment_rate = (
            filtered_df.groupby("PaymentMethod")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Churn Rate")
            .sort_values("Churn Rate", ascending=False)
        )

        fig = px.bar(
            payment_rate,
            x="Churn Rate",
            y="PaymentMethod",
            orientation="h",
            text="Churn Rate",
        )
        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            marker_color=COLOR_CHURN,
        )
        fig.update_xaxes(title="Churn Rate (%)")
        fig.update_yaxes(title="")
        add_chart_layout(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Average Tenure by Churn</div>', unsafe_allow_html=True)
        tenure_summary = (
            filtered_df.groupby("Churn")["tenure"]
            .mean()
            .reset_index()
        )
        fig = px.bar(
            tenure_summary,
            x="Churn",
            y="tenure",
            color="Churn",
            color_discrete_map={"Yes": COLOR_CHURN, "No": COLOR_STAY},
            text="tenure",
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title="Average Tenure (months)")
        add_chart_layout(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Monthly Charges by Churn</div>', unsafe_allow_html=True)
        charge_summary = (
            filtered_df.groupby("Churn")["MonthlyCharges"]
            .mean()
            .reset_index()
        )
        fig = px.bar(
            charge_summary,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_map={"Yes": COLOR_CHURN, "No": COLOR_STAY},
            text="MonthlyCharges",
        )
        fig.update_traces(texttemplate="$%{text:.2f}", textposition="outside")
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title="Average Monthly Charges")
        add_chart_layout(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown('<div class="section-title">Customer Segment Risk Table</div>', unsafe_allow_html=True)

    segment_table = (
        filtered_df.groupby(["Contract", "InternetService"])
        .agg(
            Customers=("Churn", "size"),
            Churned=("Churn", lambda x: (x == "Yes").sum()),
            AvgMonthlyCharges=("MonthlyCharges", "mean"),
            AvgTenure=("tenure", "mean"),
        )
        .reset_index()
    )
    segment_table["Churn Rate"] = (
        segment_table["Churned"] / segment_table["Customers"] * 100
    ).round(1)
    segment_table["AvgMonthlyCharges"] = segment_table["AvgMonthlyCharges"].round(2)
    segment_table["AvgTenure"] = segment_table["AvgTenure"].round(1)

    segment_table = segment_table.sort_values("Churn Rate", ascending=False)

    st.dataframe(
        segment_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "AvgMonthlyCharges": st.column_config.NumberColumn(
                "Avg Monthly Charges", format="$%.2f"
            ),
            "Churn Rate": st.column_config.NumberColumn(
                "Churn Rate", format="%.1f%%"
            ),
        },
    )

    st.caption(
        "These are observed associations in the historical dataset. "
        "They should not be interpreted as causal effects."
    )

# =========================================================
# PAGE 3 — CHURN PREDICTION
# =========================================================
elif page == "🔮 Churn Prediction":
    page_header(
        "Customer Churn Prediction",
        "Enter customer attributes to estimate churn probability and determine an appropriate retention response.",
        "🔮",
    )

    if model is None or scaler is None or model_columns is None:
        st.error(
            "Prediction is unavailable. Ensure `churn_model.pkl`, "
            "`scaler.pkl`, and `model_columns.pkl` are present."
        )
        st.stop()

    # Customer profile
    st.markdown('<div class="section-title">1. Customer Profile</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])

    with c2:
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12,
            step=1,
        )
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])

    with c3:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No phone service", "No", "Yes"],
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

    st.markdown('<div class="section-title">2. Services</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        online_security = st.selectbox(
            "Online Security",
            ["No internet service", "No", "Yes"],
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["No internet service", "No", "Yes"],
        )

    with c2:
        device_protection = st.selectbox(
            "Device Protection",
            ["No internet service", "No", "Yes"],
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["No internet service", "No", "Yes"],
        )

    with c3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No internet service", "No", "Yes"],
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No internet service", "No", "Yes"],
        )

    st.markdown('<div class="section-title">3. Billing</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

    with c2:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    with c3:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0,
            step=1.0,
        )

    total_charges_default = monthly_charges * tenure
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=float(total_charges_default),
        step=10.0,
        help="A starting estimate is calculated from monthly charges × tenure. Adjust it if the actual customer total is known.",
    )

    st.divider()

    predict_clicked = st.button(
        "🔮 Calculate Churn Risk",
        type="primary",
        use_container_width=True,
    )

    if predict_clicked:
        customer = pd.DataFrame(
            {
                "gender": [gender],
                "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
                "Partner": [partner],
                "Dependents": [dependents],
                "tenure": [tenure],
                "PhoneService": [phone_service],
                "MultipleLines": [multiple_lines],
                "InternetService": [internet_service],
                "OnlineSecurity": [online_security],
                "OnlineBackup": [online_backup],
                "DeviceProtection": [device_protection],
                "TechSupport": [tech_support],
                "StreamingTV": [streaming_tv],
                "StreamingMovies": [streaming_movies],
                "Contract": [contract],
                "PaperlessBilling": [paperless_billing],
                "PaymentMethod": [payment_method],
                "MonthlyCharges": [monthly_charges],
                "TotalCharges": [total_charges],
            }
        )

        try:
            customer_encoded = pd.get_dummies(
                customer,
                drop_first=True,
            )

            customer_encoded = customer_encoded.reindex(
                columns=model_columns,
                fill_value=0,
            )

            numerical_columns = [
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
            ]

            missing_numerical = [
                col for col in numerical_columns
                if col not in customer_encoded.columns
            ]

            if missing_numerical:
                raise ValueError(
                    "The saved model columns do not contain the expected "
                    f"numerical features: {missing_numerical}"
                )

            customer_encoded[numerical_columns] = scaler.transform(
                customer_encoded[numerical_columns]
            )

            prediction = int(model.predict(customer_encoded)[0])
            probability = float(
                model.predict_proba(customer_encoded)[0][1]
            )

            probability_percent = probability * 100
            risk, risk_class = risk_band(probability)

            # Save results in session state
            st.session_state["last_prediction"] = {
                "prediction": prediction,
                "probability": probability,
                "risk": risk,
                "customer": customer.copy(),
            }

        except Exception as exc:
            st.error("Prediction failed. Check that your saved preprocessing artifacts match the training pipeline.")
            with st.expander("Technical error"):
                st.code(str(exc))

    # Display last prediction
    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
        probability = result["probability"]
        probability_percent = probability * 100
        risk = result["risk"]

        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        r1, r2 = st.columns([1, 1.2])

        with r1:
            risk_class = {
                "HIGH": "risk-high",
                "MEDIUM": "risk-medium",
                "LOW": "risk-low",
            }[risk]

            risk_message = {
                "HIGH": "Immediate retention attention recommended.",
                "MEDIUM": "Monitor this customer and consider targeted engagement.",
                "LOW": "Current predicted churn exposure is relatively low.",
            }[risk]

            st.markdown(
                f"""
                <div class="{risk_class}">
                    <div style="font-size:12px;font-weight:700;letter-spacing:.05em;">
                        PREDICTED RISK
                    </div>
                    <div style="font-size:30px;font-weight:800;margin:6px 0;">
                        {risk}
                    </div>
                    <div style="font-size:13px;">
                        Churn probability: <b>{probability_percent:.2f}%</b>
                    </div>
                    <div style="font-size:12px;margin-top:8px;">
                        {risk_message}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if risk == "HIGH":
                st.error(
                    "Suggested action: prioritize this customer for a retention campaign, "
                    "service review, or personalized offer."
                )
            elif risk == "MEDIUM":
                st.warning(
                    "Suggested action: monitor the account and consider proactive engagement "
                    "before the risk increases."
                )
            else:
                st.success(
                    "Suggested action: maintain normal engagement while continuing routine monitoring."
                )

        with r2:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability_percent,
                    number={"suffix": "%", "font": {"size": 32}},
                    title={"text": "Churn Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": COLOR_CHURN},
                        "steps": [
                            {"range": [0, 30], "color": "#dcfce7"},
                            {"range": [30, 60], "color": "#fef3c7"},
                            {"range": [60, 100], "color": "#fee2e2"},
                        ],
                        "threshold": {
                            "line": {"color": COLOR_DARK, "width": 4},
                            "thickness": 0.75,
                            "value": probability_percent,
                        },
                    },
                )
            )
            gauge.update_layout(
                height=270,
                margin=dict(l=20, r=20, t=60, b=10),
                paper_bgcolor="white",
            )
            st.plotly_chart(gauge, use_container_width=True)

        st.markdown('<div class="section-title">Customer Risk Profile</div>', unsafe_allow_html=True)

        profile = result["customer"].T.reset_index()
        profile.columns = ["Feature", "Value"]
        st.dataframe(profile, use_container_width=True, hide_index=True)

        st.caption(
            "Important: this is a statistical prediction based on historical patterns. "
            "It is not a guarantee that the customer will churn and does not establish causation."
        )

# =========================================================
# PAGE 4 — BUSINESS INSIGHTS
# =========================================================
elif page == "💡 Business Insights":
    page_header(
        "Business Insights & Retention Strategy",
        "Translate observed churn patterns into measurable business actions and scenario planning.",
        "💡",
    )

    churned_df = df[df["Churn"] == "Yes"].copy()

    monthly_revenue_lost = churned_df["MonthlyCharges"].sum()
    annual_revenue_lost = monthly_revenue_lost * 12

    # Revenue exposure
    st.markdown('<div class="section-title">Revenue Exposure</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Customers Churned", f"{len(churned_df):,}")

    with c2:
        st.metric("Monthly Charges at Risk", money(monthly_revenue_lost))

    with c3:
        st.metric("Annualized Exposure", money(annual_revenue_lost))

    with c4:
        st.metric("Overall Churn Rate", pct(churn_rate))

    st.divider()

    # High risk segments
    st.markdown('<div class="section-title">Highest-Risk Customer Segments</div>', unsafe_allow_html=True)

    segment_summary = (
        df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .round(1)
        .reset_index(name="Churn Rate (%)")
        .sort_values("Churn Rate (%)", ascending=False)
    )

    fig = px.bar(
        segment_summary,
        x="Contract",
        y="Churn Rate (%)",
        text="Churn Rate (%)",
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker_color=COLOR_CHURN,
    )
    fig.update_yaxes(title="Churn Rate (%)")
    add_chart_layout(fig, height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Payment + service risk
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Payment Method Risk</div>', unsafe_allow_html=True)

        payment_summary = (
            df.groupby("PaymentMethod")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .round(1)
            .reset_index(name="Churn Rate (%)")
            .sort_values("Churn Rate (%)", ascending=False)
        )

        st.dataframe(
            payment_summary,
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.markdown('<div class="section-title">Service Risk Signals</div>', unsafe_allow_html=True)

        service_columns = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
        ]

        service_rows = []

        for column in service_columns:
            if column in df.columns:
                grouped = (
                    df.groupby(column)["Churn"]
                    .apply(lambda x: (x == "Yes").mean() * 100)
                    .reset_index(name="Churn Rate")
                )

                if len(grouped) >= 2:
                    service_rows.append(
                        {
                            "Service": column,
                            "Highest observed churn": grouped["Churn Rate"].max(),
                        }
                    )

        service_summary = pd.DataFrame(service_rows)

        if not service_summary.empty:
            service_summary["Highest observed churn"] = service_summary[
                "Highest observed churn"
            ].round(1)

        st.dataframe(
            service_summary,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # Retention simulator
    st.markdown('<div class="section-title">Retention Campaign Simulator</div>', unsafe_allow_html=True)

    st.write(
        "Use the slider to model a hypothetical reduction in churn. "
        "This is scenario analysis, not a guaranteed forecast."
    )

    reduction_pct = st.slider(
        "Assumed reduction in churn (%)",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
    )

    customers_saved = int(
        len(churned_df) * reduction_pct / 100
    )

    revenue_saved = (
        customers_saved
        * df["MonthlyCharges"].mean()
        * 12
    )

    s1, s2, s3 = st.columns(3)

    with s1:
        st.metric("Customers Potentially Retained", f"{customers_saved:,}")

    with s2:
        st.metric("Estimated Annual Revenue Protected", money(revenue_saved))

    with s3:
        projected_churn_rate = churn_rate * (1 - reduction_pct / 100)
        st.metric("Scenario Churn Rate", pct(projected_churn_rate))

    st.divider()

    # Action plan
    st.markdown('<div class="section-title">Recommended Retention Playbook</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)

    with a1:
        st.markdown(
            """
            <div class="insight-card">
                <h4>1. Address Month-to-Month Risk</h4>
                <p>
                Month-to-month customers show substantially higher churn than
                longer contracts in this dataset. Consider targeted incentives,
                loyalty benefits, or contract upgrade offers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="insight-card">
                <h4>2. Strengthen Early-Customer Onboarding</h4>
                <p>
                Lower-tenure customers should receive proactive onboarding,
                support check-ins, and early engagement campaigns before
                dissatisfaction becomes a cancellation decision.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:
        st.markdown(
            """
            <div class="insight-card">
                <h4>3. Review Payment Experience</h4>
                <p>
                Payment methods with higher observed churn should be investigated.
                Automatic payment options and frictionless billing may be useful
                retention levers, but should be validated with controlled testing.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="insight-card">
                <h4>4. Target Service Gaps</h4>
                <p>
                Customers without selected support and security services can be
                evaluated as potential retention segments. Bundling should be
                tested rather than assumed to cause lower churn.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown(
        """
        <div class="footer">
            ChurnIQ uses historical customer data and a trained XGBoost model.
            Business recommendations are analytical suggestions and should be
            validated through experiments and operational data.
        </div>
        """,
        unsafe_allow_html=True,
    )
