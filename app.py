import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn",
    page_icon="📊",
    layout="wide"
)
st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 15px;
}
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
}
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

import plotly.express as px

COLOR_CHURN = "#EF553B"
COLOR_STAY = "#00CC96"

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")


# =========================================================
# LOAD ORIGINAL DATASET
# =========================================================

# IMPORTANT:
# Make sure this filename exactly matches your CSV filename
# in your GitHub repository.

df = pd.read_csv(
    "Telco-Customer-Churn.csv"
)


# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows where TotalCharges could not be converted
df = df.dropna(
    subset=["TotalCharges"]
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Customer Churn")

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Overview",
        "📊 Churn Analysis",
        "🔮 Churn Prediction",
        "💡 Business Insights"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    """
    **Customer Churn Prediction**

    Machine Learning Model:
    XGBoost

    Application:
    Streamlit
    """
)


# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================

if page == "🏠 Overview":

    st.title("📊 Customer Churn Overview")

    st.markdown(
        """
        ## Welcome

        This application uses **machine learning to predict
        customer churn**.

        The objective is to identify customers who may be at
        risk of leaving a company based on their demographic,
        service, contract, and billing information.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # KPI CALCULATIONS
    # -----------------------------------------------------

    total_customers = len(df)

    churned_customers = (
        df["Churn"]
        .value_counts()
        .get("Yes", 0)
    )

    churn_rate = (
        churned_customers / total_customers
    ) * 100

    avg_monthly_charges = (
        df["MonthlyCharges"].mean()
    )

    avg_tenure = (
        df["tenure"].mean()
    )


    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👥 Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "📉 Churn Rate",
            f"{churn_rate:.1f}%"
        )

    with col3:
        st.metric(
            "💰 Avg Monthly Charges",
            f"${avg_monthly_charges:.2f}"
        )

    with col4:
        st.metric(
            "📅 Avg Tenure",
            f"{avg_tenure:.1f} months"
        )


    st.divider()


    # -----------------------------------------------------
    # CHURN DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("📈 Customer Churn Distribution")

    churn_counts = (
        df["Churn"]
        .value_counts()
        .rename_axis("Churn")
        .reset_index(name="Customers")
    )

    fig = px.bar(
    churn_counts, x="Churn", y="Customers", color="Churn",
    color_discrete_map={"Yes": COLOR_CHURN, "No": COLOR_STAY},
    title="Customer Churn Distribution", text="Customers"
)
fig.update_layout(showlegend=False, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)
    )


    st.divider()


    # -----------------------------------------------------
    # CONTRACT OVERVIEW
    # -----------------------------------------------------

    st.subheader("📄 Customers by Contract Type")

    contract_counts = (
        df["Contract"]
        .value_counts()
    )

    st.bar_chart(
        contract_counts
    )


    st.divider()


    # -----------------------------------------------------
    # PROJECT OBJECTIVE
    # -----------------------------------------------------

    st.subheader("🎯 Project Objective")

    st.write(
        """
        The goal of this project is to build a machine
        learning system that predicts whether a customer
        is likely to churn.

        The project includes:

        • Exploratory Data Analysis

        • Data preprocessing

        • One-hot encoding

        • Feature scaling

        • SMOTE for class imbalance

        • XGBoost classification

        • Model evaluation

        • Interactive prediction

        • Streamlit deployment
        """
    )


# =========================================================
# PAGE 2 — CHURN ANALYSIS
# =========================================================

elif page == "📊 Churn Analysis":

    st.title("📊 Customer Churn Analysis")

    st.markdown(
        """
        Explore customer behavior and identify patterns
        associated with churn.
        """
    )

    st.divider()


    # -----------------------------------------------------
    # CHURN DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("📈 Overall Churn Distribution")

    churn_counts = (
        df["Churn"]
        .value_counts()
        .rename_axis("Churn")
        .reset_index(name="Customers")
    )

    st.bar_chart(
        churn_counts.set_index("Churn")
    )

    st.write(
        "This shows the number of customers who stayed "
        "with the company compared with those who churned."
    )


    st.divider()


    # -----------------------------------------------------
    # CHURN BY CONTRACT
    # -----------------------------------------------------

    st.subheader("📄 Churn by Contract Type")

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"]
    )

    st.bar_chart(
        contract_churn
    )

    st.write(
        "Contract type can be useful for understanding "
        "differences in customer retention patterns."
    )


    st.divider()


    # -----------------------------------------------------
    # CHURN BY INTERNET SERVICE
    # -----------------------------------------------------

    st.subheader("🌐 Churn by Internet Service")

    internet_churn = pd.crosstab(
        df["InternetService"],
        df["Churn"]
    )

    st.bar_chart(
        internet_churn
    )


    st.divider()


    # -----------------------------------------------------
    # CHURN BY PAYMENT METHOD
    # -----------------------------------------------------

    st.subheader("💳 Churn by Payment Method")

    payment_churn = pd.crosstab(
        df["PaymentMethod"],
        df["Churn"]
    )

    st.bar_chart(
        payment_churn
    )


    st.divider()


    # -----------------------------------------------------
    # TENURE ANALYSIS
    # -----------------------------------------------------

    st.subheader("📅 Average Tenure by Churn Status")

    tenure_analysis = (
        df.groupby("Churn")["tenure"]
        .mean()
        .round(2)
    )

    st.bar_chart(
        tenure_analysis
    )


    st.divider()


    # -----------------------------------------------------
    # MONTHLY CHARGES
    # -----------------------------------------------------

    st.subheader("💰 Average Monthly Charges by Churn")

    monthly_charges_analysis = (
        df.groupby("Churn")["MonthlyCharges"]
        .mean()
        .round(2)
    )

    st.bar_chart(
        monthly_charges_analysis
    )


    st.divider()


    # -----------------------------------------------------
    # KEY OBSERVATIONS
    # -----------------------------------------------------

    st.subheader("💡 Key Observations")

    st.markdown(
        """
        **1. Contract behavior**

        Churn patterns differ across contract types.

        **2. Customer tenure**

        Tenure provides useful information about customer
        retention behavior.

        **3. Payment behavior**

        Different payment methods show different levels
        of churn in the dataset.

        **4. Customer charges**

        Monthly charges can provide useful signals for
        identifying customers at risk of churn.

        **Important:** These relationships show patterns
        in the data; they do not by themselves prove
        that one factor causes churn.
        """
    )


# =========================================================
# PAGE 3 — CHURN PREDICTION
# =========================================================

elif page == "🔮 Churn Prediction":

    st.title("🔮 Customer Churn Prediction")

    st.markdown(
        """
        Enter customer information below to estimate the
        customer's probability of churn.
        """
    )

    st.divider()


    # =====================================================
    # CUSTOMER INFORMATION
    # =====================================================

    st.header("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"]
        )


    with col2:

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["No", "Yes"]
        )


    with col3:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No phone service",
                "No",
                "Yes"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )


    # =====================================================
    # SERVICES
    # =====================================================

    st.header("🛠️ Services")

    col1, col2, col3 = st.columns(3)

    with col1:

        online_security = st.selectbox(
            "Online Security",
            [
                "No internet service",
                "No",
                "Yes"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No internet service",
                "No",
                "Yes"
            ]
        )


    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No internet service",
                "No",
                "Yes"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No internet service",
                "No",
                "Yes"
            ]
        )


    with col3:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No internet service",
                "No",
                "Yes"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No internet service",
                "No",
                "Yes"
            ]
        )


    # =====================================================
    # BILLING
    # =====================================================

    st.header("💳 Billing Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"]
        )


    with col2:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )


    with col3:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )


    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=float(
            monthly_charges * tenure
        )
    )


    st.divider()


    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    if st.button(
        "🔮 Predict Customer Churn",
        type="primary",
        use_container_width=True
    ):

        # -------------------------------------------------
        # CREATE CUSTOMER DATAFRAME
        # -------------------------------------------------

        customer = pd.DataFrame({

            "gender": [gender],

            "SeniorCitizen": [
                1 if senior_citizen == "Yes"
                else 0
            ],

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

            "MonthlyCharges": [
                monthly_charges
            ],

            "TotalCharges": [
                total_charges
            ]
        })


        # -------------------------------------------------
        # ONE-HOT ENCODING
        # -------------------------------------------------

        customer_encoded = pd.get_dummies(
            customer,
            drop_first=True
        )


        # -------------------------------------------------
        # MATCH MODEL COLUMNS
        # -------------------------------------------------

        customer_encoded = customer_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )


        # -------------------------------------------------
        # SCALE NUMERICAL FEATURES
        # -------------------------------------------------

        num_cols = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]

        customer_encoded[num_cols] = (
            scaler.transform(
                customer_encoded[num_cols]
            )
        )


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            customer_encoded
        )[0]

        probability = model.predict_proba(
            customer_encoded
        )[0][1]

        probability_percent = (
            probability * 100
        )


        # =================================================
        # DISPLAY RESULT
        # =================================================

        st.subheader("🎯 Prediction Result")


        if probability_percent < 30:

            st.success(
                f"🟢 LOW CHURN RISK\n\n"
                f"Churn Probability: "
                f"{probability_percent:.2f}%"
            )

            st.write(
                "This customer has a relatively low "
                "predicted probability of churn."
            )


        elif probability_percent < 60:

            st.warning(
                f"🟡 MEDIUM CHURN RISK\n\n"
                f"Churn Probability: "
                f"{probability_percent:.2f}%"
            )

            st.write(
                "This customer has a moderate predicted "
                "probability of churn."
            )


        else:

            st.error(
                f"🔴 HIGH CHURN RISK\n\n"
                f"Churn Probability: "
                f"{probability_percent:.2f}%"
            )

            st.write(
                "This customer has a relatively high "
                "predicted probability of churn."
            )


        # -------------------------------------------------
        # PROBABILITY BAR
        # -------------------------------------------------

        st.write("### Churn Probability")

        st.progress(
            int(probability_percent)
        )

        st.write(
            f"**{probability_percent:.2f}%**"
        )


        # -------------------------------------------------
        # BUSINESS ACTION
        # -------------------------------------------------

        st.subheader("💡 Suggested Business Action")

        if probability_percent >= 60:

            st.write(
                """
                Consider prioritizing this customer for
                proactive retention efforts, such as
                personalized customer support or a
                retention offer.
                """
            )

        elif probability_percent >= 30:

            st.write(
                """
                Consider monitoring this customer and
                evaluating whether additional engagement
                could improve retention.
                """
            )

        else:

            st.write(
                """
                The customer currently has a relatively
                low predicted churn risk. Continue normal
                customer engagement.
                """
            )


        st.caption(
            "Note: The model predicts churn risk based "
            "on historical patterns. It does not prove "
            "that a customer will churn or identify "
            "causal reasons for churn."
        )
        # =========================================================
# PAGE 4 — BUSINESS INSIGHTS
# =========================================================

elif page == "💡 Business Insights":

    st.title("💡 Business Insights")

    st.markdown(
        """
        This page translates churn patterns into
        **business impact and recommended actions**.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # REVENUE IMPACT
    # -----------------------------------------------------
    st.subheader("💰 Revenue Impact of Churn")

    churned_df = df[df["Churn"] == "Yes"]

    monthly_revenue_lost = churned_df["MonthlyCharges"].sum()
    annual_revenue_lost = monthly_revenue_lost * 12

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Customers Churned",
            f"{len(churned_df):,}"
        )

    with col2:
        st.metric(
            "Monthly Revenue Lost",
            f"${monthly_revenue_lost:,.0f}"
        )

    with col3:
        st.metric(
            "Projected Annual Loss",
            f"${annual_revenue_lost:,.0f}"
        )

    st.divider()

    # -----------------------------------------------------
    # HIGH-RISK SEGMENTS
    # -----------------------------------------------------
    st.subheader("🎯 Highest-Risk Segments")

    segment_summary = (
        df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .round(1)
        .reset_index(name="Churn Rate (%)")
        .sort_values("Churn Rate (%)", ascending=False)
    )

    st.dataframe(
        segment_summary,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "Contract type is one of the clearest churn signals "
        "in this dataset. Segments with the highest churn "
        "rate above represent the best targets for retention "
        "campaigns."
    )

    st.divider()

    # -----------------------------------------------------
    # PAYMENT METHOD RISK
    # -----------------------------------------------------
    st.subheader("💳 Churn Rate by Payment Method")

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
        hide_index=True
    )

    st.divider()

    # -----------------------------------------------------
    # RECOMMENDED ACTIONS
    # -----------------------------------------------------
    st.subheader("📋 Recommended Actions")

    st.markdown(
        """
        **1. Convert month-to-month customers**
        Offer incentives (discounted rate, free add-on month)
        to move month-to-month customers onto 1-year or 2-year
        contracts, which show consistently lower churn.

        **2. Target new customers early**
        Customers with low tenure churn more often. Introduce
        a structured onboarding or check-in process during the
        first 90 days.

        **3. Review electronic check payment experience**
        Customers paying by electronic check tend to show
        higher churn. Encouraging automatic payment methods
        (bank transfer / credit card) may improve retention.

        **4. Bundle protective services**
        Customers without Online Security or Tech Support
        show higher churn. Consider bundling these services
        into standard packages for at-risk segments.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # RETENTION CAMPAIGN SIMULATOR
    # -----------------------------------------------------
    st.subheader("🧮 Retention Campaign Simulator")

    st.write(
        "Estimate the potential savings from a retention "
        "campaign that reduces churn by a given percentage."
    )

    reduction_pct = st.slider(
        "Assumed reduction in churn rate (%)",
        min_value=0,
        max_value=50,
        value=10
    )

    customers_saved = int(
        len(churned_df) * (reduction_pct / 100)
    )

    revenue_saved = (
        customers_saved
        * df["MonthlyCharges"].mean()
        * 12
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Customers Retained",
            f"{customers_saved:,}"
        )

    with col2:
        st.metric(
            "Estimated Annual Savings",
            f"${revenue_saved:,.0f}"
        )

    st.caption(
        "Note: These figures are illustrative estimates based "
        "on historical averages, not guaranteed outcomes."
    )
