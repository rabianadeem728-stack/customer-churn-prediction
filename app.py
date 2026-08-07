import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD MODEL FILES
# =========================================================

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")


# =========================================================
# LOAD ORIGINAL DATASET
# =========================================================

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna(subset=["TotalCharges"])


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("📊 Customer Churn")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🔮 Churn Prediction"
    ]
)


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "🏠 Overview":

    st.title("📊 Customer Churn Prediction")

    st.markdown(
        """
        ### Welcome

        This project uses **machine learning to predict customer churn**.

        The objective is to identify customers who may be at risk of
        leaving a company based on their demographic, service,
        contract, and billing information.
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

    avg_monthly_charges = df["MonthlyCharges"].mean()

    avg_tenure = df["tenure"].mean()


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

    st.bar_chart(
        churn_counts.set_index("Churn")
    )


    # -----------------------------------------------------
    # CONTRACT ANALYSIS
    # -----------------------------------------------------

    st.subheader("📄 Churn by Contract Type")

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"]
    )

    st.bar_chart(contract_churn)


    # -----------------------------------------------------
    # TENURE ANALYSIS
    # -----------------------------------------------------

    st.subheader("📅 Tenure Distribution by Churn")

    tenure_churn = (
        df.groupby("Churn")["tenure"]
        .mean()
        .round(2)
    )

    st.bar_chart(tenure_churn)


    st.divider()


    # -----------------------------------------------------
    # PROJECT INFORMATION
    # -----------------------------------------------------

    st.subheader("🎯 Project Objective")

    st.write(
        """
        The goal of this project is to build a machine learning
        system that predicts whether a customer is likely to churn.

        The project includes:

        • Exploratory Data Analysis

        • Data preprocessing

        • One-hot encoding

        • Feature scaling

        • SMOTE for class imbalance

        • XGBoost classification

        • Model evaluation

        • Interactive Streamlit prediction

        • Deployment through Streamlit Community Cloud
        """
    )


# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "🔮 Churn Prediction":

    st.title("🔮 Customer Churn Prediction")

    st.write(
        "Enter customer information below to estimate "
        "their probability of churn."
    )

    st.divider()


    # -----------------------------------------------------
    # CUSTOMER INFORMATION
    # -----------------------------------------------------

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
            ["No phone service", "No", "Yes"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )


    # -----------------------------------------------------
    # SERVICES
    # -----------------------------------------------------

    st.header("🛠️ Services")

    col1, col2, col3 = st.columns(3)

    with col1:

        online_security = st.selectbox(
            "Online Security",
            ["No internet service", "No", "Yes"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No internet service", "No", "Yes"]
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            ["No internet service", "No", "Yes"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No internet service", "No", "Yes"]
        )

    with col3:

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No internet service", "No", "Yes"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No internet service", "No", "Yes"]
        )


    # -----------------------------------------------------
    # BILLING
    # -----------------------------------------------------

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
        value=float(monthly_charges * tenure)
    )


    st.divider()


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if st.button(
        "🔮 Predict Customer Churn",
        type="primary",
        use_container_width=True
    ):

        customer = pd.DataFrame({

            "gender": [gender],

            "SeniorCitizen": [
                1 if senior_citizen == "Yes" else 0
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

            "MonthlyCharges": [monthly_charges],

            "TotalCharges": [total_charges]
        })


        # One-hot encoding
        customer_encoded = pd.get_dummies(
            customer,
            drop_first=True
        )


        # Match training columns
        customer_encoded = customer_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )


        # Scale numerical columns
        num_cols = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]

        customer_encoded[num_cols] = scaler.transform(
            customer_encoded[num_cols]
        )


        # Prediction
        prediction = model.predict(
            customer_encoded
        )[0]

        probability = model.predict_proba(
            customer_encoded
        )[0][1]

        probability_percent = probability * 100


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.subheader("🎯 Prediction Result")


        if probability_percent < 30:

            st.success(
                f"🟢 LOW CHURN RISK\n\n"
                f"Churn Probability: "
                f"{probability_percent:.2f}%"
            )

        elif probability_percent < 60:

            st.warning(
                f"🟡 MEDIUM CHURN RISK\n\n"
                f"Churn Probability: "
                f"{probability_percent:.2f}%"
            )

        else:

            st.error(
                f"🔴 HIGH CHURN RISK\n\n"
                f"Churn Probability: "
                f"{probability_percent:.2f}%"
            )


        st.progress(
            int(probability_percent)
        )

        st.write(
            f"**Estimated probability of churn: "
            f"{probability_percent:.2f}%**"
        )
