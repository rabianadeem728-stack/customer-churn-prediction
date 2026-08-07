import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# Load model files
# -----------------------------

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")


# -----------------------------
# Title
# -----------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer information below to predict "
    "the likelihood of customer churn."
)

st.divider()


# -----------------------------
# Customer Information
# -----------------------------

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


# -----------------------------
# Services
# -----------------------------

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


# -----------------------------
# Billing
# -----------------------------

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


# -----------------------------
# Prediction
# -----------------------------

if st.button(
    "🔮 Predict Customer Churn",
    type="primary",
    use_container_width=True
):

    # Create input dataframe
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


    # -----------------------------
    # One-hot encoding
    # -----------------------------

    customer_encoded = pd.get_dummies(
        customer,
        drop_first=True
    )


    # -----------------------------
    # Match training columns
    # -----------------------------

    customer_encoded = customer_encoded.reindex(
        columns=model_columns,
        fill_value=0
    )


    # -----------------------------
    # Scale numerical variables
    # -----------------------------

    num_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    customer_encoded[num_cols] = scaler.transform(
        customer_encoded[num_cols]
    )


    # -----------------------------
    # Make prediction
    # -----------------------------

    prediction = model.predict(
        customer_encoded
    )[0]

    probability = model.predict_proba(
        customer_encoded
    )[0][1]

    probability_percent = probability * 100


    # -----------------------------
    # Display result
    # -----------------------------

    st.header("🎯 Prediction Result")

    if prediction == 1:

        st.error(
            f"🔴 High Risk of Churn\n\n"
            f"Estimated churn probability: "
            f"{probability_percent:.2f}%"
        )

        st.warning(
            "This customer is predicted to be at high risk "
            "of leaving the company."
        )

    else:

        st.success(
            f"🟢 Low Risk of Churn\n\n"
            f"Estimated churn probability: "
            f"{probability_percent:.2f}%"
        )

        st.info(
            "This customer is predicted to be at lower risk "
            "of leaving the company."
        )


    # -----------------------------
    # Probability bar
    # -----------------------------

    st.subheader("Churn Probability")

    st.progress(
        int(probability_percent)
    )

    st.write(
        f"**{probability_percent:.2f}% probability of churn**"
    )
