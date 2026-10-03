
# ============================================
# TELCO CUSTOMER CHURN - ML WEB APP
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():

    return joblib.load("churn_model.pkl")


pipeline = load_model()


# ============================================
# LOAD PREDICTIONS
# ============================================

@st.cache_data
def load_predictions():

    try:

        return pd.read_csv(
            "churn_predictions.csv"
        )

    except FileNotFoundError:

        return None


results = load_predictions()


# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("📊 Telco Analytics")

st.sidebar.markdown(
    """
    ### Navigation
    """
)

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "🔮 Customer Prediction"
    ]
)


st.sidebar.divider()

st.sidebar.info(
    """
    **Model**

    Logistic Regression

    **Dataset**

    Telco Customer Churn

    **Task**

    Binary Classification
    """
)


# ============================================
# DASHBOARD
# ============================================

if page == "🏠 Dashboard":

    st.title(
        "📊 Telco Customer Churn Dashboard"
    )

    st.markdown(
        """
        ### Machine Learning Analytics

        This dashboard provides insights into customer
        churn predictions using a Logistic Regression model.
        """
    )

    # ========================================
    # CHECK DATA
    # ========================================

    if results is None:

        st.warning(
            """
            `churn_predictions.csv` was not found.

            Run the training script first.
            """
        )

        st.stop()

    # ========================================
    # KPI CALCULATIONS
    # ========================================

    total_customers = len(results)

    actual_churn = results[
        "Actual_Churn"
    ].sum()

    predicted_churn = results[
        "Predicted_Churn"
    ].sum()

    average_probability = results[
        "Churn_Probability"
    ].mean()

    # ========================================
    # KPI CARDS
    # ========================================

    st.subheader("📌 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Test Customers",
            total_customers
        )

    with col2:

        st.metric(
            "Actual Churn",
            int(actual_churn)
        )

    with col3:

        st.metric(
            "Predicted Churn",
            int(predicted_churn)
        )

    with col4:

        st.metric(
            "Avg. Churn Probability",
            f"{average_probability * 100:.1f}%"
        )

    st.divider()

    # ========================================
    # CHURN DISTRIBUTION
    # ========================================

    st.subheader(
        "📈 Actual vs Predicted Churn"
    )

    col1, col2 = st.columns(2)

    # ----------------------------------------
    # Actual Churn
    # ----------------------------------------

    with col1:

        actual_counts = (
            results["Actual_Churn"]
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        ax.bar(
            ["No Churn", "Churn"],
            [
                actual_counts.get(0, 0),
                actual_counts.get(1, 0)
            ]
        )

        ax.set_ylabel(
            "Number of Customers"
        )

        ax.set_title(
            "Actual Churn Distribution"
        )

        st.pyplot(fig)

    # ----------------------------------------
    # Predicted Churn
    # ----------------------------------------

    with col2:

        predicted_counts = (
            results["Predicted_Churn"]
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        ax.bar(
            ["No Churn", "Churn"],
            [
                predicted_counts.get(0, 0),
                predicted_counts.get(1, 0)
            ]
        )

        ax.set_ylabel(
            "Number of Customers"
        )

        ax.set_title(
            "Predicted Churn Distribution"
        )

        st.pyplot(fig)

    st.divider()

    # ========================================
    # CONFUSION MATRIX
    # ========================================

    st.subheader(
        "🎯 Confusion Matrix"
    )

    cm = confusion_matrix(
        results["Actual_Churn"],
        results["Predicted_Churn"]
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Churn",
            "Churn"
        ]
    )

    disp.plot(
        ax=ax
    )

    ax.set_title(
        "Customer Churn Confusion Matrix"
    )

    st.pyplot(fig)

    st.divider()

    # ========================================
    # ROC CURVE
    # ========================================

    st.subheader(
        "📈 ROC Curve"
    )

    y_true = results[
        "Actual_Churn"
    ]

    y_probability = results[
        "Churn_Probability"
    ]

    auc_score = roc_auc_score(
        y_true,
        y_probability
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        y_probability
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {auc_score:.3f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve"
    )

    ax.legend()

    st.pyplot(fig)

    st.divider()

    # ========================================
    # PROBABILITY DISTRIBUTION
    # ========================================

    st.subheader(
        "🔍 Churn Probability Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.hist(
        results.loc[
            results["Actual_Churn"] == 0,
            "Churn_Probability"
        ],
        bins=20,
        alpha=0.6,
        label="No Churn"
    )

    ax.hist(
        results.loc[
            results["Actual_Churn"] == 1,
            "Churn_Probability"
        ],
        bins=20,
        alpha=0.6,
        label="Churn"
    )

    ax.set_xlabel(
        "Predicted Churn Probability"
    )

    ax.set_ylabel(
        "Number of Customers"
    )

    ax.set_title(
        "Churn Probability Distribution"
    )

    ax.legend()

    st.pyplot(fig)


# ============================================
# CUSTOMER PREDICTION
# ============================================

elif page == "🔮 Customer Prediction":

    st.title(
        "🔮 Customer Churn Prediction"
    )

    st.markdown(
        """
        Enter customer information below and the
        machine learning model will estimate the
        probability of churn.
        """
    )

    # ========================================
    # CUSTOMER INFORMATION
    # ========================================

    st.subheader(
        "👤 Customer Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with col2:

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

    with col3:

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

    with col2:

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=12
        )

    with col3:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

    # ========================================
    # SERVICES
    # ========================================

    st.subheader(
        "📡 Services"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )

    with col2:

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

    with col3:

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col3:

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col2:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col3:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

    # ========================================
    # BILLING
    # ========================================

    st.subheader(
        "💳 Billing"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
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
            max_value=200.0,
            value=70.0,
            step=0.01
        )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=840.0,
        step=0.01
    )

    # ========================================
    # PREDICT
    # ========================================

    st.divider()

    predict = st.button(
        "🚀 PREDICT CUSTOMER CHURN",
        use_container_width=True
    )

    if predict:

        customer = pd.DataFrame({

            "gender": [gender],

            "SeniorCitizen": [
                senior_citizen
            ],

            "Partner": [
                partner
            ],

            "Dependents": [
                dependents
            ],

            "tenure": [
                tenure
            ],

            "PhoneService": [
                phone_service
            ],

            "MultipleLines": [
                multiple_lines
            ],

            "InternetService": [
                internet_service
            ],

            "OnlineSecurity": [
                online_security
            ],

            "OnlineBackup": [
                online_backup
            ],

            "DeviceProtection": [
                device_protection
            ],

            "TechSupport": [
                tech_support
            ],

            "StreamingTV": [
                streaming_tv
            ],

            "StreamingMovies": [
                streaming_movies
            ],

            "Contract": [
                contract
            ],

            "PaperlessBilling": [
                paperless_billing
            ],

            "PaymentMethod": [
                payment_method
            ],

            "MonthlyCharges": [
                monthly_charges
            ],

            "TotalCharges": [
                total_charges
            ]
        })

        # ====================================
        # MODEL PREDICTION
        # ====================================

        prediction = pipeline.predict(
            customer
        )[0]

        probability = pipeline.predict_proba(
            customer
        )[0][1]

        probability_percent = (
            probability * 100
        )

        st.divider()

        st.subheader(
            "🎯 Prediction Result"
        )

        col1, col2 = st.columns(2)

        # ====================================
        # RESULT
        # ====================================

        with col1:

            if prediction == 1:

                st.error(
                    "🔴 HIGH CHURN RISK"
                )

                st.write(
                    "The model predicts that "
                    "this customer is likely to churn."
                )

            else:

                st.success(
                    "🟢 LOW CHURN RISK"
                )

                st.write(
                    "The model predicts that "
                    "this customer is likely to stay."
                )

        # ====================================
        # PROBABILITY
        # ====================================

        with col2:

            st.metric(
                "Churn Probability",
                f"{probability_percent:.2f}%"
            )

            st.progress(
                float(probability)
            )

        # ====================================
        # PROBABILITY INTERPRETATION
        # ====================================

        if probability >= 0.7:

            st.error(
                "⚠️ This customer has a high predicted churn probability."
            )

        elif probability >= 0.4:

            st.warning(
                "⚠️ This customer has a moderate predicted churn probability."
            )

        else:

            st.success(
                "✅ This customer has a relatively low predicted churn probability."
            )

        # ====================================
        # CUSTOMER DATA
        # ====================================

        with st.expander(
            "👁️ View Customer Data"
        ):

            st.dataframe(
                customer,
                use_container_width=True
            )


# ============================================
# FOOTER
# ============================================

st.divider()

st.caption(
    "Telco Customer Churn Prediction | "
    "Machine Learning Portfolio Project"
)
