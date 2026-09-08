import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

@st.cache_resource
def load_model():

    preprocessor = joblib.load(r"model/preprocessor.pkl")
    model = joblib.load(r"model/model.pkl")

    return preprocessor, model


preprocessor, model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("💳 Credit Card Default Prediction")

st.write(
    """
    Enter the customer's information below to predict whether
    the customer is likely to default on their credit card payment
    next month.
    """
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    limit_bal = st.number_input(
        "Credit Limit (LIMIT_BAL)",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )


with col2:

    education = st.number_input(
        "Education",
        min_value=0,
        max_value=6,
        value=2,
        step=1
    )

    pay_0 = st.number_input(
        "PAY_0",
        min_value=-2,
        max_value=9,
        value=0,
        step=1
    )


with col3:

    pay_2 = st.number_input(
        "PAY_2",
        min_value=-2,
        max_value=9,
        value=0,
        step=1
    )

    pay_3 = st.number_input(
        "PAY_3",
        min_value=-2,
        max_value=9,
        value=0,
        step=1
    )


st.subheader("📅 Payment History")

col1, col2, col3 = st.columns(3)


with col1:

    pay_4 = st.number_input(
        "PAY_4",
        min_value=-2,
        max_value=9,
        value=0,
        step=1
    )

    pay_5 = st.number_input(
        "PAY_5",
        min_value=-2,
        max_value=9,
        value=0,
        step=1
    )


with col2:

    pay_6 = st.number_input(
        "PAY_6",
        min_value=-2,
        max_value=9,
        value=0,
        step=1
    )


st.divider()


# ============================================================
# BILL AMOUNTS
# ============================================================

st.subheader("💰 Bill Amounts")

col1, col2, col3 = st.columns(3)


with col1:

    bill_amt1 = st.number_input(
        "BILL_AMT1",
        min_value=0.0,
        value=20000.0,
        step=1000.0
    )

    bill_amt2 = st.number_input(
        "BILL_AMT2",
        min_value=0.0,
        value=20000.0,
        step=1000.0
    )


with col2:

    bill_amt3 = st.number_input(
        "BILL_AMT3",
        min_value=0.0,
        value=20000.0,
        step=1000.0
    )

    bill_amt4 = st.number_input(
        "BILL_AMT4",
        min_value=0.0,
        value=20000.0,
        step=1000.0
    )


with col3:

    bill_amt5 = st.number_input(
        "BILL_AMT5",
        min_value=0.0,
        value=20000.0,
        step=1000.0
    )

    bill_amt6 = st.number_input(
        "BILL_AMT6",
        min_value=0.0,
        value=20000.0,
        step=1000.0
    )


st.divider()


# ============================================================
# PAYMENT AMOUNTS
# ============================================================

st.subheader("💵 Previous Payment Amounts")

col1, col2, col3 = st.columns(3)


with col1:

    pay_amt1 = st.number_input(
        "PAY_AMT1",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    pay_amt2 = st.number_input(
        "PAY_AMT2",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )


with col2:

    pay_amt3 = st.number_input(
        "PAY_AMT3",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    pay_amt4 = st.number_input(
        "PAY_AMT4",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )


with col3:

    pay_amt5 = st.number_input(
        "PAY_AMT5",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    pay_amt6 = st.number_input(
        "PAY_AMT6",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Default",
    type="primary",
    use_container_width=True
):

    # Create input dataframe
    input_data = pd.DataFrame([{

        "AGE": age,
        "LIMIT_BAL": limit_bal,
        "EDUCATION": education,

        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,

        "BILL_AMT1": bill_amt1,
        "BILL_AMT2": bill_amt2,
        "BILL_AMT3": bill_amt3,
        "BILL_AMT4": bill_amt4,
        "BILL_AMT5": bill_amt5,
        "BILL_AMT6": bill_amt6,

        "PAY_AMT1": pay_amt1,
        "PAY_AMT2": pay_amt2,
        "PAY_AMT3": pay_amt3,
        "PAY_AMT4": pay_amt4,
        "PAY_AMT5": pay_amt5,
        "PAY_AMT6": pay_amt6

    }])


    # ========================================================
    # PREPROCESS INPUT
    # ========================================================

    transformed_data = preprocessor.transform(input_data)


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    prediction = model.predict(transformed_data)[0]

    probability = model.predict_proba(
        transformed_data
    )[0][1]


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.subheader("📊 Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Customer is likely to DEFAULT"
        )

    else:

        st.success(
            "✅ Customer is unlikely to DEFAULT"
        )


    # Display probability

    st.metric(
        label="Default Probability",
        value=f"{probability:.2%}"
    )


    # Probability progress bar

    st.progress(float(probability))
