import streamlit as st
from src.predict import explain_customer

st.set_page_config(page_title="Bank Customer Churn Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Retail Banking Churn & Retention Engine")
st.markdown("Predict customer attrition risk and surface top **SHAP** drivers for proactive retention.")

with st.form("customer_form"):
    col1, col2 = st.columns(2)
    with col1:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 92, 42)
        tenure = st.slider("Tenure (Years)", 0, 10, 5)
        credit_score = st.number_input("Credit Score", 300, 850, 650)
    with col2:
        balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 75000.0, step=1000.0)
        estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 80000.0, step=1000.0)
        num_products = st.selectbox("Number of Bank Products", [1, 2, 3, 4], index=0)
        has_cr_card = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    submitted = st.form_submit_button("Analyze Churn Risk", use_container_width=True)

if submitted:
    customer_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": estimated_salary,
    }
    result = explain_customer(customer_data)

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Churn Probability", f"{result['churn_probability']}%")
    m2.metric("Risk Tier", result["risk_tier"])

    st.subheader("Top 3 SHAP Risk Drivers")
    for d in result["top_drivers"]:
        icon = "🔴" if d["impact"] > 0 else "🟢"
        st.write(f"{icon} **{d['feature']}** — `{d['effect']}` (SHAP impact: `{d['impact']:+}`)")