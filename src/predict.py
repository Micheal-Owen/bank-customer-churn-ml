import joblib
import pandas as pd
import shap
from src.features import engineer_features

ACTION_PLAYBOOK = {
    "NumOfProducts": "Product Consolidation Review: Audit account for unwanted bundled fees; offer fee-free tier restructuring.",
    "IsActiveMember": "Digital Re-Engagement: Trigger mobile app cashback promo or relationship manager check-in call.",
    "Age": "Wealth & Retirement Advisory: Assign senior relationship officer for tailored fixed-deposit or pension options.",
    "Balance": "High-Value Depositor Retention: Offer preferential savings interest rate or zero-fee transfers.",
    "Balance_to_Salary": "Primary-Bank Loyalty Package: Pitch salary-account perks and overdraft protection.",
    "Geography_Germany": "Regional Retention Desk: Route to Germany specialist retention queue with competitive rate match.",
}
DEFAULT_ACTION = "Standard Retention Outreach: Send personalized service satisfaction survey + loyalty reward points."


def explain_customer(customer_dict: dict, model_path: str = "models/churn_lgbm_v1.joblib") -> dict:
    """Predict churn probability and return top 3 SHAP drivers plus recommended retention action."""
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_names = artifact["feature_names"]

    # Prepare single row and align columns with training schema
    raw_df = pd.DataFrame([customer_dict])
    processed = engineer_features(raw_df)
    aligned = processed.reindex(columns=feature_names, fill_value=0)

    prob = float(model.predict_proba(aligned)[0, 1])

    # Local SHAP explanation for this specific customer
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(aligned)
    row_shap = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

    contributions = sorted(
        zip(feature_names, row_shap),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    # Pick the strongest risk-increasing driver for the retention action
    risk_drivers = [feat for feat, val in contributions if val > 0]
    primary_risk_feature = risk_drivers[0] if risk_drivers else contributions[0][0]
    recommended_action = ACTION_PLAYBOOK.get(primary_risk_feature, DEFAULT_ACTION)

    return {
        "churn_probability": round(prob * 100, 2),
        "risk_tier": "High" if prob >= 0.65 else ("Medium" if prob >= 0.35 else "Low"),
        "top_drivers": [
            {"feature": feat, "impact": round(float(val), 3), "effect": "Increases Risk" if val > 0 else "Reduces Risk"}
            for feat, val in contributions
        ],
        "recommended_action": recommended_action,
    }