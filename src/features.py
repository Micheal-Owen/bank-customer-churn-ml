import pandas as pd

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Clean raw banking customer data and create domain ratios."""
    df_feat = data.copy()

    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    df_feat = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])

    # Domain ratios
    df_feat["Balance_to_Salary"] = df_feat["Balance"] / (df_feat["EstimatedSalary"] + 1)
    df_feat["Tenure_to_Age"] = df_feat["Tenure"] / df_feat["Age"]
    df_feat["Products_Per_Tenure"] = df_feat["NumOfProducts"] / (df_feat["Tenure"] + 1)
    df_feat["Zero_Balance_Flag"] = (df_feat["Balance"] == 0).astype(int)

    # Encode categorical variables
    df_feat = pd.get_dummies(df_feat, columns=["Geography", "Gender"], drop_first=True, dtype=int)
    return df_feat