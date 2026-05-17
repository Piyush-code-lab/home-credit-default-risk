import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def drop_high_missing_cols(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    missing_percent = df.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing")
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
    print("Missing values filled")
    return df


def cap_outliers(df: pd.DataFrame, cols: list, percentile: float) -> pd.DataFrame:
    for col in cols:
        upper = df[col].quantile(percentile)
        df[col] = df[col].clip(upper=upper)
    print(f"Outliers capped at {percentile*100}th percentile for {cols}")
    return df


def create_age_feature(df: pd.DataFrame) -> pd.DataFrame:
    df["age"] = abs(df["DAYS_BIRTH"]) / 365
    return df


def fix_days_employed(df: pd.DataFrame) -> pd.DataFrame:
    df["DAYS_EMPLOYED_ANOMALY"] = np.where(
        df["DAYS_EMPLOYED"] == 365243, 1, 0
    )
    median_employed = df.loc[
        df["DAYS_EMPLOYED"] != 365243, "DAYS_EMPLOYED"
    ].median()
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, median_employed)
    df["EMPLOYED_YEARS"] = abs(df["DAYS_EMPLOYED"]) / 365
    return df


def encode_categoricals(df: pd.DataFrame,label_cols: list,onehot_cols: list) -> pd.DataFrame:
    le = LabelEncoder()
    for col in label_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
    print(f"Encoding done. Shape: {df.shape}")
    return df


def run_preprocessing(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    threshold = config["preprocessing"]["missing_threshold"]
    percentile = config["preprocessing"]["outlier_cap_percentile"]
    label_cols = config["preprocessing"]["label_encode_cols"]
    onehot_cols = config["preprocessing"]["onehot_cols"]
    outlier_cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]

    df = drop_high_missing_cols(df, threshold)
    df = fill_missing_values(df)
    df = cap_outliers(df, outlier_cols, percentile)
    df = create_age_feature(df)
    df = fix_days_employed(df)
    df = encode_categoricals(df, label_cols, onehot_cols)

    print(f"Preprocessing complete. Final shape: {df.shape}")
    return df