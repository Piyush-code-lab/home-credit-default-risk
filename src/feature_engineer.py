import numpy as np
import pandas as pd


def add_main_features(df: pd.DataFrame) -> pd.DataFrame:
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["DAYS_EMPLOYED_RATIO"] = df["EMPLOYED_YEARS"] / df["age"]
    df["AGE_INCOME_RATIO"] = df["age"] / df["AMT_INCOME_TOTAL"]
    print("Main features added")
    return df


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    total_credits = bureau.groupby("SK_ID_CURR")["SK_ID_BUREAU"].count()
    active_credits = bureau[bureau["CREDIT_ACTIVE"] == "Active"].groupby("SK_ID_CURR")["SK_ID_BUREAU"].count()
    closed_credits = bureau[bureau["CREDIT_ACTIVE"] == "Closed"].groupby("SK_ID_CURR")["SK_ID_BUREAU"].count()
    total_days_overdue = bureau.groupby("SK_ID_CURR")["CREDIT_DAY_OVERDUE"].sum()
    max_days_overdue = bureau.groupby("SK_ID_CURR")["CREDIT_DAY_OVERDUE"].max()
    total_overdue_amt = bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_OVERDUE"].sum()
    total_credit_sum = bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM"].sum()
    total_debt_sum = bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"].sum()

    bureau_agg = pd.DataFrame({
        "TOTAL_CREDITS": total_credits,
        "ACTIVE_CREDITS": active_credits,
        "CLOSED_CREDITS": closed_credits,
        "TOTAL_DAYS_OVERDUE": total_days_overdue,
        "MAX_DAYS_OVERDUE": max_days_overdue,
        "TOTAL_OVERDUE_AMT": total_overdue_amt,
        "TOTAL_CREDIT_SUM": total_credit_sum,
        "TOTAL_DEBT_SUM": total_debt_sum,
    }).fillna(0)

    bureau_agg["TOTAL_DEBT_SUM"] = bureau_agg["TOTAL_DEBT_SUM"].clip(lower=0)
    bureau_agg["DEBT_TO_CREDIT_RATIO"] = np.where(
        bureau_agg["TOTAL_CREDIT_SUM"] != 0,
        bureau_agg["TOTAL_DEBT_SUM"] / bureau_agg["TOTAL_CREDIT_SUM"],
        0
    ).clip(0)

    for col in ["MAX_DAYS_OVERDUE", "TOTAL_DAYS_OVERDUE"]:
        upper = bureau_agg[col].quantile(0.99)
        bureau_agg[col] = bureau_agg[col].clip(upper=upper)

    print(f"Bureau aggregated: {bureau_agg.reset_index().shape}")
    return bureau_agg.reset_index()


def aggregate_previous_app(prev: pd.DataFrame) -> pd.DataFrame:
    prev["CREDIT_APP_RATIO"] = np.where(
        prev["AMT_APPLICATION"] != 0,
        prev["AMT_CREDIT"] / prev["AMT_APPLICATION"],
        0
    )

    prev_agg = pd.DataFrame({
        "TOTAL_PREV_APPS": prev.groupby("SK_ID_CURR")["SK_ID_PREV"].count(),
        "APPROVED_APPS": prev[prev["NAME_CONTRACT_STATUS"] == "Approved"].groupby("SK_ID_CURR")["SK_ID_PREV"].count(),
        "REFUSED_APPS": prev[prev["NAME_CONTRACT_STATUS"] == "Refused"].groupby("SK_ID_CURR")["SK_ID_PREV"].count(),
        "CANCELLED_APPS": prev[prev["NAME_CONTRACT_STATUS"] == "Canceled"].groupby("SK_ID_CURR")["SK_ID_PREV"].count(),
        "UNUSED_APPS": prev[prev["NAME_CONTRACT_STATUS"] == "Unused offer"].groupby("SK_ID_CURR")["SK_ID_PREV"].count(),
        "MEAN_AMT_APPLICATION": prev.groupby("SK_ID_CURR")["AMT_APPLICATION"].mean(),
        "MAX_AMT_APPLICATION": prev.groupby("SK_ID_CURR")["AMT_APPLICATION"].max(),
        "MEAN_AMT_CREDIT": prev.groupby("SK_ID_CURR")["AMT_CREDIT"].mean(),
        "MAX_AMT_CREDIT": prev.groupby("SK_ID_CURR")["AMT_CREDIT"].max(),
        "MEAN_CREDIT_APP_RATIO": prev.groupby("SK_ID_CURR")["CREDIT_APP_RATIO"].mean(),
        "RECENT_APPLICATION_DAYS": prev.groupby("SK_ID_CURR")["DAYS_DECISION"].max(),
        "MEAN_CNT_PAYMENT": prev.groupby("SK_ID_CURR")["CNT_PAYMENT"].mean(),
    }).fillna(0)

    print(f"Previous app aggregated: {prev_agg.reset_index().shape}")
    return prev_agg.reset_index()


def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    inst["DAYS_PAYMENT_DIFF"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["AMT_PAYMENT_DIFF"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]

    late_payments = inst[inst["DAYS_PAYMENT_DIFF"] > 0].groupby("SK_ID_CURR")["DAYS_PAYMENT_DIFF"].count()
    total_installments = inst.groupby("SK_ID_CURR")["NUM_INSTALMENT_NUMBER"].count()

    inst_agg = pd.DataFrame({
        "MEAN_DAYS_PAYMENT_DIFF": inst.groupby("SK_ID_CURR")["DAYS_PAYMENT_DIFF"].mean(),
        "MAX_DAYS_PAYMENT_DIFF": inst.groupby("SK_ID_CURR")["DAYS_PAYMENT_DIFF"].max(),
        "LATE_PAYMENTS": late_payments,
        "MEAN_AMT_PAYMENT_DIFF": inst.groupby("SK_ID_CURR")["AMT_PAYMENT_DIFF"].mean(),
        "MAX_AMT_PAYMENT_DIFF": inst.groupby("SK_ID_CURR")["AMT_PAYMENT_DIFF"].max(),
        "TOTAL_INSTALLMENTS": total_installments,
        "MEAN_AMT_INSTALMENT": inst.groupby("SK_ID_CURR")["AMT_INSTALMENT"].mean(),
        "MEAN_AMT_PAYMENT": inst.groupby("SK_ID_CURR")["AMT_PAYMENT"].mean(),
    }).fillna(0)

    inst_agg["LATE_PAYMENT_RATIO"] = (
        inst_agg["LATE_PAYMENTS"] / inst_agg["TOTAL_INSTALLMENTS"]
    ).fillna(0)

    print(f"Installments aggregated: {inst_agg.reset_index().shape}")
    return inst_agg.reset_index()


def aggregate_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    months_overdue = pos[pos["SK_DPD"] > 0].groupby("SK_ID_CURR")["SK_DPD"].count()
    total_pos_records = pos.groupby("SK_ID_CURR")["SK_ID_PREV"].count()

    pos_agg = pd.DataFrame({
        "MEAN_SK_DPD": pos.groupby("SK_ID_CURR")["SK_DPD"].mean(),
        "MAX_SK_DPD": pos.groupby("SK_ID_CURR")["SK_DPD"].max(),
        "MEAN_SK_DPD_DEF": pos.groupby("SK_ID_CURR")["SK_DPD_DEF"].mean(),
        "MAX_SK_DPD_DEF": pos.groupby("SK_ID_CURR")["SK_DPD_DEF"].max(),
        "MONTHS_OVERDUE": months_overdue,
        "TOTAL_POS_RECORDS": total_pos_records,
        "COMPLETED_CONTRACTS": pos[pos["NAME_CONTRACT_STATUS"] == "Completed"].groupby("SK_ID_CURR")["SK_ID_PREV"].count(),
        "ACTIVE_CONTRACTS": pos[pos["NAME_CONTRACT_STATUS"] == "Active"].groupby("SK_ID_CURR")["SK_ID_PREV"].count(),
        "MEAN_CNT_INSTALMENT_FUTURE": pos.groupby("SK_ID_CURR")["CNT_INSTALMENT_FUTURE"].mean(),
    }).fillna(0)

    pos_agg["OVERDUE_RATIO"] = (
        pos_agg["MONTHS_OVERDUE"] / pos_agg["TOTAL_POS_RECORDS"]
    ).fillna(0)

    print(f"POS Cash aggregated: {pos_agg.reset_index().shape}")
    return pos_agg.reset_index()


def aggregate_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    cc["UTILIZATION_RATIO"] = np.where(
        cc["AMT_CREDIT_LIMIT_ACTUAL"] != 0,
        cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"],
        0
    )

    months_overdue = cc[cc["SK_DPD"] > 0].groupby("SK_ID_CURR")["SK_DPD"].count()
    total_cc_records = cc.groupby("SK_ID_CURR")["SK_ID_PREV"].count()

    cc_agg = pd.DataFrame({
        "MEAN_AMT_BALANCE": cc.groupby("SK_ID_CURR")["AMT_BALANCE"].mean(),
        "MAX_AMT_BALANCE": cc.groupby("SK_ID_CURR")["AMT_BALANCE"].max(),
        "MEAN_CREDIT_LIMIT": cc.groupby("SK_ID_CURR")["AMT_CREDIT_LIMIT_ACTUAL"].mean(),
        "MEAN_UTILIZATION_RATIO": cc.groupby("SK_ID_CURR")["UTILIZATION_RATIO"].mean(),
        "MEAN_SK_DPD_CC": cc.groupby("SK_ID_CURR")["SK_DPD"].mean(),
        "MAX_SK_DPD_CC": cc.groupby("SK_ID_CURR")["SK_DPD"].max(),
        "MEAN_SK_DPD_DEF_CC": cc.groupby("SK_ID_CURR")["SK_DPD_DEF"].mean(),
        "MAX_SK_DPD_DEF_CC": cc.groupby("SK_ID_CURR")["SK_DPD_DEF"].max(),
        "MONTHS_OVERDUE_CC": months_overdue,
        "MEAN_PAYMENT_TOTAL_CURRENT": cc.groupby("SK_ID_CURR")["AMT_PAYMENT_TOTAL_CURRENT"].mean(),
        "MEAN_DRAWINGS_CURRENT": cc.groupby("SK_ID_CURR")["AMT_DRAWINGS_CURRENT"].mean(),
        "TOTAL_CC_RECORDS": total_cc_records,
    }).fillna(0)

    cc_agg["OVERDUE_RATIO_CC"] = (
        cc_agg["MONTHS_OVERDUE_CC"] / cc_agg["TOTAL_CC_RECORDS"]
    ).fillna(0)

    print(f"Credit card aggregated: {cc_agg.reset_index().shape}")
    return cc_agg.reset_index()


def merge_all(df: pd.DataFrame,bureau_agg: pd.DataFrame,prev_agg: pd.DataFrame,inst_agg: pd.DataFrame,pos_agg: pd.DataFrame,cc_agg: pd.DataFrame) -> pd.DataFrame:
    for agg, name in [(bureau_agg, "bureau"),(prev_agg, "previous_app"),(inst_agg, "installments"),(pos_agg, "pos_cash"),(cc_agg, "credit_card"),]:
        cols = agg.columns.drop("SK_ID_CURR")
        df = df.merge(agg, on="SK_ID_CURR", how="left")
        df[cols] = df[cols].fillna(0)
        print(f"Merged {name}. Shape: {df.shape}")

    return df