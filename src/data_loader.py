import pandas as pd
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_main_data(config: dict) -> pd.DataFrame:
    df = pd.read_csv(config["data"]["train_path"])
    print(f"Main data loaded: {df.shape}")
    return df


def load_bureau(config: dict) -> pd.DataFrame:
    df = pd.read_csv(config["data"]["bureau_path"])
    print(f"Bureau data loaded: {df.shape}")
    return df


def load_previous_app(config: dict) -> pd.DataFrame:
    df = pd.read_csv(config["data"]["previous_app_path"])
    print(f"Previous application data loaded: {df.shape}")
    return df


def load_installments(config: dict) -> pd.DataFrame:
    df = pd.read_csv(config["data"]["installments_path"])
    print(f"Installments data loaded: {df.shape}")
    return df


def load_pos_cash(config: dict) -> pd.DataFrame:
    df = pd.read_csv(config["data"]["pos_cash_path"])
    print(f"POS Cash data loaded: {df.shape}")
    return df


def load_credit_card(config: dict) -> pd.DataFrame:
    df = pd.read_csv(config["data"]["credit_card_path"])
    print(f"Credit card data loaded: {df.shape}")
    return df