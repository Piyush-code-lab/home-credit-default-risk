from src.data_loader import (
    load_config,
    load_main_data,
    load_bureau,
    load_previous_app,
    load_installments,
    load_pos_cash,
    load_credit_card
)
from src.preprocessor import run_preprocessing
from src.feature_engineer import (
    add_main_features,
    aggregate_bureau,
    aggregate_previous_app,
    aggregate_installments,
    aggregate_pos_cash,
    aggregate_credit_card,
    merge_all
)
from src.trainer import (
    get_features_and_target,
    split_data,
    build_model,
    train_model,
    save_model,
    save_feature_columns,
    drop_low_importance_features
)
from src.evaluator import (
    evaluate_model,
    plot_roc_curve,
    plot_confusion_matrix,
    get_shap_importance,
    plot_shap_summary,
    get_low_importance_features
)


def main():
   
    config = load_config("configs/config.yaml")
    print("Config loaded.")

    df = load_main_data(config)
    bureau = load_bureau(config)
    prev = load_previous_app(config)
    inst = load_installments(config)
    pos = load_pos_cash(config)
    cc = load_credit_card(config)

    df = run_preprocessing(df, config)

    df = add_main_features(df)

    bureau_agg = aggregate_bureau(bureau)
    prev_agg = aggregate_previous_app(prev)
    inst_agg = aggregate_installments(inst)
    pos_agg = aggregate_pos_cash(pos)
    cc_agg = aggregate_credit_card(cc)

 
    df = merge_all(df, bureau_agg, prev_agg, inst_agg, pos_agg, cc_agg)
    print(f"Final dataset shape: {df.shape}")

    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, config)

    model = build_model(config)
    model = train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)

    importance_df, shap_values = get_shap_importance(model, X_test)
    cols_to_drop = get_low_importance_features(importance_df, threshold=0.001)
    X_train, X_test = drop_low_importance_features(X_train, X_test, cols_to_drop)

    model = build_model(config)
    model = train_model(model, X_train, y_train)
    auc = evaluate_model(model, X_test, y_test)
    print(f"Final ROC-AUC: {auc:.4f}")

    plot_roc_curve(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    plot_shap_summary(shap_values, X_test)

    save_model(model, config["output"]["model_path"])
    save_feature_columns(
        X_train.columns.tolist(),
        config["output"]["features_path"]
    )
    print("Pipeline complete.")


if __name__ == "__main__":
    main()