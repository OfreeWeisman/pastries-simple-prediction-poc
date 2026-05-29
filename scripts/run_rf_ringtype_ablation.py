from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_DIR = Path("data")
NORMAL_TRAIN_PATH = DATA_DIR / "compas-2x_pastries_features_sample_train_feature_vectors.csv"
NORMAL_TEST_PATH = DATA_DIR / "compas-2x_pastries_features_sample_test_feature_vectors.csv"
HIDDEN_TRAIN_PATH = DATA_DIR / "compas-2x_pastries_features_sample_train_hidden_ring_types_feature_vectors.csv"
HIDDEN_TEST_PATH = DATA_DIR / "compas-2x_pastries_features_sample_test_hidden_ring_types_feature_vectors.csv"

TARGET_COLUMNS = ["homo", "lumo", "gap", "aip", "aea", "energy"]
ALL_TARGET_COLUMNS = {
    "homo",
    "lumo",
    "homo-1",
    "lumo+1",
    "gap",
    "zero_point_energy",
    "dispersion",
    "energy",
    "aip",
    "aea",
    "dipole_norm",
    "homo_corr",
    "lumo_corr",
    "gap_corr",
    "energy_corr",
    "aip_corr",
    "aea_corr",
    "nfod",
}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in ALL_TARGET_COLUMNS
        and column != "name"
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rf_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=1,
    )


def evaluate_dataset(
    dataset_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics = []
    importances = []
    x_train = train[features]
    x_test = test[features]

    for target in TARGET_COLUMNS:
        y_train = train[target]
        y_test = test[target]

        baseline = np.full(len(test), float(y_train.mean()))
        baseline_mae = mean_absolute_error(y_test, baseline)
        test_std = y_test.std(ddof=0)
        train_std = y_train.std(ddof=0)
        metrics.append(
            {
                "dataset": dataset_name,
                "target": target,
                "model": "train_mean_baseline",
                "feature_count": len(features),
                "mae": baseline_mae,
                "rmse": rmse(y_test, baseline),
                "r2": r2_score(y_test, baseline),
                "mae_over_test_std": baseline_mae / test_std if test_std else np.nan,
                "mae_over_train_std": baseline_mae / train_std if train_std else np.nan,
                "mae_percent": 100 * baseline_mae / y_test.abs().mean(),
            }
        )

        model = rf_model()
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        mae = mean_absolute_error(y_test, predicted)
        metrics.append(
            {
                "dataset": dataset_name,
                "target": target,
                "model": "random_forest",
                "feature_count": len(features),
                "mae": mae,
                "rmse": rmse(y_test, predicted),
                "r2": r2_score(y_test, predicted),
                "mae_over_test_std": mae / test_std if test_std else np.nan,
                "mae_over_train_std": mae / train_std if train_std else np.nan,
                "mae_percent": 100 * mae / y_test.abs().mean(),
            }
        )

        for feature, importance in zip(features, model.feature_importances_):
            importances.append(
                {
                    "dataset": dataset_name,
                    "target": target,
                    "feature": feature,
                    "importance": importance,
                }
            )

    return metrics, importances


def main() -> None:
    normal_train = pd.read_csv(NORMAL_TRAIN_PATH)
    normal_test = pd.read_csv(NORMAL_TEST_PATH)
    hidden_train = pd.read_csv(HIDDEN_TRAIN_PATH)
    hidden_test = pd.read_csv(HIDDEN_TEST_PATH)

    features = feature_columns(normal_train)
    hidden_features = feature_columns(hidden_train)
    if features != hidden_features:
        raise ValueError("Normal and hidden-ring feature vectors do not have identical feature columns.")

    all_metrics = []
    all_importances = []
    for dataset_name, train, test in [
        ("with_ring_types", normal_train, normal_test),
        ("hidden_ring_types", hidden_train, hidden_test),
    ]:
        metrics, importances = evaluate_dataset(dataset_name, train, test, features)
        all_metrics.extend(metrics)
        all_importances.extend(importances)

    metrics_frame = pd.DataFrame(all_metrics).sort_values(["target", "dataset", "model"])
    importances_frame = pd.DataFrame(all_importances).sort_values(
        ["dataset", "target", "importance"],
        ascending=[True, True, False],
    )

    metrics_frame.to_csv(DATA_DIR / "rf_ringtype_ablation_metrics_10k_80_20.csv", index=False)
    importances_frame.to_csv(DATA_DIR / "rf_ringtype_ablation_importances_10k_80_20.csv", index=False)
    importances_frame.groupby(["dataset", "target"]).head(20).to_csv(
        DATA_DIR / "rf_ringtype_ablation_top20_10k_80_20.csv",
        index=False,
    )

    print(f"Feature count: {len(features)}")
    print("Wrote data/rf_ringtype_ablation_metrics_10k_80_20.csv")
    print("Wrote data/rf_ringtype_ablation_importances_10k_80_20.csv")
    print("Wrote data/rf_ringtype_ablation_top20_10k_80_20.csv")


if __name__ == "__main__":
    main()
