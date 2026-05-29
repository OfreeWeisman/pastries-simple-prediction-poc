from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "compas-2x_pastries_features_sample_train_feature_vectors.csv"
TEST_PATH = DATA_DIR / "compas-2x_pastries_features_sample_test_feature_vectors.csv"

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

PAPER_LALAS_COLUMNS = [
    "paper_lalas_number_of_rings",
    "paper_lalas_number_of_branching_points",
    "paper_lalas_lal_subsequence_count",
    "paper_lalas_l_ratio",
    "paper_lalas_longest_l",
    "paper_lalas_longest_l_degeneracy",
    "paper_lalas_upper_a_ratio",
    "paper_lalas_lower_to_upper_a_ratio",
    "paper_lalas_longest_upper_a",
    "paper_lalas_longest_a_case_insensitive",
]


def all_feature_columns(frame: pd.DataFrame) -> list[str]:
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
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=1,
    )


def evaluate_feature_set(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set_name: str,
    features: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics = []
    importances = []
    x_train = train[features]
    x_test = test[features]

    for target in TARGET_COLUMNS:
        y_train = train[target]
        y_test = test[target]

        baseline_pred = np.full(len(test), float(y_train.mean()))
        baseline_mae = mean_absolute_error(y_test, baseline_pred)
        baseline_std = y_test.std(ddof=0)
        metrics.append(
            {
                "feature_set": feature_set_name,
                "target": target,
                "model": "train_mean_baseline",
                "feature_count": len(features),
                "mae": baseline_mae,
                "rmse": rmse(y_test, baseline_pred),
                "r2": r2_score(y_test, baseline_pred),
                "mae_percent": 100 * baseline_mae / y_test.abs().mean(),
                "mae_over_test_std": baseline_mae / baseline_std if baseline_std else np.nan,
                "mae_over_train_std": baseline_mae / y_train.std(ddof=0) if y_train.std(ddof=0) else np.nan,
            }
        )

        model = rf_model()
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        mae = mean_absolute_error(y_test, predicted)
        test_std = y_test.std(ddof=0)
        train_std = y_train.std(ddof=0)

        metrics.append(
            {
                "feature_set": feature_set_name,
                "target": target,
                "model": "random_forest",
                "feature_count": len(features),
                "mae": mae,
                "rmse": rmse(y_test, predicted),
                "r2": r2_score(y_test, predicted),
                "mae_percent": 100 * mae / y_test.abs().mean(),
                "mae_over_test_std": mae / test_std if test_std else np.nan,
                "mae_over_train_std": mae / train_std if train_std else np.nan,
            }
        )

        for feature, importance in zip(features, model.feature_importances_):
            importances.append(
                {
                    "feature_set": feature_set_name,
                    "target": target,
                    "feature": feature,
                    "importance": importance,
                }
            )

    return metrics, importances


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    full_features = all_feature_columns(train)

    feature_sets = {
        "paper_lalas_only": PAPER_LALAS_COLUMNS,
        "full_current_features": full_features,
    }

    all_metrics = []
    all_importances = []
    for feature_set_name, features in feature_sets.items():
        missing = [feature for feature in features if feature not in train.columns]
        if missing:
            raise ValueError(f"{feature_set_name} is missing columns: {missing}")
        metrics, importances = evaluate_feature_set(train, test, feature_set_name, features)
        all_metrics.extend(metrics)
        all_importances.extend(importances)

    metrics_frame = pd.DataFrame(all_metrics).sort_values(["target", "feature_set", "model"])
    importances_frame = pd.DataFrame(all_importances).sort_values(
        ["feature_set", "target", "importance"],
        ascending=[True, True, False],
    )

    metrics_frame.to_csv(DATA_DIR / "rf_paper_feature_comparison_metrics.csv", index=False)
    importances_frame.to_csv(DATA_DIR / "rf_paper_feature_comparison_importances.csv", index=False)
    importances_frame.groupby(["feature_set", "target"]).head(20).to_csv(
        DATA_DIR / "rf_paper_feature_comparison_top20.csv",
        index=False,
    )

    print(f"Full feature count: {len(full_features)}")
    print(f"Paper LALAS feature count: {len(PAPER_LALAS_COLUMNS)}")
    print("Wrote data/rf_paper_feature_comparison_metrics.csv")
    print("Wrote data/rf_paper_feature_comparison_importances.csv")
    print("Wrote data/rf_paper_feature_comparison_top20.csv")


if __name__ == "__main__":
    main()
