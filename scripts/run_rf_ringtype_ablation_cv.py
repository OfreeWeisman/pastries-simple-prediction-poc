from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_DIR = Path("data")
RESULTS_DIR = Path("results") / "sample"
NORMAL_SAMPLE_PATH = DATA_DIR / "compas-2x_pastries_features_sample_feature_vectors.csv"
HIDDEN_SAMPLE_PATH = DATA_DIR / "compas-2x_pastries_features_sample_hidden_ring_types_feature_vectors.csv"

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


def rf_model(n_estimators: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=1,
    )


def metric_row(
    dataset: str,
    fold: int,
    target: str,
    model: str,
    feature_count: int,
    y_train: pd.Series,
    y_test: pd.Series,
    predicted: np.ndarray,
) -> dict[str, object]:
    mae = mean_absolute_error(y_test, predicted)
    test_std = y_test.std(ddof=0)
    train_std = y_train.std(ddof=0)
    return {
        "dataset": dataset,
        "fold": fold,
        "target": target,
        "model": model,
        "feature_count": feature_count,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "mae": mae,
        "rmse": rmse(y_test, predicted),
        "r2": r2_score(y_test, predicted),
        "mae_percent": 100 * mae / y_test.abs().mean(),
        "mae_over_test_std": mae / test_std if test_std else np.nan,
        "mae_over_train_std": mae / train_std if train_std else np.nan,
    }


def make_folds(row_count: int, fold_count: int, random_state: int) -> list[np.ndarray]:
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(row_count)
    return [fold.astype(int) for fold in np.array_split(indices, fold_count)]


def run_dataset(
    dataset_name: str,
    frame: pd.DataFrame,
    features: list[str],
    folds: list[np.ndarray],
    n_estimators: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics = []
    importances = []
    all_indices = np.arange(len(frame))

    for fold_index, test_indices in enumerate(folds, start=1):
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=False)
        train = frame.iloc[train_indices]
        test = frame.iloc[test_indices]
        x_train = train[features]
        x_test = test[features]

        for target in TARGET_COLUMNS:
            y_train = train[target]
            y_test = test[target]

            baseline = np.full(len(test), float(y_train.mean()))
            metrics.append(
                metric_row(
                    dataset_name,
                    fold_index,
                    target,
                    "train_mean_baseline",
                    len(features),
                    y_train,
                    y_test,
                    baseline,
                )
            )

            model = rf_model(n_estimators)
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            metrics.append(
                metric_row(
                    dataset_name,
                    fold_index,
                    target,
                    "random_forest",
                    len(features),
                    y_train,
                    y_test,
                    predicted,
                )
            )

            for feature, importance in zip(features, model.feature_importances_):
                importances.append(
                    {
                        "dataset": dataset_name,
                        "fold": fold_index,
                        "target": target,
                        "feature": feature,
                        "importance": importance,
                    }
                )

    return metrics, importances


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["mae", "rmse", "r2", "mae_percent", "mae_over_test_std", "mae_over_train_std"]
    grouped = metrics.groupby(["dataset", "target", "model"], as_index=False)
    means = grouped[metric_columns].mean()
    stds = grouped[metric_columns].std(ddof=0).rename(
        columns={column: f"{column}_std" for column in metric_columns}
    )
    return means.merge(stds, on=["dataset", "target", "model"], how="left")


def aggregate_importances(importances: pd.DataFrame) -> pd.DataFrame:
    return (
        importances.groupby(["dataset", "target", "feature"], as_index=False)["importance"]
        .mean()
        .sort_values(["dataset", "target", "importance"], ascending=[True, True, False])
    )


def plot_metric(mean_metrics: pd.DataFrame, metric: str, ylabel: str, output_name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rf = mean_metrics[mean_metrics["model"] == "random_forest"]
    baseline = mean_metrics[mean_metrics["model"] == "train_mean_baseline"]
    targets = TARGET_COLUMNS

    with_ring = rf[rf["dataset"] == "with_ring_types"].set_index("target").reindex(targets)
    hidden = rf[rf["dataset"] == "hidden_ring_types"].set_index("target").reindex(targets)
    base = baseline[baseline["dataset"] == "with_ring_types"].set_index("target").reindex(targets)

    x = np.arange(len(targets))
    width = 0.25

    plt.figure(figsize=(11, 5.5))
    plt.bar(x - width, with_ring[metric], width, label="RF with ring types")
    plt.bar(x, hidden[metric], width, label="RF hidden ring types")
    plt.bar(x + width, base[metric], width, label="train-mean baseline")
    plt.xticks(x, targets, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(ylabel + " by Target, 5-Fold Mean")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / output_name, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold RF ring-type ablation on the 10k PAS sample.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    normal = pd.read_csv(NORMAL_SAMPLE_PATH)
    hidden = pd.read_csv(HIDDEN_SAMPLE_PATH)
    features = feature_columns(normal)
    hidden_features = feature_columns(hidden)

    if features != hidden_features:
        raise ValueError("Normal and hidden-ring feature vectors do not have identical feature columns.")
    if len(normal) != len(hidden):
        raise ValueError("Normal and hidden-ring datasets have different row counts.")
    if not normal["name"].equals(hidden["name"]):
        raise ValueError("Normal and hidden-ring datasets are not row-aligned.")

    folds = make_folds(len(normal), args.folds, args.random_state)
    all_metrics = []
    all_importances = []

    for dataset_name, frame in [
        ("with_ring_types", normal),
        ("hidden_ring_types", hidden),
    ]:
        metrics, importances = run_dataset(dataset_name, frame, features, folds, args.n_estimators)
        all_metrics.extend(metrics)
        all_importances.extend(importances)

    metrics_frame = pd.DataFrame(all_metrics)
    importances_frame = pd.DataFrame(all_importances)
    mean_metrics = aggregate_metrics(metrics_frame)
    mean_importances = aggregate_importances(importances_frame)
    top20 = mean_importances.groupby(["dataset", "target"]).head(20)

    metrics_frame.to_csv(DATA_DIR / "rf_ringtype_ablation_5fold_metrics.csv", index=False)
    mean_metrics.to_csv(DATA_DIR / "rf_ringtype_ablation_5fold_mean_metrics.csv", index=False)
    mean_importances.to_csv(DATA_DIR / "rf_ringtype_ablation_5fold_mean_importances.csv", index=False)
    top20.to_csv(DATA_DIR / "rf_ringtype_ablation_5fold_top20.csv", index=False)

    plot_metric(
        mean_metrics,
        "mae_percent",
        "MAE / mean(|actual|) (%)",
        "rf_ringtype_ablation_5fold_error_percent.png",
    )
    plot_metric(
        mean_metrics,
        "mae_over_test_std",
        "MAE / test target std",
        "rf_ringtype_ablation_5fold_normalized_mae.png",
    )
    plot_metric(
        mean_metrics,
        "r2",
        "R2",
        "rf_ringtype_ablation_5fold_r2.png",
    )

    print(f"Rows: {len(normal)}")
    print(f"Folds: {args.folds}")
    print(f"Feature count: {len(features)}")
    print(f"RF trees: {args.n_estimators}")
    print("Wrote data/rf_ringtype_ablation_5fold_metrics.csv")
    print("Wrote data/rf_ringtype_ablation_5fold_mean_metrics.csv")
    print("Wrote data/rf_ringtype_ablation_5fold_mean_importances.csv")
    print("Wrote data/rf_ringtype_ablation_5fold_top20.csv")
    print("Wrote results/sample/rf_ringtype_ablation_5fold_*.png")


if __name__ == "__main__":
    main()
