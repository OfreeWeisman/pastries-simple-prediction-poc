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
RESULTS_DIR = Path("results") / "rf_rng_ringtype_ablation_5fold"
NORMAL_SAMPLE_PATH = DATA_DIR / "compas-2x_pastries_features_sample_feature_vectors.csv"
RNG_SAMPLE_PATH = DATA_DIR / "compas-2x_pastries_features_sample_hidden_rng_feature_vectors.csv"

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
RING_TYPES = [
    "benzene",
    "pyridine",
    "pyrazine",
    "thiophene",
    "furan",
    "borole",
    "cyclobutadiene",
    "pyrrole",
    "dhdiborinine",
    "14diborinine",
    "borinine",
]
RING_IDENTITY_FEATURES = [
    *RING_TYPES,
    "b",
    "s",
    "o",
    "n",
    "heteroatoms",
    "heterocycles",
    "aromatic_rings",
    "heterocycle_fraction",
    "heteroatom_density",
    "aromatic_ring_fraction",
    "pas_donor_ring_count",
    "pas_acceptor_ring_count",
    "pas_donor_acceptor_balance",
    "pas_donor_acceptor_ratio",
    "pas_donor_fraction",
    "pas_acceptor_fraction",
    "pas_terminal_non_benzene_count",
    "pas_middle_non_benzene_count",
    "pas_branch_non_benzene_count",
    "pas_terminal_heterocycle_count",
    "pas_middle_heterocycle_count",
    "pas_branch_heterocycle_count",
    "pas_terminal_donor_count",
    "pas_middle_donor_count",
    "pas_branch_donor_count",
    "pas_terminal_acceptor_count",
    "pas_middle_acceptor_count",
    "pas_branch_acceptor_count",
]
T_95_DF4 = 2.7764451051977987


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


def rf_model(n_estimators: int, n_jobs: int, random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        min_samples_leaf=2,
        n_jobs=n_jobs,
    )


def make_hidden_rng_features(normal: pd.DataFrame) -> pd.DataFrame:
    hidden = normal.copy()
    total_rings = hidden["pas_total_rings"] if "pas_total_rings" in hidden else hidden["rings"]
    branch_count = hidden["pas_branch_count"] if "pas_branch_count" in hidden else hidden.get("branch", 0)

    for column in RING_IDENTITY_FEATURES:
        if column in hidden:
            hidden[column] = 0.0

    for ring_type in RING_TYPES:
        for suffix in ["count", "fraction", "terminal_count", "middle_count", "branch_count"]:
            column = f"pas_{ring_type}_{suffix}"
            if column in hidden:
                hidden[column] = 0.0

    hidden["rng"] = total_rings.astype(float)
    hidden["pas_rng_count"] = total_rings.astype(float)
    hidden["pas_rng_fraction"] = np.where(total_rings > 0, 1.0, 0.0)
    hidden["pas_rng_terminal_count"] = np.minimum(total_rings, 2).astype(float)
    hidden["pas_rng_middle_count"] = np.maximum(total_rings - 2, 0).astype(float)
    hidden["pas_rng_branch_count"] = branch_count.astype(float)
    return hidden


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
    mean_abs_actual = y_test.abs().mean()
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
        "mae_percent": 100 * mae / mean_abs_actual if mean_abs_actual else np.nan,
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
    n_jobs: int,
    random_state: int,
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

            test_mean_baseline = np.full(len(test), float(y_test.mean()))
            metrics.append(
                metric_row(
                    dataset_name,
                    fold_index,
                    target,
                    "test_mean_baseline",
                    0,
                    y_train,
                    y_test,
                    test_mean_baseline,
                )
            )

            baseline = np.full(len(test), float(y_train.mean()))
            metrics.append(
                metric_row(
                    dataset_name,
                    fold_index,
                    target,
                    "train_mean_baseline",
                    0,
                    y_train,
                    y_test,
                    baseline,
                )
            )

            model = rf_model(n_estimators, n_jobs, random_state + fold_index)
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


def aggregate_with_ci(frame: pd.DataFrame, group_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, as_index=False)
    means = grouped[metric_columns].mean()
    stds = grouped[metric_columns].std(ddof=1).rename(
        columns={column: f"{column}_std" for column in metric_columns}
    )
    counts = grouped.size().rename(columns={"size": "n"})
    result = means.merge(stds, on=group_columns, how="left").merge(counts, on=group_columns, how="left")
    for column in metric_columns:
        result[f"{column}_ci95"] = T_95_DF4 * result[f"{column}_std"] / np.sqrt(result["n"])
    return result


def aggregate_importances(importances: pd.DataFrame) -> pd.DataFrame:
    return aggregate_with_ci(
        importances,
        ["dataset", "target", "feature"],
        ["importance"],
    ).sort_values(["dataset", "target", "importance"], ascending=[True, True, False])


def plot_metric(metric_summary: pd.DataFrame, metric: str, ylabel: str, output_name: str) -> None:
    rf = metric_summary[metric_summary["model"] == "random_forest"]
    baseline = metric_summary[metric_summary["model"] == "train_mean_baseline"]
    targets = TARGET_COLUMNS
    with_ring = rf[rf["dataset"] == "with_ring_types"].set_index("target").reindex(targets)
    hidden = rf[rf["dataset"] == "hidden_rng"].set_index("target").reindex(targets)
    train_base = baseline[baseline["dataset"] == "with_ring_types"].set_index("target").reindex(targets)
    test_baseline = metric_summary[metric_summary["model"] == "test_mean_baseline"]
    test_base = test_baseline[test_baseline["dataset"] == "with_ring_types"].set_index("target").reindex(targets)

    x = np.arange(len(targets))
    width = 0.2
    plt.figure(figsize=(11.5, 5.8))
    plt.bar(
        x - 1.5 * width,
        with_ring[metric],
        width,
        yerr=with_ring[f"{metric}_ci95"],
        capsize=3,
        label="RF with ring types",
    )
    plt.bar(
        x - 0.5 * width,
        hidden[metric],
        width,
        yerr=hidden[f"{metric}_ci95"],
        capsize=3,
        label="RF hidden as Rng",
    )
    plt.bar(
        x + 0.5 * width,
        train_base[metric],
        width,
        yerr=train_base[f"{metric}_ci95"],
        capsize=3,
        label="train-mean baseline",
    )
    plt.bar(
        x + 1.5 * width,
        test_base[metric],
        width,
        yerr=test_base[f"{metric}_ci95"],
        capsize=3,
        label="test-mean baseline",
    )
    plt.xticks(x, targets, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by target, 5-fold mean with 95% CI")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / output_name, dpi=220, bbox_inches="tight")
    plt.close()


def plot_importance_table(importances: pd.DataFrame, dataset: str, target: str, top_n: int) -> None:
    subset = importances[(importances["dataset"] == dataset) & (importances["target"] == target)].head(top_n)
    display = subset[["feature", "importance", "importance_ci95"]].copy()
    display["importance"] = display["importance"].map(lambda value: f"{value:.4f}")
    display["importance_ci95"] = display["importance_ci95"].map(lambda value: f"{value:.4f}")
    display = display.rename(
        columns={
            "feature": "Feature",
            "importance": "Mean importance",
            "importance_ci95": "95% CI",
        }
    )

    row_height = 0.34
    fig_height = max(3.0, row_height * (len(display) + 2))
    fig, axis = plt.subplots(figsize=(9.5, fig_height))
    axis.axis("off")
    table = axis.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.58, 0.2, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.25)
    for (row, _column), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8e8e8")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f7f7")
    axis.set_title(f"Top {top_n} RF Feature Importances: {dataset}, {target}", pad=12)
    plt.savefig(
        RESULTS_DIR / f"feature_importance_table_{dataset}_{target}.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()


def write_feature_list(features_by_dataset: dict[str, list[str]]) -> None:
    rows = []
    for dataset, features in features_by_dataset.items():
        for index, feature in enumerate(features, start=1):
            rows.append({"dataset": dataset, "feature_index": index, "feature": feature})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "features_used_by_dataset.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold RF ablation with generic Rng hidden ring types.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    normal = pd.read_csv(NORMAL_SAMPLE_PATH)
    hidden_rng = make_hidden_rng_features(normal)
    hidden_rng.to_csv(RNG_SAMPLE_PATH, index=False)

    normal_features = feature_columns(normal)
    hidden_features = feature_columns(hidden_rng)
    folds = make_folds(len(normal), args.folds, args.random_state)
    write_feature_list({"with_ring_types": normal_features, "hidden_rng": hidden_features})

    all_metrics = []
    all_importances = []
    for dataset_name, frame, features in [
        ("with_ring_types", normal, normal_features),
        ("hidden_rng", hidden_rng, hidden_features),
    ]:
        metrics, importances = run_dataset(
            dataset_name,
            frame,
            features,
            folds,
            args.n_estimators,
            args.n_jobs,
            args.random_state,
        )
        all_metrics.extend(metrics)
        all_importances.extend(importances)

    metrics_frame = pd.DataFrame(all_metrics)
    importances_frame = pd.DataFrame(all_importances)
    metric_columns = ["mae", "rmse", "r2", "mae_percent", "mae_over_test_std", "mae_over_train_std"]
    metric_summary = aggregate_with_ci(
        metrics_frame,
        ["dataset", "target", "model"],
        metric_columns,
    )
    importance_summary = aggregate_importances(importances_frame)
    top_importances = importance_summary.groupby(["dataset", "target"]).head(args.top_n)

    metrics_frame.to_csv(RESULTS_DIR / "rf_rng_ablation_5fold_metrics_by_fold.csv", index=False)
    metric_summary.to_csv(RESULTS_DIR / "rf_rng_ablation_5fold_metric_summary_ci95.csv", index=False)
    importances_frame.to_csv(RESULTS_DIR / "rf_rng_ablation_5fold_importances_by_fold.csv", index=False)
    importance_summary.to_csv(RESULTS_DIR / "rf_rng_ablation_5fold_importance_summary_ci95.csv", index=False)
    top_importances.to_csv(RESULTS_DIR / f"rf_rng_ablation_5fold_top{args.top_n}_importances.csv", index=False)

    plot_metric(metric_summary, "mae", "MAE (eV)", "rf_rng_ablation_5fold_mae_ci95.png")
    plot_metric(
        metric_summary,
        "mae_percent",
        "MAE / mean(|actual|) (%)",
        "rf_rng_ablation_5fold_error_percent_ci95.png",
    )
    plot_metric(
        metric_summary,
        "mae_over_test_std",
        "MAE / test target std",
        "rf_rng_ablation_5fold_normalized_mae_ci95.png",
    )
    plot_metric(metric_summary, "r2", "R2", "rf_rng_ablation_5fold_r2_ci95.png")

    for dataset in ["with_ring_types", "hidden_rng"]:
        for target in TARGET_COLUMNS:
            plot_importance_table(importance_summary, dataset, target, args.top_n)

    print(f"Rows: {len(normal)}")
    print(f"Folds: {args.folds}")
    print(f"RF trees: {args.n_estimators}")
    print(f"with_ring_types feature count: {len(normal_features)}")
    print(f"hidden_rng feature count: {len(hidden_features)}")
    print(f"Wrote hidden Rng feature vectors to {RNG_SAMPLE_PATH}")
    print(f"Wrote outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
