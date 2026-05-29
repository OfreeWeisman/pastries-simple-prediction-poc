from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path("data")
RESULTS_DIR = Path("results") / "sample"
MAIN_TARGETS = ["gap", "homo", "lumo", "dipole_norm", "aip", "aea", "nfod"]
TREE_MODELS = ["random_forest", "gradient_boosting", "xgboost"]


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def add_error_percent(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    denominators = (
        predictions.groupby(["target", "model"])["actual"]
        .apply(lambda values: values.abs().mean())
        .rename("mean_abs_actual")
        .reset_index()
    )
    merged = metrics.merge(denominators, on=["target", "model"], how="left")
    merged["mae_percent"] = 100 * merged["mae"] / merged["mean_abs_actual"]
    return merged


def plot_error_percent(metrics: pd.DataFrame) -> None:
    frame = metrics[metrics["target"].isin(MAIN_TARGETS)].copy()
    pivot = frame.pivot(index="target", columns="model", values="mae_percent").reindex(MAIN_TARGETS)

    plt.figure(figsize=(11, 5.5))
    x = np.arange(len(pivot.index))
    width = min(0.8 / max(1, len(pivot.columns)), 0.16)
    offset = (len(pivot.columns) - 1) / 2
    for index, model in enumerate(pivot.columns):
        values = pivot[model].to_numpy()
        plt.bar(x + (index - offset) * width, values, width, label=model)

    plt.xticks(x, pivot.index, rotation=30, ha="right")
    plt.ylabel("MAE / mean(|actual|) (%)")
    plt.title("Prediction Error Percentage by Target and Model")
    plt.legend(ncols=3, fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    savefig("prediction_error_percent.png")


def plot_r2(metrics: pd.DataFrame) -> None:
    frame = metrics[metrics["target"].isin(MAIN_TARGETS)].copy()
    pivot = frame.pivot(index="target", columns="model", values="r2").reindex(MAIN_TARGETS)

    plt.figure(figsize=(11, 5.5))
    x = np.arange(len(pivot.index))
    width = min(0.8 / max(1, len(pivot.columns)), 0.16)
    offset = (len(pivot.columns) - 1) / 2
    for index, model in enumerate(pivot.columns):
        values = pivot[model].to_numpy()
        plt.bar(x + (index - offset) * width, values, width, label=model)

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, pivot.index, rotation=30, ha="right")
    plt.ylabel("R2 on test set")
    plt.title("Prediction R2 by Target and Model")
    plt.legend(ncols=3, fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    savefig("prediction_r2.png")


def plot_predicted_vs_actual(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    best_models = (
        metrics[metrics["target"].isin(MAIN_TARGETS)]
        .sort_values(["target", "rmse"])
        .groupby("target")
        .first()
        .reset_index()[["target", "model", "r2"]]
    )

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.ravel()

    for axis, (_, row) in zip(axes, best_models.iterrows()):
        target = row["target"]
        model = row["model"]
        subset = predictions[(predictions["target"] == target) & (predictions["model"] == model)]
        axis.scatter(subset["actual"], subset["predicted"], s=14, alpha=0.7)
        min_value = min(subset["actual"].min(), subset["predicted"].min())
        max_value = max(subset["actual"].max(), subset["predicted"].max())
        axis.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1)
        axis.set_title(f"{target}: {model} (R2={row['r2']:.2f})")
        axis.set_xlabel("Actual")
        axis.set_ylabel("Predicted")
        axis.grid(alpha=0.2)

    for axis in axes[len(best_models) :]:
        axis.axis("off")

    fig.suptitle("Predicted vs Actual for Best Model per Target", y=1.01, fontsize=14)
    savefig("predicted_vs_actual_best_models.png")


def plot_top_importances(importances: pd.DataFrame) -> None:
    for target in MAIN_TARGETS:
        target_frame = importances[
            (importances["target"] == target) & (importances["model"].isin(TREE_MODELS))
        ]
        if target_frame.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for axis, model in zip(axes, TREE_MODELS):
            model_frame = (
                target_frame[target_frame["model"] == model]
                .sort_values("importance", ascending=False)
                .head(10)
                .sort_values("importance", ascending=True)
            )
            axis.barh(model_frame["feature"], model_frame["importance"])
            axis.set_title(model)
            axis.set_xlabel("importance")
            axis.tick_params(axis="y", labelsize=8)
            axis.grid(axis="x", alpha=0.2)

        fig.suptitle(f"Top 10 Feature Importances for {target}", y=1.02, fontsize=14)
        savefig(f"top_features_{target}.png")


def plot_clusters(clusters: pd.DataFrame) -> None:
    preferred_methods = ["kmeans_5", "birch_5", "agglomerative_5", "spectral_5"]
    methods = [method for method in preferred_methods if method in clusters.columns]
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.8))
    if len(methods) == 1:
        axes = [axes]

    for axis, method in zip(axes, methods):
        scatter = axis.scatter(
            clusters["pca_1"],
            clusters["pca_2"],
            c=clusters[method],
            s=15,
            alpha=0.8,
            cmap="tab10",
        )
        axis.set_title(method)
        axis.set_xlabel("PCA 1")
        axis.set_ylabel("PCA 2")
        axis.grid(alpha=0.2)
        legend = axis.legend(*scatter.legend_elements(), title="cluster", fontsize=8)
        axis.add_artist(legend)

    fig.suptitle("Sample Clusters on PCA Projection", y=1.02, fontsize=14)
    savefig("cluster_pca_maps.png")


def plot_cluster_target_means(clusters: pd.DataFrame, feature_vectors: pd.DataFrame) -> None:
    cluster_column = "kmeans_5" if "kmeans_5" in clusters.columns else clusters.columns[3]
    merged = clusters[["name", cluster_column]].merge(feature_vectors, on="name", how="left")
    means = merged.groupby(cluster_column)[["gap", "homo", "lumo", "dipole_norm", "nfod"]].mean()
    normalized = (means - means.mean()) / means.std(ddof=0)

    plt.figure(figsize=(9, 4.8))
    image = plt.imshow(normalized.T, aspect="auto", cmap="coolwarm")
    plt.colorbar(image, label="cluster mean z-score")
    plt.xticks(np.arange(len(normalized.index)), normalized.index)
    plt.yticks(np.arange(len(normalized.columns)), normalized.columns)
    plt.xlabel(f"{cluster_column} cluster")
    plt.title("Target Property Profiles by Cluster")
    savefig("cluster_target_profiles.png")


def main() -> None:
    ensure_results_dir()

    metrics = pd.read_csv(DATA_DIR / "sample_prediction_metrics.csv")
    predictions = pd.read_csv(DATA_DIR / "sample_predictions.csv")
    importances = pd.read_csv(DATA_DIR / "sample_feature_importances.csv")
    clusters = pd.read_csv(DATA_DIR / "sample_clusters.csv")
    feature_vectors = pd.read_csv(DATA_DIR / "compas-2x_pastries_features_sample_feature_vectors.csv")

    metrics = add_error_percent(metrics, predictions)
    metrics.to_csv(DATA_DIR / "sample_prediction_metrics_with_error_percent.csv", index=False)

    plot_error_percent(metrics)
    plot_r2(metrics)
    plot_predicted_vs_actual(predictions, metrics)
    plot_top_importances(importances)
    plot_clusters(clusters)
    plot_cluster_target_means(clusters, feature_vectors)

    print(f"Wrote visualizations to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
