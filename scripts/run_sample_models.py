from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans, SpectralClustering
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None


DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "compas-2x_pastries_features_sample_train_feature_vectors.csv"
TEST_PATH = DATA_DIR / "compas-2x_pastries_features_sample_test_feature_vectors.csv"
SAMPLE_PATH = DATA_DIR / "compas-2x_pastries_features_sample_feature_vectors.csv"

TARGET_COLUMNS = [
    "gap",
    "homo",
    "lumo",
    "dipole_norm",
    "aip",
    "aea",
    "gap_corr",
    "homo_corr",
    "lumo_corr",
    "nfod",
]

ALL_TARGET_COLUMNS = [
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
]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"name", *ALL_TARGET_COLUMNS}
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def prediction_models() -> dict[str, object]:
    models: dict[str, object] = {
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=2,
        ),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "pls": make_pipeline(StandardScaler(), PLSRegression(n_components=8)),
    }

    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )

    return models


def mean_baseline_predictions(y_train: pd.Series, rows: int) -> np.ndarray:
    return np.full(rows, float(y_train.mean()))


def run_prediction_models(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> None:
    x_train = train[features]
    x_test = test[features]
    metrics = []
    predictions = []
    importances = []

    for target in TARGET_COLUMNS:
        y_train = train[target]
        y_test = test[target]

        baseline_pred = mean_baseline_predictions(y_train, len(test))
        metrics.append(
            {
                "target": target,
                "model": "train_mean_baseline",
                "mae": mean_absolute_error(y_test, baseline_pred),
                "rmse": rmse(y_test, baseline_pred),
                "r2": r2_score(y_test, baseline_pred),
                "train_rows": len(train),
                "test_rows": len(test),
                "feature_count": len(features),
            }
        )
        for name, actual, predicted in zip(test["name"], y_test, baseline_pred):
            predictions.append(
                {
                    "name": name,
                    "target": target,
                    "model": "train_mean_baseline",
                    "actual": actual,
                    "predicted": predicted,
                    "error": predicted - actual,
                }
            )

        for model_name, model in prediction_models().items():
            model.fit(x_train, y_train)
            y_pred = np.asarray(model.predict(x_test)).reshape(-1)

            metrics.append(
                {
                    "target": target,
                    "model": model_name,
                    "mae": mean_absolute_error(y_test, y_pred),
                    "rmse": rmse(y_test, y_pred),
                    "r2": r2_score(y_test, y_pred),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "feature_count": len(features),
                }
            )

            for name, actual, predicted in zip(test["name"], y_test, y_pred):
                predictions.append(
                    {
                        "name": name,
                        "target": target,
                        "model": model_name,
                        "actual": actual,
                        "predicted": predicted,
                        "error": predicted - actual,
                    }
                )

            estimator = model
            if hasattr(model, "named_steps"):
                estimator = model.named_steps[list(model.named_steps)[-1]]

            if hasattr(estimator, "feature_importances_"):
                for feature, importance in zip(features, estimator.feature_importances_):
                    importances.append(
                        {
                            "target": target,
                            "model": model_name,
                            "feature": feature,
                            "importance": importance,
                        }
                    )

    pd.DataFrame(metrics).sort_values(["target", "rmse"]).to_csv(
        DATA_DIR / "sample_prediction_metrics.csv",
        index=False,
    )
    pd.DataFrame(predictions).to_csv(DATA_DIR / "sample_predictions.csv", index=False)
    pd.DataFrame(importances).sort_values(
        ["target", "model", "importance"],
        ascending=[True, True, False],
    ).to_csv(DATA_DIR / "sample_feature_importances.csv", index=False)

    top_importances = (
        pd.DataFrame(importances)
        .sort_values(["target", "model", "importance"], ascending=[True, True, False])
        .groupby(["target", "model"])
        .head(10)
    )
    top_importances.to_csv(DATA_DIR / "sample_top10_feature_importances.csv", index=False)


def run_clustering(sample: pd.DataFrame, features: list[str]) -> None:
    x_scaled = StandardScaler().fit_transform(sample[features])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x_scaled)

    clusters = pd.DataFrame(
        {
            "name": sample["name"],
            "pca_1": coords[:, 0],
            "pca_2": coords[:, 1],
        }
    )

    for n_clusters in [3, 5, 8]:
        clusters[f"kmeans_{n_clusters}"] = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,
        ).fit_predict(x_scaled)

        clusters[f"birch_{n_clusters}"] = Birch(n_clusters=n_clusters).fit_predict(x_scaled)

        if len(sample) <= 3000:
            clusters[f"agglomerative_{n_clusters}"] = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage="ward",
            ).fit_predict(x_scaled)

            clusters[f"spectral_{n_clusters}"] = SpectralClustering(
                n_clusters=n_clusters,
                random_state=42,
                assign_labels="kmeans",
                affinity="nearest_neighbors",
                n_neighbors=20,
            ).fit_predict(x_scaled)

    clusters.to_csv(DATA_DIR / "sample_clusters.csv", index=False)

    summary_rows = []
    for column in clusters.columns:
        if column in {"name", "pca_1", "pca_2"}:
            continue
        counts = clusters[column].value_counts().sort_index()
        for cluster_id, count in counts.items():
            summary_rows.append(
                {
                    "method": column,
                    "cluster": cluster_id,
                    "count": count,
                }
            )

    pd.DataFrame(summary_rows).to_csv(DATA_DIR / "sample_cluster_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "pca_component": 1,
                "explained_variance_ratio": pca.explained_variance_ratio_[0],
            },
            {
                "pca_component": 2,
                "explained_variance_ratio": pca.explained_variance_ratio_[1],
            },
        ]
    ).to_csv(DATA_DIR / "sample_pca_summary.csv", index=False)


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    features = feature_columns(train)

    run_prediction_models(train, test, features)
    run_clustering(sample, features)

    print(f"Used {len(features)} feature columns")
    print("Wrote data/sample_prediction_metrics.csv")
    print("Wrote data/sample_predictions.csv")
    print("Wrote data/sample_feature_importances.csv")
    print("Wrote data/sample_clusters.csv")
    print("Wrote data/sample_cluster_summary.csv")
    print("Wrote data/sample_pca_summary.csv")


if __name__ == "__main__":
    main()
