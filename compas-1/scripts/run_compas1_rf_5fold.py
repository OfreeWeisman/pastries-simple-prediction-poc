from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_config import PAS_COLUMNS_SUPERSEDED_BY_PAPER


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SOURCE_PATH = DATA_DIR / "compas-1D_b3lyp_lalas.csv"
FEATURE_PATH = DATA_DIR / "compas-1D_b3lyp_lalas_feature_vectors.csv"

TARGET_MAP = {
    "homo": "HOMO_eV",
    "lumo": "LUMO_eV",
    "gap": "GAP_eV",
    "aip": "aIP_eV",
    "aea": "aEA_eV",
    "energy": "Erel_eV",
}
TARGET_COLUMNS = list(TARGET_MAP)
ALL_TARGET_COLUMNS = set(TARGET_COLUMNS)
T_CRITICAL_95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}
PAPER_REPORTED_MAE_RANGES = {
    "homo": (0.02, 0.13),
    "lumo": (0.02, 0.13),
    "gap": (0.04, 0.26),
    "aip": (0.02, 0.13),
    "aea": (0.02, 0.13),
    "energy": (0.04, 0.25),
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
DONOR_RINGS = {"thiophene", "furan", "pyrrole"}
ACCEPTOR_RINGS = {"pyridine", "pyrazine", "borole"}
HETEROCYCLE_RINGS = {
    "pyridine",
    "pyrazine",
    "thiophene",
    "furan",
    "borole",
    "pyrrole",
    "dhdiborinine",
    "14diborinine",
    "borinine",
}
SOURCE_NUMERIC_FEATURES = [
    "rings",
    "aromatic_rings",
    "atoms",
    "heteroatoms",
    "heterocycles",
    "branch",
    "cyclobutadiene",
    "pyrrole",
    "borole",
    "furan",
    "thiophene",
    "dhdiborinine",
    "14diborinine",
    "pyrazine",
    "pyridine",
    "borinine",
    "benzene",
    "h",
    "c",
    "b",
    "s",
    "o",
    "n",
]


def parse_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(parsed) else parsed


def split_representation(representation: str) -> tuple[str, str]:
    if " " not in representation:
        return representation, ""
    return representation.rsplit(" ", 1)


def clean_ring_token(token: str) -> tuple[str, bool]:
    stripped = token.strip()
    is_branch = stripped.startswith("(") or stripped.endswith(")")
    stripped = stripped.strip("()")
    return "benzene", is_branch


def parse_rings(ring_part: str) -> list[tuple[str, bool]]:
    rings = []
    for token in ring_part.split(","):
        ring_type, is_branch = clean_ring_token(token)
        if ring_type:
            rings.append((ring_type, is_branch))
    return rings


def longest_run(sequence: str, value: str) -> int:
    longest = 0
    current = 0
    for char in sequence:
        if char == value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def longest_run_degeneracy(sequence: str, value: str) -> int:
    target_length = longest_run(sequence, value)
    if target_length == 0:
        return 0
    degeneracy = 0
    current = 0
    for char in sequence:
        if char == value:
            current += 1
        else:
            if current == target_length:
                degeneracy += 1
            current = 0
    if current == target_length:
        degeneracy += 1
    return degeneracy


def subsequence_count(sequence: str, subsequence: str) -> int:
    return sum(
        1
        for index in range(0, len(sequence) - len(subsequence) + 1)
        if sequence[index : index + len(subsequence)] == subsequence
    )


def max_parenthesis_depth(value: str) -> int:
    depth = 0
    max_depth = 0
    for char in value:
        if char == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ")":
            depth = max(0, depth - 1)
    return max_depth


def formula_counts(molecule: str) -> tuple[int, int]:
    match = re.search(r"c(\d+)h(\d+)", molecule.lower())
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def extract_features(row: dict[str, str]) -> dict[str, float | str]:
    ring_part, topology_part = split_representation(row["representation"])
    raw_annulation_sequence = "".join(char for char in topology_part if char in {"L", "l", "A", "a"})
    raw_annulation_upper = raw_annulation_sequence.upper()
    topology = topology_part.upper()
    annulation_sequence = "".join(char for char in topology if char in {"L", "A"})
    total_annulations = len(annulation_sequence)
    total_l = annulation_sequence.count("L")
    total_a = annulation_sequence.count("A")
    la_count = annulation_sequence.count("LA")
    al_count = annulation_sequence.count("AL")

    rings = parse_rings(ring_part)
    total_rings = int(parse_float(row.get("n_rings", len(rings)))) or len(rings)
    ring_counts = Counter({"benzene": total_rings})
    branch_ring_counts = Counter(ring_type for ring_type, is_branch in rings if is_branch)
    terminal_counts = Counter()
    middle_counts = Counter()
    branch_counts = Counter()
    for index, (ring_type, is_branch) in enumerate(rings):
        if index == 0 or index == len(rings) - 1:
            terminal_counts[ring_type] += 1
        else:
            middle_counts[ring_type] += 1
        if is_branch:
            branch_counts[ring_type] += 1

    c_count, h_count = formula_counts(row["molecule"])
    atom_count = c_count + h_count
    branch_count = topology.count("(")

    def class_position_count(position_counts: Counter[str], ring_class: set[str]) -> int:
        return sum(position_counts[ring_type] for ring_type in ring_class)

    def non_benzene_count(position_counts: Counter[str]) -> int:
        return sum(count for ring_type, count in position_counts.items() if ring_type != "benzene")

    features: dict[str, float | str] = {
        "name": row["molecule"],
        "pas_total_annulations": total_annulations,
        "pas_total_l": total_l,
        "pas_total_a": total_a,
        "pas_a_fraction": total_a / total_annulations if total_annulations else 0.0,
        "pas_l_to_a_ratio": total_l / total_a if total_a else float(total_l),
        "pas_la_count": la_count,
        "pas_al_count": al_count,
        "pas_alternation_count": la_count + al_count,
        "pas_alternation_fraction": (la_count + al_count) / max(1, total_annulations - 1),
        "pas_branch_density": branch_count / total_rings if total_rings else 0.0,
        "pas_max_branch_depth": max_parenthesis_depth(topology),
        "pas_parenthesis_count": topology.count("(") + topology.count(")"),
        "pas_topology_length": len(topology),
        "paper_lalas_number_of_rings": total_rings,
        "paper_lalas_number_of_branching_points": branch_count,
        "paper_lalas_lal_subsequence_count": subsequence_count(raw_annulation_upper, "LAL"),
        "paper_lalas_l_ratio": raw_annulation_upper.count("L") / len(raw_annulation_sequence)
        if raw_annulation_sequence
        else 0.0,
        "paper_lalas_longest_l": longest_run(raw_annulation_upper, "L"),
        "paper_lalas_longest_l_degeneracy": longest_run_degeneracy(raw_annulation_upper, "L"),
        "paper_lalas_upper_a_ratio": raw_annulation_sequence.count("A") / len(raw_annulation_sequence)
        if raw_annulation_sequence
        else 0.0,
        "paper_lalas_lower_to_upper_a_ratio": raw_annulation_sequence.count("a")
        / raw_annulation_sequence.count("A")
        if raw_annulation_sequence.count("A")
        else float(raw_annulation_sequence.count("a")),
        "paper_lalas_longest_upper_a": longest_run(raw_annulation_sequence, "A"),
        "paper_lalas_longest_a_case_insensitive": longest_run(raw_annulation_upper, "A"),
        "pas_donor_ring_count": 0,
        "pas_acceptor_ring_count": 0,
        "pas_donor_acceptor_balance": 0,
        "pas_donor_acceptor_ratio": 0.0,
        "pas_donor_fraction": 0.0,
        "pas_acceptor_fraction": 0.0,
        "pas_terminal_non_benzene_count": non_benzene_count(terminal_counts),
        "pas_middle_non_benzene_count": non_benzene_count(middle_counts),
        "pas_branch_non_benzene_count": non_benzene_count(branch_counts),
        "pas_terminal_heterocycle_count": class_position_count(terminal_counts, HETEROCYCLE_RINGS),
        "pas_middle_heterocycle_count": class_position_count(middle_counts, HETEROCYCLE_RINGS),
        "pas_branch_heterocycle_count": class_position_count(branch_counts, HETEROCYCLE_RINGS),
        "pas_terminal_donor_count": class_position_count(terminal_counts, DONOR_RINGS),
        "pas_middle_donor_count": class_position_count(middle_counts, DONOR_RINGS),
        "pas_branch_donor_count": class_position_count(branch_counts, DONOR_RINGS),
        "pas_terminal_acceptor_count": class_position_count(terminal_counts, ACCEPTOR_RINGS),
        "pas_middle_acceptor_count": class_position_count(middle_counts, ACCEPTOR_RINGS),
        "pas_branch_acceptor_count": class_position_count(branch_counts, ACCEPTOR_RINGS),
        "heterocycle_fraction": 0.0,
        "heteroatom_density": 0.0,
        "aromatic_ring_fraction": 1.0 if total_rings else 0.0,
    }

    source_values = {
        "rings": total_rings,
        "aromatic_rings": total_rings,
        "atoms": atom_count,
        "heteroatoms": 0,
        "heterocycles": 0,
        "branch": branch_count,
        "benzene": total_rings,
        "h": h_count,
        "c": c_count,
        "b": 0,
        "s": 0,
        "o": 0,
        "n": 0,
    }
    for ring_type in RING_TYPES:
        source_values.setdefault(ring_type, 0)
    for column in SOURCE_NUMERIC_FEATURES:
        features[column] = float(source_values[column])

    for ring_type in RING_TYPES:
        count = ring_counts[ring_type]
        features[f"pas_{ring_type}_count"] = count
        features[f"pas_{ring_type}_fraction"] = count / total_rings if total_rings else 0.0
        features[f"pas_{ring_type}_terminal_count"] = terminal_counts[ring_type]
        features[f"pas_{ring_type}_middle_count"] = middle_counts[ring_type]
        features[f"pas_{ring_type}_branch_count"] = branch_ring_counts[ring_type]

    for target, source_column in TARGET_MAP.items():
        features[target] = parse_float(row[source_column])

    return features


def write_feature_vectors() -> pd.DataFrame:
    with SOURCE_PATH.open("r", newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            if not row.get("representation"):
                continue
            if any(row.get(source_column) in {None, ""} for source_column in TARGET_MAP.values()):
                continue
            rows.append(extract_features(row))
    frame = pd.DataFrame(rows)
    frame.to_csv(FEATURE_PATH, index=False)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in ALL_TARGET_COLUMNS
        and column != "name"
        and column not in PAS_COLUMNS_SUPERSEDED_BY_PAPER
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def rf_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=1,
    )


def rmse(y_true: pd.Series, predicted: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, predicted)))


def make_folds(row_count: int, fold_count: int = 5, random_state: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(row_count)
    return [fold.astype(int) for fold in np.array_split(indices, fold_count)]


def run_5fold(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    importances = []
    all_indices = np.arange(len(frame))
    folds = make_folds(len(frame))

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
            for model_name, predicted in [("train_mean_baseline", baseline)]:
                mae = mean_absolute_error(y_test, predicted)
                metrics.append(
                    {
                        "fold": fold_index,
                        "target": target,
                        "model": model_name,
                        "feature_count": len(features),
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "mae": mae,
                        "rmse": rmse(y_test, predicted),
                        "r2": r2_score(y_test, predicted),
                        "mae_percent": 100 * mae / y_test.abs().mean(),
                        "mae_over_test_std": mae / y_test.std(ddof=0),
                        "mae_over_train_std": mae / y_train.std(ddof=0),
                    }
                )

            model = rf_model()
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            mae = mean_absolute_error(y_test, predicted)
            metrics.append(
                {
                    "fold": fold_index,
                    "target": target,
                    "model": "random_forest",
                    "feature_count": len(features),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "mae": mae,
                    "rmse": rmse(y_test, predicted),
                    "r2": r2_score(y_test, predicted),
                    "mae_percent": 100 * mae / y_test.abs().mean(),
                    "mae_over_test_std": mae / y_test.std(ddof=0),
                    "mae_over_train_std": mae / y_train.std(ddof=0),
                }
            )
            for feature, importance in zip(features, model.feature_importances_):
                importances.append(
                    {
                        "fold": fold_index,
                        "target": target,
                        "feature": feature,
                        "importance": importance,
                    }
                )

    return pd.DataFrame(metrics), pd.DataFrame(importances)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["mae", "rmse", "r2", "mae_percent", "mae_over_test_std", "mae_over_train_std"]
    grouped = metrics.groupby(["target", "model"], as_index=False)
    means = grouped[metric_columns].mean()
    counts = grouped.size().rename(columns={"size": "fold_count"})
    stds = grouped[metric_columns].std(ddof=1)
    stds = stds.rename(columns={column: f"{column}_std" for column in metric_columns})
    summary = means.merge(counts, on=["target", "model"], how="left").merge(
        stds, on=["target", "model"], how="left"
    )
    for column in metric_columns:
        summary[f"{column}_ci95"] = summary.apply(
            lambda row: T_CRITICAL_95.get(int(row["fold_count"]), 1.96)
            * row[f"{column}_std"]
            / math.sqrt(row["fold_count"])
            if row["fold_count"] > 1
            else np.nan,
            axis=1,
        )
    return summary


def write_paper_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rf = summary[summary["model"] == "random_forest"].set_index("target").reindex(TARGET_COLUMNS)
    rows = []
    for target in TARGET_COLUMNS:
        low, high = PAPER_REPORTED_MAE_RANGES[target]
        mae = rf.loc[target, "mae"]
        rows.append(
            {
                "target": target,
                "mae": mae,
                "mae_ci95": rf.loc[target, "mae_ci95"],
                "mae_ci95_low": mae - rf.loc[target, "mae_ci95"],
                "mae_ci95_high": mae + rf.loc[target, "mae_ci95"],
                "rmse": rf.loc[target, "rmse"],
                "r2": rf.loc[target, "r2"],
                "mae_over_test_std": rf.loc[target, "mae_over_test_std"],
                "normalized_mae_ci95": rf.loc[target, "mae_over_test_std_ci95"],
                "mae_percent": rf.loc[target, "mae_percent"],
                "error_percent_ci95": rf.loc[target, "mae_percent_ci95"],
                "paper_reported_mae_range": f"{low:.2f}-{high:.2f} eV",
                "within_paper_range": "yes" if low <= mae <= high else "no",
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(DATA_DIR / "compas1_vs_paper_summary.csv", index=False)
    return comparison


def plot_summary(summary: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    targets = TARGET_COLUMNS
    rf = summary[summary["model"] == "random_forest"].set_index("target").reindex(targets)
    baseline = summary[summary["model"] == "train_mean_baseline"].set_index("target").reindex(targets)
    x = np.arange(len(targets))
    width = 0.35

    for metric, ylabel, filename in [
        ("mae_over_test_std", "MAE / test target std", "compas1_rf_5fold_normalized_mae.png"),
        ("mae_percent", "MAE / mean(|actual|) (%)", "compas1_rf_5fold_error_percent.png"),
        ("r2", "R2", "compas1_rf_5fold_r2.png"),
    ]:
        plt.figure(figsize=(10, 5))
        plt.bar(
            x - width / 2,
            rf[metric],
            width,
            yerr=rf[f"{metric}_ci95"],
            capsize=4,
            label="Random Forest",
        )
        plt.bar(
            x + width / 2,
            baseline[metric],
            width,
            yerr=baseline[f"{metric}_ci95"],
            capsize=4,
            label="train-mean baseline",
        )
        plt.xticks(x, targets, rotation=30, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"COMPAS-1 {ylabel}, 5-Fold Mean with 95% CI")
        plt.legend()
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / filename, dpi=180, bbox_inches="tight")
        plt.close()


def main() -> None:
    frame = write_feature_vectors()
    features = feature_columns(frame)
    metrics, importances = run_5fold(frame, features)
    summary = summarize_metrics(metrics)
    mean_importances = (
        importances.groupby(["target", "feature"], as_index=False)["importance"]
        .mean()
        .sort_values(["target", "importance"], ascending=[True, False])
    )
    top20 = mean_importances.groupby("target").head(20)

    metrics.to_csv(DATA_DIR / "compas1_rf_5fold_metrics.csv", index=False)
    summary.to_csv(DATA_DIR / "compas1_rf_5fold_mean_metrics.csv", index=False)
    mean_importances.to_csv(DATA_DIR / "compas1_rf_5fold_mean_importances.csv", index=False)
    top20.to_csv(DATA_DIR / "compas1_rf_5fold_top20.csv", index=False)
    write_paper_comparison(summary)
    plot_summary(summary)

    print(f"Rows: {len(frame)}")
    print(f"Feature count: {len(features)}")
    print("Wrote COMPAS-1 RF 5-fold outputs")


if __name__ == "__main__":
    main()
