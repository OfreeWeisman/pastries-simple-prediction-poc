"""
COMPAS-2X RF Experiment: Full features vs. Hidden ring types.

Samples 10,000 molecules from compas-2x_pastries_features.csv, extracts
feature vectors using the rf_all_used_features.csv feature list, then runs
three experiments:
  1. full_features  – all features from rf_all_used_features.csv
  2. hidden_rings   – same, but ring type identities masked (topology only)
  3. paper_lalas    – the 10 curated features from Fite et al. 2023 (Table 1)

Each experiment uses 5 repeats of 5-fold CV (25 total evaluations).
Results with 95% CI are compared against the train-mean baseline.

Outputs CSV tables and PNG figures to compas-2/predictions/results/.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import RepeatedKFold

# ─── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
COMPAS2_CSV = ROOT / "compas-2" / "data" / "compas-2x_pastries_features.csv"
RF_FEATURES_CSV = DATA_DIR / "rf_all_used_features.csv"
RESULTS_DIR = ROOT / "compas-2" / "predictions" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_SIZE = 10_000
RANDOM_SEED = 42
N_SPLITS = 5
N_REPEATS = 5
CI_LEVEL = 0.95

TARGET_COLUMNS = ["homo", "lumo", "gap", "aip", "aea", "energy"]
TARGET_UNITS = {t: "eV" for t in TARGET_COLUMNS}

ALL_TARGET_COLUMNS = {
    "homo", "lumo", "homo-1", "lumo+1", "gap", "zero_point_energy",
    "dispersion", "energy", "aip", "aea", "dipole_norm", "homo_corr",
    "lumo_corr", "gap_corr", "energy_corr", "aip_corr", "aea_corr", "nfod",
}

# Fite et al. 2023 Table 1 – augmented-LALAS feature vector (aug-LFV)
PAPER_LFV_FEATURES = [
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

EXPERIMENT_LABELS = {
    "full_features": "Full Features",
    "hidden_rings": "Hidden Ring Types",
    "paper_lalas": "Paper LFV (aug-LALAS)",
    "baseline": "Baseline (train mean)",
}

EXPERIMENT_COLORS = {
    "full_features": "#2196F3",
    "hidden_rings": "#FF9800",
    "paper_lalas": "#4CAF50",
    "baseline": "#9E9E9E",
}

# ─── Ring metadata ─────────────────────────────────────────────────────────────

RING_ALIASES = {
    # Codes actually used in COMPAS-2X representation strings
    "ben": "benzene", "pyd": "pyridine", "pyz": "pyrazine", "thi": "thiophene",
    "fur": "furan",   "bor": "borole",   "cbd": "cyclobutadiene", "pyl": "pyrrole",
    "dhdb": "dhdiborinine", "dbrn": "14diborinine", "brn": "borinine",
}
RING_TYPES = [
    "benzene", "pyridine", "pyrazine", "thiophene", "furan", "borole",
    "cyclobutadiene", "pyrrole", "dhdiborinine", "14diborinine", "borinine",
]
DONOR_RINGS = {"thiophene", "furan", "pyrrole"}
ACCEPTOR_RINGS = {"pyridine", "pyrazine", "borole"}
HETEROCYCLE_RINGS = {
    "pyridine", "pyrazine", "thiophene", "furan", "borole", "pyrrole",
    "dhdiborinine", "14diborinine", "borinine",
}
SOURCE_NUMERIC = [
    "rings", "aromatic_rings", "atoms", "heteroatoms", "heterocycles",
    "branch", "cyclobutadiene", "pyrrole", "borole", "furan", "thiophene",
    "dhdiborinine", "14diborinine", "pyrazine", "pyridine", "borinine",
    "benzene", "h", "c", "b", "s", "o", "n",
]

# ─── Feature extraction ────────────────────────────────────────────────────────

def _parse_float(val: str) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(f) else f


def _split_representation(rep: str) -> tuple[str, str]:
    if " " not in rep:
        return rep, ""
    return rep.rsplit(" ", 1)


def _parse_rings(ring_part: str) -> list[tuple[str, bool]]:
    result = []
    for token in ring_part.split(","):
        stripped = token.strip()
        is_branch = stripped.startswith("(") or stripped.endswith(")")
        stripped = stripped.strip("()")
        code = stripped.split("[", 1)[0]
        ring_type = RING_ALIASES.get(code, code.lower())
        if ring_type:
            result.append((ring_type, is_branch))
    return result


def _longest_run(seq: str, val: str) -> int:
    best = cur = 0
    for ch in seq:
        if ch == val:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _longest_run_degeneracy(seq: str, val: str) -> int:
    target = _longest_run(seq, val)
    if target == 0:
        return 0
    deg = cur = 0
    for ch in seq:
        if ch == val:
            cur += 1
        else:
            if cur == target:
                deg += 1
            cur = 0
    if cur == target:
        deg += 1
    return deg


def _subseq_count(seq: str, sub: str) -> int:
    if not sub:
        return 0
    return sum(1 for i in range(len(seq) - len(sub) + 1) if seq[i:i + len(sub)] == sub)


def _max_paren_depth(s: str) -> int:
    depth = mx = 0
    for ch in s:
        if ch == "(":
            depth += 1
            mx = max(mx, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    return mx


def _dipole_norm(val: str) -> float:
    parts = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", val or "")
    v = [float(p) for p in parts]
    return math.sqrt(sum(x * x for x in v)) if v else 0.0


def extract_features(row: dict[str, str], hide_ring_identity: bool = False) -> dict:
    ring_part, topo_part = _split_representation(row["representation"])
    raw_ann = "".join(c for c in topo_part if c in {"L", "l", "A", "a"})
    raw_ann_up = raw_ann.upper()
    topo_up = topo_part.upper()
    ann_seq = "".join(c for c in topo_up if c in {"L", "A"})
    n_ann = len(ann_seq)
    n_l = ann_seq.count("L")
    n_a = ann_seq.count("A")
    la = ann_seq.count("LA")
    al = ann_seq.count("AL")

    rings = _parse_rings(ring_part)
    if hide_ring_identity:
        rings = [("benzene", is_br) for _, is_br in rings]
    n_rings = len(rings)
    ring_counts: Counter = Counter(rt for rt, _ in rings)
    branch_ring_counts: Counter = Counter(rt for rt, is_br in rings if is_br)
    term: Counter = Counter()
    mid: Counter = Counter()
    br: Counter = Counter()
    for idx, (rt, is_br) in enumerate(rings):
        if idx == 0 or idx == n_rings - 1:
            term[rt] += 1
        else:
            mid[rt] += 1
        if is_br:
            br[rt] += 1

    donor_ct = sum(ring_counts[rt] for rt in DONOR_RINGS)
    acceptor_ct = sum(ring_counts[rt] for rt in ACCEPTOR_RINGS)
    branch_ct = topo_up.count("(")

    def cls(pos: Counter, cls_set: set) -> int:
        return sum(pos[rt] for rt in cls_set)

    def non_benz(pos: Counter) -> int:
        return sum(c for rt, c in pos.items() if rt != "benzene")

    feats: dict = {
        "name": row["name"],
        # Topology
        "pas_total_rings": n_rings,
        "pas_total_annulations": n_ann,
        "pas_total_l": n_l,
        "pas_total_a": n_a,
        "pas_l_fraction": n_l / n_ann if n_ann else 0.0,
        "pas_a_fraction": n_a / n_ann if n_ann else 0.0,
        "pas_l_to_a_ratio": n_l / n_a if n_a else float(n_l),
        "pas_longest_l_run": _longest_run(ann_seq, "L"),
        "pas_longest_a_run": _longest_run(ann_seq, "A"),
        "pas_la_count": la,
        "pas_al_count": al,
        "pas_alternation_count": la + al,
        "pas_alternation_fraction": (la + al) / max(1, n_ann - 1),
        "pas_branch_count": branch_ct,
        "pas_branch_density": branch_ct / n_rings if n_rings else 0.0,
        "pas_max_branch_depth": _max_paren_depth(topo_up),
        "pas_parenthesis_count": topo_up.count("(") + topo_up.count(")"),
        "pas_topology_length": len(topo_up),
        # Donor/acceptor
        "pas_donor_ring_count": donor_ct,
        "pas_acceptor_ring_count": acceptor_ct,
        "pas_donor_acceptor_balance": donor_ct - acceptor_ct,
        "pas_donor_acceptor_ratio": donor_ct / (acceptor_ct + 1),
        "pas_donor_fraction": donor_ct / n_rings if n_rings else 0.0,
        "pas_acceptor_fraction": acceptor_ct / n_rings if n_rings else 0.0,
        # Position features
        "pas_terminal_non_benzene_count": non_benz(term),
        "pas_middle_non_benzene_count": non_benz(mid),
        "pas_branch_non_benzene_count": non_benz(br),
        "pas_terminal_heterocycle_count": cls(term, HETEROCYCLE_RINGS),
        "pas_middle_heterocycle_count": cls(mid, HETEROCYCLE_RINGS),
        "pas_branch_heterocycle_count": cls(br, HETEROCYCLE_RINGS),
        "pas_terminal_donor_count": cls(term, DONOR_RINGS),
        "pas_middle_donor_count": cls(mid, DONOR_RINGS),
        "pas_branch_donor_count": cls(br, DONOR_RINGS),
        "pas_terminal_acceptor_count": cls(term, ACCEPTOR_RINGS),
        "pas_middle_acceptor_count": cls(mid, ACCEPTOR_RINGS),
        "pas_branch_acceptor_count": cls(br, ACCEPTOR_RINGS),
        # Paper LALAS features (Fite et al. 2023, Table 1)
        "paper_lalas_number_of_rings": n_rings,
        "paper_lalas_number_of_branching_points": branch_ct,
        "paper_lalas_lal_subsequence_count": _subseq_count(raw_ann_up, "LAL"),
        "paper_lalas_l_ratio": raw_ann_up.count("L") / len(raw_ann) if raw_ann else 0.0,
        "paper_lalas_longest_l": _longest_run(raw_ann_up, "L"),
        "paper_lalas_longest_l_degeneracy": _longest_run_degeneracy(raw_ann_up, "L"),
        "paper_lalas_upper_a_ratio": raw_ann.count("A") / len(raw_ann) if raw_ann else 0.0,
        "paper_lalas_lower_to_upper_a_ratio": (
            raw_ann.count("a") / raw_ann.count("A") if raw_ann.count("A") else float(raw_ann.count("a"))
        ),
        "paper_lalas_longest_upper_a": _longest_run(raw_ann, "A"),
        "paper_lalas_longest_lower_a": _longest_run(raw_ann, "a"),
        "paper_lalas_a_ratio": raw_ann_up.count("A") / len(raw_ann) if raw_ann else 0.0,
        "paper_lalas_longest_a_case_insensitive": _longest_run(raw_ann_up, "A"),
        # Derived ratios
        "heterocycle_fraction": _parse_float(row["heterocycles"]) / _parse_float(row["rings"]) if _parse_float(row["rings"]) else 0.0,
        "heteroatom_density": _parse_float(row["heteroatoms"]) / _parse_float(row["atoms"]) if _parse_float(row["atoms"]) else 0.0,
        "aromatic_ring_fraction": _parse_float(row["aromatic_rings"]) / _parse_float(row["rings"]) if _parse_float(row["rings"]) else 0.0,
    }

    for col in SOURCE_NUMERIC:
        feats[col] = _parse_float(row[col])

    if hide_ring_identity:
        feats.update({
            "heterocycle_fraction": 0.0,
            "heteroatom_density": 0.0,
            "aromatic_ring_fraction": 1.0 if n_rings else 0.0,
            "aromatic_rings": float(n_rings),
            "heteroatoms": 0.0,
            "heterocycles": 0.0,
            "b": 0.0, "s": 0.0, "o": 0.0, "n": 0.0,
            "benzene": float(n_rings),
        })
        for rt in ["cyclobutadiene", "pyrrole", "borole", "furan", "thiophene",
                   "dhdiborinine", "14diborinine", "pyrazine", "pyridine", "borinine"]:
            feats[rt] = 0.0

    for rt in RING_TYPES:
        ct = ring_counts[rt]
        feats[f"pas_{rt}_count"] = ct
        feats[f"pas_{rt}_fraction"] = ct / n_rings if n_rings else 0.0
        feats[f"pas_{rt}_terminal_count"] = term[rt]
        feats[f"pas_{rt}_middle_count"] = mid[rt]
        feats[f"pas_{rt}_branch_count"] = branch_ring_counts[rt]

    # Targets
    for tgt in TARGET_COLUMNS:
        feats[tgt] = _parse_float(row.get(tgt, ""))
    feats["dipole_norm"] = _dipole_norm(row.get("dipole", ""))

    return feats


# ─── Model ─────────────────────────────────────────────────────────────────────

def make_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_SEED,
        min_samples_leaf=2,
        n_jobs=-1,
    )


# ─── Statistics ────────────────────────────────────────────────────────────────

def ci95(values: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, std, ci_low, ci_high) with 95% CI using t-distribution."""
    arr = np.array(values)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if n < 2:
        return mean, std, mean, mean
    sem = std / math.sqrt(n)
    lo, hi = stats.t.interval(CI_LEVEL, df=n - 1, loc=mean, scale=sem)
    return mean, std, float(lo), float(hi)


# ─── Run experiment ────────────────────────────────────────────────────────────

def run_experiment(
    df: pd.DataFrame,
    feature_cols: list[str],
    experiment_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run 5x5 RepeatedKFold CV for an RF model and baseline.
    Returns (metrics_df, importances_df).
    """
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
    fold_records: list[dict] = []
    importance_accumulator: dict[str, dict[str, list[float]]] = {
        tgt: {feat: [] for feat in feature_cols} for tgt in TARGET_COLUMNS
    }

    X = df[feature_cols].values
    indices = list(rkf.split(X))

    for fold_idx, (train_idx, test_idx) in enumerate(indices):
        X_train = df.iloc[train_idx][feature_cols]
        X_test = df.iloc[test_idx][feature_cols]

        for tgt in TARGET_COLUMNS:
            y_train = df.iloc[train_idx][tgt]
            y_test = df.iloc[test_idx][tgt]
            test_mean = float(y_test.abs().mean())
            train_std = float(y_train.std(ddof=0))
            test_std = float(y_test.std(ddof=0))

            # Baseline
            baseline = np.full(len(test_idx), float(y_train.mean()))
            b_mae = mean_absolute_error(y_test, baseline)
            fold_records.append({
                "experiment": experiment_name,
                "model": "baseline",
                "target": tgt,
                "fold": fold_idx,
                "mae": b_mae,
                "rmse": float(np.sqrt(mean_squared_error(y_test, baseline))),
                "r2": r2_score(y_test, baseline),
                "mae_percent": 100 * b_mae / test_mean if test_mean else np.nan,
                "mae_over_std": b_mae / test_std if test_std else np.nan,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            })

            # RF
            model = make_rf()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rf_mae = mean_absolute_error(y_test, y_pred)
            fold_records.append({
                "experiment": experiment_name,
                "model": "random_forest",
                "target": tgt,
                "fold": fold_idx,
                "mae": rf_mae,
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "r2": r2_score(y_test, y_pred),
                "mae_percent": 100 * rf_mae / test_mean if test_mean else np.nan,
                "mae_over_std": rf_mae / test_std if test_std else np.nan,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            })
            for feat, imp in zip(feature_cols, model.feature_importances_):
                importance_accumulator[tgt][feat].append(imp)

        if (fold_idx + 1) % 5 == 0:
            print(f"  {experiment_name}: completed {fold_idx + 1}/{N_SPLITS * N_REPEATS} folds")

    # Aggregate fold-level → per-target CI
    folds_df = pd.DataFrame(fold_records)
    agg_rows = []
    for model_name in ["baseline", "random_forest"]:
        for tgt in TARGET_COLUMNS:
            subset = folds_df[(folds_df["model"] == model_name) & (folds_df["target"] == tgt)]
            for metric in ["mae", "rmse", "r2", "mae_percent", "mae_over_std"]:
                vals = subset[metric].tolist()
                mean, std, ci_lo, ci_hi = ci95(vals)
                agg_rows.append({
                    "experiment": experiment_name,
                    "model": model_name,
                    "target": tgt,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "ci_low_95": ci_lo,
                    "ci_high_95": ci_hi,
                    "n_folds": len(vals),
                })

    # Feature importances (mean across folds)
    imp_rows = []
    for tgt in TARGET_COLUMNS:
        for feat, vals in importance_accumulator[tgt].items():
            imp_rows.append({
                "experiment": experiment_name,
                "target": tgt,
                "feature": feat,
                "mean_importance": float(np.mean(vals)),
                "std_importance": float(np.std(vals, ddof=1)),
            })

    return pd.DataFrame(agg_rows), pd.DataFrame(imp_rows)


# ─── Plotting helpers ──────────────────────────────────────────────────────────

def _bar_with_ci(
    ax,
    experiments: list[str],
    means: list[float],
    ci_lows: list[float],
    ci_highs: list[float],
    colors: list[str],
    ylabel: str,
    title: str,
) -> None:
    x = np.arange(len(experiments))
    bars = ax.bar(x, means, color=colors, alpha=0.82, edgecolor="white", linewidth=0.5)
    for i, (m, lo, hi) in enumerate(zip(means, ci_lows, ci_highs)):
        ax.errorbar(x[i], m, yerr=[[m - lo], [hi - m]], fmt="none", color="black",
                    capsize=4, linewidth=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels([EXPERIMENT_LABELS.get(e, e) for e in experiments],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)


def _get_metric(agg: pd.DataFrame, experiment: str, model: str, target: str, metric: str) -> tuple[float, float, float]:
    row = agg[
        (agg["experiment"] == experiment) &
        (agg["model"] == model) &
        (agg["target"] == target) &
        (agg["metric"] == metric)
    ]
    if row.empty:
        return float("nan"), float("nan"), float("nan")
    return float(row["mean"].iloc[0]), float(row["ci_low_95"].iloc[0]), float(row["ci_high_95"].iloc[0])


# ─── Main figures ──────────────────────────────────────────────────────────────

def plot_mae_percent(agg: pd.DataFrame, experiments: list[str]) -> None:
    """One figure with subplots per target – MAE as % of mean absolute target."""
    n_tgt = len(TARGET_COLUMNS)
    fig, axes = plt.subplots(1, n_tgt, figsize=(4 * n_tgt, 5), sharey=False)
    fig.suptitle("MAE (% of mean |target|) with 95% CI\nRF Random Forest vs Baseline",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, tgt in zip(axes, TARGET_COLUMNS):
        exp_keys = ["baseline"] + experiments
        means, ci_los, ci_his, colors = [], [], [], []
        for exp in experiments:
            m, lo, hi = _get_metric(agg, exp, "baseline", tgt, "mae_percent")
            if not means:  # only add baseline once (same for all experiments)
                means.append(m); ci_los.append(lo); ci_his.append(hi)
                colors.append(EXPERIMENT_COLORS["baseline"])
                break

        for exp in experiments:
            m, lo, hi = _get_metric(agg, exp, "random_forest", tgt, "mae_percent")
            means.append(m); ci_los.append(lo); ci_his.append(hi)
            colors.append(EXPERIMENT_COLORS.get(exp, "#607D8B"))

        labels = ["Baseline"] + [EXPERIMENT_LABELS.get(e, e) for e in experiments]
        x = np.arange(len(labels))
        bars = ax.bar(x, means, color=colors, alpha=0.82, edgecolor="white", linewidth=0.5)
        for i, (m, lo, hi) in enumerate(zip(means, ci_los, ci_his)):
            ax.errorbar(x[i], m, yerr=[[m - lo], [hi - m]], fmt="none", color="black",
                        capsize=3, linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel("MAE (%)" if tgt == TARGET_COLUMNS[0] else "", fontsize=9)
        ax.set_title(tgt.upper(), fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.2, f"{m:.1f}",
                    ha="center", va="bottom", fontsize=6.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mae_percent_by_target.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved mae_percent_by_target.png")


def plot_mae_absolute(agg: pd.DataFrame, experiments: list[str]) -> None:
    """One figure with subplots per target – MAE in raw units (eV)."""
    n_tgt = len(TARGET_COLUMNS)
    fig, axes = plt.subplots(1, n_tgt, figsize=(4 * n_tgt, 5), sharey=False)
    fig.suptitle("MAE (eV) with 95% CI\nRF Random Forest vs Baseline",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, tgt in zip(axes, TARGET_COLUMNS):
        means, ci_los, ci_his, colors, labels = [], [], [], [], []
        # Baseline (from first experiment – identical across all)
        m, lo, hi = _get_metric(agg, experiments[0], "baseline", tgt, "mae")
        means.append(m); ci_los.append(lo); ci_his.append(hi)
        colors.append(EXPERIMENT_COLORS["baseline"])
        labels.append("Baseline")

        for exp in experiments:
            m, lo, hi = _get_metric(agg, exp, "random_forest", tgt, "mae")
            means.append(m); ci_los.append(lo); ci_his.append(hi)
            colors.append(EXPERIMENT_COLORS.get(exp, "#607D8B"))
            labels.append(EXPERIMENT_LABELS.get(exp, exp))

        x = np.arange(len(labels))
        bars = ax.bar(x, means, color=colors, alpha=0.82, edgecolor="white", linewidth=0.5)
        for i, (m, lo, hi) in enumerate(zip(means, ci_los, ci_his)):
            ax.errorbar(x[i], m, yerr=[[m - lo], [hi - m]], fmt="none", color="black",
                        capsize=3, linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel("MAE (eV)" if tgt == TARGET_COLUMNS[0] else "", fontsize=9)
        ax.set_title(tgt.upper(), fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.002, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mae_absolute_by_target.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved mae_absolute_by_target.png")


def plot_r2(agg: pd.DataFrame, experiments: list[str]) -> None:
    """R² comparison across experiments and targets."""
    n_tgt = len(TARGET_COLUMNS)
    fig, axes = plt.subplots(1, n_tgt, figsize=(4 * n_tgt, 5), sharey=False)
    fig.suptitle("R² Score with 95% CI", fontsize=13, fontweight="bold", y=1.02)

    for ax, tgt in zip(axes, TARGET_COLUMNS):
        means, ci_los, ci_his, colors, labels = [], [], [], [], []
        for exp in experiments:
            m, lo, hi = _get_metric(agg, exp, "random_forest", tgt, "r2")
            means.append(m); ci_los.append(lo); ci_his.append(hi)
            colors.append(EXPERIMENT_COLORS.get(exp, "#607D8B"))
            labels.append(EXPERIMENT_LABELS.get(exp, exp))

        x = np.arange(len(labels))
        bars = ax.bar(x, means, color=colors, alpha=0.82, edgecolor="white", linewidth=0.5)
        for i, (m, lo, hi) in enumerate(zip(means, ci_los, ci_his)):
            err_lo = max(m - lo, 0.0)  # clamp to non-negative
            err_hi = max(hi - m, 0.0)
            ax.errorbar(x[i], m, yerr=[[err_lo], [err_hi]], fmt="none", color="black",
                        capsize=3, linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel("R²" if tgt == TARGET_COLUMNS[0] else "", fontsize=9)
        ax.set_title(tgt.upper(), fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        for bar, m in zip(bars, means):
            offset = 0.01 if m >= 0 else -0.03
            ax.text(bar.get_x() + bar.get_width() / 2, m + offset, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=6.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "r2_by_target.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved r2_by_target.png")


def plot_experiment_comparison(agg: pd.DataFrame, experiments: list[str]) -> None:
    """
    Side-by-side comparison of full_features vs hidden_rings for each target.
    Shows both MAE% and R².
    """
    if len(experiments) < 2:
        return

    fig, axes = plt.subplots(2, len(TARGET_COLUMNS), figsize=(4 * len(TARGET_COLUMNS), 9))
    fig.suptitle("Full Features vs. Hidden Ring Types\n(5×5 RepeatedKFold CV, 95% CI)",
                 fontsize=13, fontweight="bold")

    for col, tgt in enumerate(TARGET_COLUMNS):
        # MAE%
        ax_mae = axes[0, col]
        for exp in experiments[:2]:  # Only first two experiments for this comparison
            m, lo, hi = _get_metric(agg, exp, "random_forest", tgt, "mae_percent")
            color = EXPERIMENT_COLORS.get(exp, "#607D8B")
            label = EXPERIMENT_LABELS.get(exp, exp)
            x_pos = experiments[:2].index(exp)
            bar = ax_mae.bar(x_pos, m, color=color, alpha=0.82, width=0.6,
                             edgecolor="white", linewidth=0.5, label=label)
            ax_mae.errorbar(x_pos, m, yerr=[[m - lo], [hi - m]], fmt="none",
                            color="black", capsize=4, linewidth=1.4)
            ax_mae.text(x_pos, m + 0.2, f"{m:.1f}%", ha="center", va="bottom", fontsize=8)

        ax_mae.set_xticks([0, 1])
        ax_mae.set_xticklabels(["Full", "Hidden"], fontsize=9)
        ax_mae.set_title(tgt.upper(), fontsize=11, fontweight="bold")
        ax_mae.set_ylabel("MAE (%)" if col == 0 else "", fontsize=9)
        ax_mae.spines["top"].set_visible(False)
        ax_mae.spines["right"].set_visible(False)
        ax_mae.set_ylim(bottom=0)

        # R²
        ax_r2 = axes[1, col]
        for exp in experiments[:2]:
            m, lo, hi = _get_metric(agg, exp, "random_forest", tgt, "r2")
            color = EXPERIMENT_COLORS.get(exp, "#607D8B")
            x_pos = experiments[:2].index(exp)
            ax_r2.bar(x_pos, m, color=color, alpha=0.82, width=0.6,
                      edgecolor="white", linewidth=0.5)
            err_lo = max(m - lo, 0.0)
            err_hi = max(hi - m, 0.0)
            ax_r2.errorbar(x_pos, m, yerr=[[err_lo], [err_hi]], fmt="none",
                           color="black", capsize=4, linewidth=1.4)
            ax_r2.text(x_pos, m + 0.01, f"{m:.2f}", ha="center", va="bottom", fontsize=8)

        ax_r2.set_xticks([0, 1])
        ax_r2.set_xticklabels(["Full", "Hidden"], fontsize=9)
        ax_r2.set_ylabel("R²" if col == 0 else "", fontsize=9)
        ax_r2.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax_r2.spines["top"].set_visible(False)
        ax_r2.spines["right"].set_visible(False)

    # Legend
    patches = [
        mpatches.Patch(color=EXPERIMENT_COLORS["full_features"], alpha=0.82, label="Full Features"),
        mpatches.Patch(color=EXPERIMENT_COLORS["hidden_rings"], alpha=0.82, label="Hidden Ring Types"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "full_vs_hidden_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved full_vs_hidden_comparison.png")


def plot_normalized_summary(agg: pd.DataFrame, experiments: list[str]) -> None:
    """
    Bar chart: average MAE/std across all targets for each experiment.
    Similar to Figure 3 (top row) in Fite et al. 2023.
    """
    exp_means, exp_ci_los, exp_ci_his, colors = [], [], [], []

    for exp in experiments:
        per_target_scores = []
        for tgt in TARGET_COLUMNS:
            m, lo, hi = _get_metric(agg, exp, "random_forest", tgt, "mae_over_std")
            per_target_scores.append(m)
        overall_mean = float(np.nanmean(per_target_scores))
        overall_sem = float(np.nanstd(per_target_scores, ddof=1) / math.sqrt(len(per_target_scores)))
        lo = overall_mean - 1.96 * overall_sem
        hi = overall_mean + 1.96 * overall_sem
        exp_means.append(overall_mean)
        exp_ci_los.append(lo)
        exp_ci_his.append(hi)
        colors.append(EXPERIMENT_COLORS.get(exp, "#607D8B"))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(experiments))
    bars = ax.bar(x, exp_means, color=colors, alpha=0.82, edgecolor="white", linewidth=0.5)
    for i, (m, lo, hi) in enumerate(zip(exp_means, exp_ci_los, exp_ci_his)):
        ax.errorbar(x[i], m, yerr=[[m - lo], [hi - m]], fmt="none", color="black",
                    capsize=5, linewidth=1.5)
        ax.text(x[i], m + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([EXPERIMENT_LABELS.get(e, e) for e in experiments],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Average MAE / std (normalized, unitless)", fontsize=10)
    ax.set_title("Overall Performance: Normalized Average MAE across All Targets\n"
                 "(lower is better; analogous to Fite et al. 2023 Fig. 3 top)",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "normalized_mae_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved normalized_mae_summary.png")


def plot_top_features(imp_df: pd.DataFrame, experiment: str, n_top: int = 15) -> None:
    """Top N feature importances for each target, for a given experiment."""
    exp_imp = imp_df[imp_df["experiment"] == experiment]
    n_tgt = len(TARGET_COLUMNS)
    fig, axes = plt.subplots(1, n_tgt, figsize=(5.5 * n_tgt, 6))
    fig.suptitle(f"Top {n_top} Feature Importances — {EXPERIMENT_LABELS.get(experiment, experiment)}\n"
                 f"(mean ± std across {N_SPLITS * N_REPEATS} CV folds)",
                 fontsize=12, fontweight="bold", y=1.01)

    for ax, tgt in zip(axes, TARGET_COLUMNS):
        tgt_imp = exp_imp[exp_imp["target"] == tgt].nlargest(n_top, "mean_importance")
        y = np.arange(len(tgt_imp))
        ax.barh(y, tgt_imp["mean_importance"], xerr=tgt_imp["std_importance"],
                color=EXPERIMENT_COLORS.get(experiment, "#607D8B"), alpha=0.82,
                error_kw={"elinewidth": 1.0, "capsize": 2})
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f.replace("paper_lalas_", "pl_").replace("pas_", "") for f in tgt_imp["feature"]],
            fontsize=7,
        )
        ax.set_xlabel("Mean Importance", fontsize=8)
        ax.set_title(tgt.upper(), fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.invert_yaxis()

    plt.tight_layout()
    fname = RESULTS_DIR / f"top_features_{experiment}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved top_features_{experiment}.png")


# ─── Summary tables ────────────────────────────────────────────────────────────

def build_summary_table(agg: pd.DataFrame, experiments: list[str]) -> pd.DataFrame:
    """Wide-format table: targets as rows, experiments × metrics as columns."""
    rows = []
    for tgt in TARGET_COLUMNS:
        row = {"target": tgt}
        # Baseline (same for all experiments, use first)
        m, lo, hi = _get_metric(agg, experiments[0], "baseline", tgt, "mae")
        row["baseline_mae_mean"] = round(m, 4)
        row["baseline_mae_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
        m_pct, lo_pct, hi_pct = _get_metric(agg, experiments[0], "baseline", tgt, "mae_percent")
        row["baseline_mae_pct_mean"] = round(m_pct, 2)

        for exp in experiments:
            prefix = exp
            for metric, col_suffix in [("mae", "mae_mean"), ("mae_percent", "mae_pct_mean"),
                                        ("r2", "r2_mean"), ("mae_over_std", "mae_std_mean")]:
                m, lo, hi = _get_metric(agg, exp, "random_forest", tgt, metric)
                row[f"{prefix}_{col_suffix}"] = round(m, 4)
                row[f"{prefix}_{col_suffix}_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def build_ci_table(agg: pd.DataFrame, experiments: list[str]) -> pd.DataFrame:
    """Long-format CI table for all experiments, models, targets, metrics."""
    return agg[agg["experiment"].isin(experiments)].copy()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    # 1. Load rf_all_used_features.csv
    print("Loading feature list from rf_all_used_features.csv ...")
    features_df = pd.read_csv(RF_FEATURES_CSV)
    rf_feature_list = features_df["feature"].tolist()
    print(f"  {len(rf_feature_list)} features in rf_all_used_features.csv")

    # 2. Load COMPAS-2X data
    print(f"Loading COMPAS-2X data from {COMPAS2_CSV.name} ...")
    raw = pd.read_csv(COMPAS2_CSV)
    print(f"  Total rows: {len(raw):,}")

    # 3. Sample 10K molecules (same sample for both experiments)
    sample_raw = raw.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"  Sampled {SAMPLE_SIZE:,} molecules (seed={RANDOM_SEED})")

    # 4. Extract feature vectors – full and hidden ring types
    print("Extracting feature vectors (full) ...")
    full_rows = [extract_features(row, hide_ring_identity=False)
                 for row in sample_raw.to_dict(orient="records")]
    df_full = pd.DataFrame(full_rows)

    print("Extracting feature vectors (hidden ring types) ...")
    hidden_rows = [extract_features(row, hide_ring_identity=True)
                   for row in sample_raw.to_dict(orient="records")]
    df_hidden = pd.DataFrame(hidden_rows)

    # Save feature vectors
    df_full.to_csv(RESULTS_DIR / "sample_10k_feature_vectors_full.csv", index=False)
    df_hidden.to_csv(RESULTS_DIR / "sample_10k_feature_vectors_hidden.csv", index=False)
    print("  Saved sample_10k_feature_vectors_full.csv and _hidden.csv")

    # 5. Validate feature lists
    available_full = set(df_full.columns)
    missing_rf = [f for f in rf_feature_list if f not in available_full]
    if missing_rf:
        print(f"  WARNING: {len(missing_rf)} rf features not found in extracted data: {missing_rf[:5]}...")
    rf_features_valid = [f for f in rf_feature_list if f in available_full]
    print(f"  Using {len(rf_features_valid)} features for full/hidden experiments")

    paper_features_valid = [f for f in PAPER_LFV_FEATURES if f in available_full]
    print(f"  Using {len(paper_features_valid)} paper LFV features")

    # 6. Run experiments (skip if cached results exist)
    cv_cache = RESULTS_DIR / "cv_results_ci95.csv"
    imp_cache = RESULTS_DIR / "feature_importances.csv"
    experiments = ["full_features", "hidden_rings", "paper_lalas"]

    if cv_cache.exists() and imp_cache.exists():
        print("\nLoading cached CV results ...")
        agg = pd.read_csv(cv_cache)
        imp = pd.read_csv(imp_cache)
        print(f"  Loaded {len(agg)} rows from cache")
    else:
        all_agg: list[pd.DataFrame] = []
        all_imp: list[pd.DataFrame] = []

        experiment_configs = [
            ("full_features", df_full, rf_features_valid),
            ("hidden_rings", df_hidden, rf_features_valid),
            ("paper_lalas", df_full, paper_features_valid),
        ]

        for exp_name, df, feats in experiment_configs:
            print(f"\nRunning experiment: {exp_name} ({len(feats)} features) ...")
            agg_df, imp_df = run_experiment(df, feats, exp_name)
            all_agg.append(agg_df)
            all_imp.append(imp_df)
            elapsed = time.time() - t0
            print(f"  Done ({elapsed:.0f}s elapsed)")

        agg = pd.concat(all_agg, ignore_index=True)
        imp = pd.concat(all_imp, ignore_index=True)

    # 7. Save tables
    print("\nSaving result tables ...")
    agg.to_csv(RESULTS_DIR / "cv_results_ci95.csv", index=False)
    imp.to_csv(RESULTS_DIR / "feature_importances.csv", index=False)
    top20 = (
        imp.sort_values("mean_importance", ascending=False)
        .groupby(["experiment", "target"], group_keys=False)
        .head(20)
    )
    top20.to_csv(RESULTS_DIR / "top20_feature_importances.csv", index=False)

    summary = build_summary_table(agg, experiments)
    summary.to_csv(RESULTS_DIR / "results_summary_table.csv", index=False)
    print("  Saved cv_results_ci95.csv, feature_importances.csv, results_summary_table.csv")

    # 8. Generate figures
    print("\nGenerating figures ...")
    plot_mae_percent(agg, experiments)
    plot_mae_absolute(agg, experiments)
    plot_r2(agg, experiments)
    plot_experiment_comparison(agg, experiments)
    plot_normalized_summary(agg, experiments)
    for exp_name in ["full_features", "hidden_rings"]:
        plot_top_features(imp, exp_name)

    # 9. Print summary to console
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (mean across 25 CV folds, 95% CI)")
    print("=" * 70)
    print(f"{'Target':<8} {'Experiment':<25} {'MAE':>8} {'MAE%':>8} {'R²':>8}")
    print("-" * 70)
    for tgt in TARGET_COLUMNS:
        b_mae, _, _ = _get_metric(agg, "full_features", "baseline", tgt, "mae")
        b_pct, _, _ = _get_metric(agg, "full_features", "baseline", tgt, "mae_percent")
        print(f"{tgt:<8} {'baseline':<25} {b_mae:>8.4f} {b_pct:>7.2f}% {'---':>8}")
        for exp in experiments:
            m_mae, lo_mae, hi_mae = _get_metric(agg, exp, "random_forest", tgt, "mae")
            m_pct, _, _ = _get_metric(agg, exp, "random_forest", tgt, "mae_percent")
            m_r2, _, _ = _get_metric(agg, exp, "random_forest", tgt, "r2")
            label = EXPERIMENT_LABELS.get(exp, exp)[:24]
            print(f"{'':>8} {label:<25} {m_mae:>8.4f} {m_pct:>7.2f}% {m_r2:>8.4f}")
        print()

    print(f"\nTotal runtime: {time.time() - t0:.0f}s")
    print(f"All results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
