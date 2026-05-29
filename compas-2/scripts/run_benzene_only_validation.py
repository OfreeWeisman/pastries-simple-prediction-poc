"""
Benzene-only validation of the paper LFV (aug-LALAS) 10-feature vector.

Filters COMPAS-2X to molecules containing only benzene rings (pure carbocyclic PAHs),
then runs the same RF + 5x5 RepeatedKFold CV using only the 10 paper LFV features.

Hypothesis: on a chemically homogeneous subset where topology is the dominant
structural variable, the 10-feature LFV should be competitive — validating the
paper's claim while explaining why the full COMPAS-2X experiment underperformed.

Saves results to compas-2/predictions/results/ with benzene_only_ prefix.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import RepeatedKFold

# Import shared utilities from the main experiment script
from run_compas2_experiment import (
    extract_features,
    PAPER_LFV_FEATURES,
    TARGET_COLUMNS,
    EXPERIMENT_COLORS,
    ci95,
    make_rf,
)

RESULTS_DIR = ROOT / "compas-2" / "predictions" / "results"
COMPAS2_CSV = ROOT / "compas-2" / "data" / "compas-2x_pastries_features.csv"

RING_COLS = [
    "pyridine", "pyrazine", "pyrrole", "furan", "thiophene", "borole",
    "borinine", "14diborinine", "dhdiborinine", "cyclobutadiene",
]

N_SPLITS  = 5
N_REPEATS = 5
CI_LEVEL  = 0.95
RANDOM_SEED = 42


def run_cv(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """5x5 RepeatedKFold CV; returns per-fold records."""
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
    records = []
    X = df[feature_cols].values
    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X)):
        for tgt in TARGET_COLUMNS:
            y_train = df.iloc[train_idx][tgt]
            y_test  = df.iloc[test_idx][tgt]
            test_mean = float(y_test.abs().mean())
            test_std  = float(y_test.std(ddof=0))

            # Baseline
            baseline = np.full(len(test_idx), float(y_train.mean()))
            b_mae = mean_absolute_error(y_test, baseline)
            records.append(dict(
                model="baseline", target=tgt, fold=fold_idx,
                mae=b_mae,
                rmse=float(np.sqrt(mean_squared_error(y_test, baseline))),
                r2=r2_score(y_test, baseline),
                mae_percent=100 * b_mae / test_mean if test_mean else np.nan,
                mae_over_std=b_mae / test_std if test_std else np.nan,
            ))

            # RF
            model = make_rf()
            model.fit(df.iloc[train_idx][feature_cols], y_train)
            y_pred = model.predict(df.iloc[test_idx][feature_cols])
            rf_mae = mean_absolute_error(y_test, y_pred)
            records.append(dict(
                model="random_forest", target=tgt, fold=fold_idx,
                mae=rf_mae,
                rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
                r2=r2_score(y_test, y_pred),
                mae_percent=100 * rf_mae / test_mean if test_mean else np.nan,
                mae_over_std=rf_mae / test_std if test_std else np.nan,
            ))
    return pd.DataFrame(records)


def aggregate(folds_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in ["baseline", "random_forest"]:
        for tgt in TARGET_COLUMNS:
            subset = folds_df[(folds_df["model"] == model_name) & (folds_df["target"] == tgt)]
            for metric in ["mae", "rmse", "r2", "mae_percent", "mae_over_std"]:
                vals = subset[metric].tolist()
                mean, std, ci_lo, ci_hi = ci95(vals)
                rows.append(dict(
                    model=model_name, target=tgt, metric=metric,
                    mean=mean, std=std, ci_low_95=ci_lo, ci_high_95=ci_hi,
                    n_folds=len(vals),
                ))
    return pd.DataFrame(rows)


def get_metric(agg, model, target, metric):
    row = agg[(agg["model"] == model) & (agg["target"] == target) & (agg["metric"] == metric)]
    if row.empty:
        return float("nan"), float("nan"), float("nan")
    return float(row["mean"].iloc[0]), float(row["ci_low_95"].iloc[0]), float(row["ci_high_95"].iloc[0])


def plot_results(agg: pd.DataFrame, n_molecules: int) -> None:
    n_tgt = len(TARGET_COLUMNS)
    color_rf   = EXPERIMENT_COLORS["paper_lalas"]
    color_base = EXPERIMENT_COLORS["baseline"]

    fig, axes = plt.subplots(2, n_tgt, figsize=(4 * n_tgt, 9))
    fig.suptitle(
        f"Benzene-Only Validation: Paper LFV (10 features)\n"
        f"n={n_molecules} molecules, 5×5 RepeatedKFold CV, 95% CI",
        fontsize=13, fontweight="bold",
    )

    for col, tgt in enumerate(TARGET_COLUMNS):
        # MAE %
        ax = axes[0, col]
        b_m, b_lo, b_hi = get_metric(agg, "baseline", tgt, "mae_percent")
        r_m, r_lo, r_hi = get_metric(agg, "random_forest", tgt, "mae_percent")
        means  = [b_m, r_m]
        ci_los = [b_lo, r_lo]
        ci_his = [b_hi, r_hi]
        colors = [color_base, color_rf]
        labels = ["Baseline", "Paper LFV"]
        x = np.arange(2)
        bars = ax.bar(x, means, color=colors, alpha=0.82, edgecolor="white")
        for i, (m, lo, hi) in enumerate(zip(means, ci_los, ci_his)):
            ax.errorbar(x[i], m, yerr=[[m - lo], [hi - m]], fmt="none",
                        color="black", capsize=4, linewidth=1.2)
            ax.text(x[i], m + 0.3, f"{m:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(tgt.upper(), fontsize=11, fontweight="bold")
        ax.set_ylabel("MAE (%)" if col == 0 else "", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)

        # R²
        ax2 = axes[1, col]
        r_m2, r_lo2, r_hi2 = get_metric(agg, "random_forest", tgt, "r2")
        bar = ax2.bar([0], [r_m2], color=color_rf, alpha=0.82, edgecolor="white")
        err_lo = max(r_m2 - r_lo2, 0.0)
        err_hi = max(r_hi2 - r_m2, 0.0)
        ax2.errorbar(0, r_m2, yerr=[[err_lo], [err_hi]], fmt="none",
                     color="black", capsize=4, linewidth=1.2)
        ax2.text(0, r_m2 + 0.01, f"{r_m2:.2f}", ha="center", va="bottom", fontsize=9)
        ax2.set_xticks([0])
        ax2.set_xticklabels(["Paper LFV"], fontsize=9)
        ax2.set_ylabel("R²" if col == 0 else "", fontsize=9)
        ax2.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    out = RESULTS_DIR / "benzene_only_paper_lfv_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


def main():
    print("Loading COMPAS-2X ...")
    raw = pd.read_csv(COMPAS2_CSV)
    benzene_only_raw = raw[(raw["benzene"] > 0) & (raw[RING_COLS].sum(axis=1) == 0)].reset_index(drop=True)
    n = len(benzene_only_raw)
    print(f"  Benzene-only molecules: {n}")

    print("Extracting features ...")
    rows = [extract_features(row, hide_ring_identity=False)
            for row in benzene_only_raw.to_dict(orient="records")]
    df = pd.DataFrame(rows)

    valid_feats = [f for f in PAPER_LFV_FEATURES if f in df.columns]
    print(f"  Using {len(valid_feats)} paper LFV features")

    print("Running 5x5 CV ...")
    folds_df = run_cv(df, valid_feats)
    agg = aggregate(folds_df)

    # Save tables
    agg.to_csv(RESULTS_DIR / "benzene_only_cv_results.csv", index=False)
    print("  Saved benzene_only_cv_results.csv")

    # Save summary
    summary_rows = []
    for tgt in TARGET_COLUMNS:
        b_mae, *_ = get_metric(agg, "baseline", tgt, "mae")
        b_pct, *_ = get_metric(agg, "baseline", tgt, "mae_percent")
        r_mae, r_lo, r_hi = get_metric(agg, "random_forest", tgt, "mae")
        r_pct, *_ = get_metric(agg, "random_forest", tgt, "mae_percent")
        r_r2,  *_ = get_metric(agg, "random_forest", tgt, "r2")
        summary_rows.append(dict(
            target=tgt,
            n_molecules=n,
            baseline_mae=round(b_mae, 4),
            baseline_mae_pct=round(b_pct, 2),
            paper_lfv_mae=round(r_mae, 4),
            paper_lfv_mae_ci95=f"[{r_lo:.4f}, {r_hi:.4f}]",
            paper_lfv_mae_pct=round(r_pct, 2),
            paper_lfv_r2=round(r_r2, 4),
        ))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "benzene_only_summary.csv", index=False)
    print("  Saved benzene_only_summary.csv")

    # Print to console
    print("\n" + "=" * 65)
    print(f"BENZENE-ONLY VALIDATION  (n={n}, 25 CV folds)")
    print("=" * 65)
    print(f"{'Target':<8} {'Baseline MAE':>13} {'RF MAE':>10} {'MAE%':>8} {'R²':>8}")
    print("-" * 65)
    for row in summary_rows:
        print(f"{row['target']:<8} {row['baseline_mae']:>13.4f} {row['paper_lfv_mae']:>10.4f} "
              f"{row['paper_lfv_mae_pct']:>7.2f}% {row['paper_lfv_r2']:>8.4f}")

    plot_results(agg, n)
    print(f"\nAll results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
