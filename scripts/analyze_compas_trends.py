from __future__ import annotations

from pathlib import Path
import difflib
import re
import warnings

import matplotlib

matplotlib.use("Agg")

from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, is_color_like
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("data") / "compas-2x_pastries_features.csv"
OUT_DIR = Path("data_analysis")
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
MANUAL_COLORS_PATH = OUT_DIR / "colors.csv"
RING_TYPE_LABELS_PATH = OUT_DIR / "ring types.txt"

PROPERTY_COLUMNS = ["homo", "lumo", "gap", "aip", "aea"]
RING_COLUMNS = [
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
]
NON_BENZENE_RINGS = [ring for ring in RING_COLUMNS if ring != "benzene"]
FIGURE9B_RING_ORDER = [
    "cyclobutadiene",  # Cbd
    "borinine",  # Bz
    "14diborinine",  # DBn
    "dhdiborinine",  # DhDBn
    "borole",  # Bl
    "pyridine",  # Pd
    "pyrazine",  # Pz
    "pyrrole",  # Py
    "furan",  # Fu
    "thiophene",  # Th
]
HETEROATOM_COLUMNS = ["b", "n", "o", "s"]
ANTIAROMATIC_RINGS = ["cyclobutadiene", "borole", "dhdiborinine"]

RING_CODE_TO_TYPE = {
    "Bn": "benzene",
    "Cbd": "cyclobutadiene",
    "Py": "pyrrole",
    "Bl": "borole",
    "Fu": "furan",
    "Th": "thiophene",
    "DhDBn": "dhdiborinine",
    "DBn": "14diborinine",
    "Pz": "pyrazine",
    "Pd": "pyridine",
    "Bz": "borinine",
}

RING_TYPE_TO_CODE = {ring_type: code for code, ring_type in RING_CODE_TO_TYPE.items()}

DEFAULT_RING_LABELS = {
    "benzene": "Benzene (Bn)",
    "pyridine": "Pyridine (Pd)",
    "pyrazine": "Pyrazine (Pz)",
    "pyrrole": "Pyrrole (Py)",
    "furan": "Furan (Fu)",
    "thiophene": "Thiophene (Th)",
    "borole": "Borole (Bl)",
    "borinine": "Borazine (Bz)",
    "14diborinine": "DiBorinine (DBn)",
    "dhdiborinine": "Dhydrodiborinine (DhDBn)",
    "cyclobutadiene": "Cyclobutadiene (Cbd)",
}


def load_ring_labels() -> dict[str, str]:
    labels = DEFAULT_RING_LABELS.copy()
    if not RING_TYPE_LABELS_PATH.exists():
        return labels

    with RING_TYPE_LABELS_PATH.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or "(" not in line or ")" not in line:
                continue
            code = line.rsplit("(", 1)[1].split(")", 1)[0].strip()
            ring_type = RING_CODE_TO_TYPE.get(code)
            if ring_type:
                labels[ring_type] = line
    return labels


RING_LABELS = load_ring_labels()

RING_COLORS = {
    "cyclobutadiene": "#6f6f6f",
    "pyrrole": "#4f8fc9",
    "borole": "#d95f5f",
    "furan": "#4faa68",
    "thiophene": "#9b79c6",
    "dhdiborinine": "#b23a48",
    "14diborinine": "#e08367",
    "pyrazine": "#2f6fae",
    "pyridine": "#75aadb",
    "borinine": "#f0a35c",
    "benzene": "#bdbdbd",
}

ATOM_COLORS = {
    "b": "#c44e52",
    "n": "#4c72b0",
    "o": "#55a868",
    "s": "#8172b3",
}

ELEMENT_COLORMAPS = {
    "B": "Greens",
    "N": "Blues",
    "O": "Purples",
    "S": "YlOrBr",
    "none": "Greys",
}

FALLBACK_ELEMENT_COLORMAPS = {
    "B": "YlOrRd",
    "N": "Blues",
    "O": "Greens",
    "S": "Purples",
    "none": "Greys",
}

RING_PRIMARY_ELEMENT = {
    "benzene": "none",
    "cyclobutadiene": "none",
    "pyrrole": "N",
    "borole": "B",
    "furan": "O",
    "thiophene": "S",
    "dhdiborinine": "B",
    "14diborinine": "B",
    "pyrazine": "N",
    "pyridine": "N",
    "borinine": "B",
}


def split_ring_tokens(ring_part: str) -> list[str]:
    tokens = []
    current = []
    bracket_depth = 0
    for char in ring_part:
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        if char == "," and bracket_depth == 0:
            token = "".join(current).strip().strip("()")
            if token:
                tokens.append(token)
            current = []
        else:
            current.append(char)
    token = "".join(current).strip().strip("()")
    if token:
        tokens.append(token)
    return tokens


def parse_representation_variants(representation: str) -> list[tuple[str, str, str, str]]:
    ring_part = representation.rsplit(" ", 1)[0]
    variants = []
    for token in split_ring_tokens(ring_part):
        code = token.split("[", 1)[0]
        ring_type = RING_CODE_TO_TYPE.get(code, code.lower())
        spec = ""
        if "[" in token and "]" in token:
            spec = token.split("[", 1)[1].split("]", 1)[0]
        element = "none"
        positions = "none"
        if ":" in spec:
            element, positions = spec.split(":", 1)
        variant = f"{ring_type}__{element}_{positions.replace(',', '_')}"
        variants.append((ring_type, element, positions, variant))
    return variants


def indexed_variant_label(variant: str) -> str:
    ring_type, spec = variant.split("__", 1)
    element, positions = spec.split("_", 1)
    if element == "none":
        return RING_LABELS.get(ring_type, ring_type)
    return f"{RING_LABELS.get(ring_type, ring_type)} [{element}:{positions.replace('_', ',')}]"


def short_variant_label(variant: str) -> str:
    ring_type, spec = variant.split("__", 1)
    element, positions = spec.split("_", 1)
    code = RING_TYPE_TO_CODE.get(ring_type, ring_type)
    if element == "none":
        return code
    return f"{code} [{element}:{positions.replace('_', ',')}]"


def parse_manual_color_key(key: str) -> list[str]:
    key = key.strip()
    if not key:
        return []
    matches = [key]

    if "__" in key:
        ring_type = key.split("__", 1)[0]
        matches.append(ring_type)
        return matches

    bracket_match = re.fullmatch(r"([A-Za-z0-9]+)\s+\[([A-Za-z]+):([0-9,]+)\]", key)
    if bracket_match:
        code, element, positions = bracket_match.groups()
        ring_type = RING_CODE_TO_TYPE.get(code, code.lower())
        variant = f"{ring_type}__{element}_{positions.replace(',', '_')}"
        matches.extend([variant, ring_type, indexed_variant_label(variant), short_variant_label(variant)])
        return matches

    ring_type = RING_CODE_TO_TYPE.get(key, key.lower())
    matches.extend([ring_type, f"{ring_type}__none_none"])
    return matches


def load_manual_colors() -> tuple[dict[str, str], list[str]]:
    if not MANUAL_COLORS_PATH.exists():
        return {}, []

    colors: dict[str, str] = {}
    order: list[str] = []
    with MANUAL_COLORS_PATH.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            color_match = re.search(r"(.+?)[,\s]+(\#[0-9A-Fa-f]{3,8}|[A-Za-z][A-Za-z0-9_]*)$", line)
            if not color_match:
                warnings.warn(
                    f"Skipping malformed color row {line_number}: {raw_line.rstrip()}",
                    stacklevel=2,
                )
                continue
            key, color = color_match.groups()
            key = key.strip()
            color = color.strip()

            if not is_color_like(color):
                warnings.warn(
                    f"Skipping invalid color '{color}' on row {line_number}: {raw_line.rstrip()}",
                    stacklevel=2,
                )
                continue

            parsed_keys = parse_manual_color_key(key)
            for parsed_key in parsed_keys:
                colors[parsed_key] = color
            for parsed_key in parsed_keys:
                if "__" in parsed_key and parsed_key not in order:
                    order.append(parsed_key)
                    break
    return colors, order


MANUAL_COLORS, MANUAL_COLOR_ORDER = load_manual_colors()


def colors_by_element(items: list[str], elements: dict[str, str]) -> dict[str, object]:
    colors = {}
    grouped: dict[str, list[str]] = {}
    for item in items:
        if item in MANUAL_COLORS:
            colors[item] = MANUAL_COLORS[item]
            continue
        grouped.setdefault(elements.get(item, "none"), []).append(item)

    for element, element_items in grouped.items():
        cmap = element_colormap(element)
        if element == "none":
            shades = np.linspace(0.2, 0.8, max(1, len(element_items)))
        else:
            shades = np.linspace(0.2, 0.8, max(1, len(element_items)))
        for item, shade in zip(element_items, shades):
            colors[item] = MANUAL_COLORS.get(item, cmap(shade))
    return colors


def element_colormap(element: str):
    value = ELEMENT_COLORMAPS.get(element, FALLBACK_ELEMENT_COLORMAPS.get(element, "Greys"))
    if value in colormaps:
        return colormaps[value]

    if is_color_like(value):
        return LinearSegmentedColormap.from_list(f"{element}_{value}", ["#f7f7f7", value])

    fallback = FALLBACK_ELEMENT_COLORMAPS.get(element, "Greys")
    suggestion = difflib.get_close_matches(value, list(colormaps), n=1)
    hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
    warnings.warn(
        f"Invalid colormap/color '{value}' for element '{element}'. Using '{fallback}'.{hint}",
        stacklevel=2,
    )
    return colormaps[fallback]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    for extension in ["png", "svg"]:
        plt.savefig(FIG_DIR / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close()


def load_data() -> pd.DataFrame:
    usecols = [
        "name",
        "formula",
        "representation",
        "rings",
        "aromatic_rings",
        "atoms",
        "heteroatoms",
        "heterocycles",
        "branch",
        *RING_COLUMNS,
        *HETEROATOM_COLUMNS,
        *PROPERTY_COLUMNS,
        "nfod",
    ]
    frame = pd.read_csv(DATA_PATH, usecols=usecols)
    for column in [
        "rings",
        "aromatic_rings",
        "atoms",
        "heteroatoms",
        "heterocycles",
        "branch",
        *RING_COLUMNS,
        *HETEROATOM_COLUMNS,
        *PROPERTY_COLUMNS,
        "nfod",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["non_benzene_ring_count"] = frame[NON_BENZENE_RINGS].sum(axis=1)
    frame["antiaromatic_ring_count"] = frame[ANTIAROMATIC_RINGS].sum(axis=1)
    frame["heterocycle_fraction"] = frame["heterocycles"] / frame["rings"].replace(0, np.nan)
    frame["heteroatom_density"] = frame["heteroatoms"] / frame["atoms"].replace(0, np.nan)
    frame["aromatic_ring_fraction"] = frame["aromatic_rings"] / frame["rings"].replace(0, np.nan)
    frame["dominant_non_benzene_ring"] = frame[NON_BENZENE_RINGS].idxmax(axis=1)
    frame.loc[frame["non_benzene_ring_count"] == 0, "dominant_non_benzene_ring"] = "benzene_only"
    frame["mixed_non_benzene_ring_types"] = (frame[NON_BENZENE_RINGS] > 0).sum(axis=1)
    return frame


def trimmed_bins(values: pd.Series, bins: int = 20) -> np.ndarray:
    lower, upper = values.quantile([0.01, 0.99])
    if lower == upper:
        lower, upper = values.min(), values.max()
    return np.linspace(lower, upper, bins + 1)


def property_stacked_prevalence(
    frame: pd.DataFrame,
    property_column: str,
    component_columns: list[str],
    bins: int = 20,
    normalize_by_global: bool = False,
) -> pd.DataFrame:
    edges = trimmed_bins(frame[property_column], bins=bins)
    work = frame[(frame[property_column] >= edges[0]) & (frame[property_column] <= edges[-1])].copy()
    work["bin"] = pd.cut(work[property_column], bins=edges, include_lowest=True)
    grouped = work.groupby("bin", observed=False)[component_columns].sum()
    if normalize_by_global:
        global_fraction = frame[component_columns].sum()
        global_fraction = global_fraction / global_fraction.sum()
        grouped = grouped.div(global_fraction.replace(0, np.nan), axis=1)
    fractions = grouped.div(grouped.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    fractions.insert(0, "bin_midpoint", [interval.mid for interval in fractions.index])
    fractions.insert(1, "bin_left", [interval.left for interval in fractions.index])
    fractions.insert(2, "bin_right", [interval.right for interval in fractions.index])
    fractions.insert(3, "molecule_count", work.groupby("bin", observed=False).size().to_numpy())
    return fractions.reset_index(drop=True)


def plot_stacked_property_histograms(
    frame: pd.DataFrame,
    component_columns: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    name: str,
    title: str,
    normalize_by_global: bool = False,
) -> None:
    plot_colors = colors_by_element(
        component_columns,
        {component: RING_PRIMARY_ELEMENT.get(component, component.upper()) for component in component_columns},
    )
    fig, axes = plt.subplots(len(PROPERTY_COLUMNS), 1, figsize=(12, 13), sharex=False)
    all_tables = []
    for axis, property_column in zip(axes, PROPERTY_COLUMNS):
        table = property_stacked_prevalence(
            frame,
            property_column,
            component_columns,
            bins=22,
            normalize_by_global=normalize_by_global,
        )
        table.insert(0, "property", property_column)
        all_tables.append(table)

        x = table["bin_midpoint"].to_numpy()
        width = np.median(np.diff(x)) * 0.92 if len(x) > 1 else 0.1
        bottom = np.zeros(len(table))
        for component in component_columns:
            values = table[component].to_numpy()
            axis.bar(
                x,
                values,
                width=width,
                bottom=bottom,
                color=plot_colors.get(component, colors.get(component, "#999999")),
                label=labels.get(component, component),
                edgecolor="white",
                linewidth=0.25,
            )
            bottom += values
        axis.set_ylabel(property_column.upper())
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(ncols=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.38))
    fig.suptitle(title, y=0.995, fontsize=14)
    fig.text(0.5, 0.012, "Property value, trimmed to 1st-99th percentile", ha="center")
    fig.text(0.01, 0.5, "Fraction within property bin", va="center", rotation=90)
    pd.concat(all_tables, ignore_index=True).to_csv(TABLE_DIR / f"{name}.csv", index=False)
    savefig(name)


def plot_figure9_scatter(frame: pd.DataFrame) -> None:
    benzene_only = frame[frame["non_benzene_ring_count"] == 0]
    single_type = frame[frame["mixed_non_benzene_ring_types"] == 1]

    fig, axes = plt.subplots(2, 5, figsize=(17, 7.6), sharex=True, sharey=True)
    axes = axes.ravel()
    scatter_tables = []
    for axis, ring in zip(axes, NON_BENZENE_RINGS):
        subset = single_type[single_type[ring] > 0]
        if len(benzene_only) > 45000:
            base = benzene_only.sample(45000, random_state=42)
        else:
            base = benzene_only
        axis.scatter(base["homo"], base["lumo"], s=2, alpha=0.14, color="#bdbdbd", linewidths=0)
        axis.scatter(
            subset["homo"],
            subset["lumo"],
            s=3,
            alpha=0.22,
            color=MANUAL_COLORS.get(ring, RING_COLORS[ring]),
            linewidths=0,
        )
        axis.set_title(f"{RING_LABELS[ring]}\nn={len(subset):,}", fontsize=9)
        axis.grid(alpha=0.18)
        scatter_tables.append(
            {
                "ring_type": ring,
                "single_non_benzene_type_molecules": len(subset),
                "mean_homo": subset["homo"].mean(),
                "mean_lumo": subset["lumo"].mean(),
                "mean_gap": subset["gap"].mean(),
                "mean_aip": subset["aip"].mean(),
                "mean_aea": subset["aea"].mean(),
                "mean_nfod": subset["nfod"].mean(),
            }
        )
    for axis in axes[5:]:
        axis.set_xlabel("HOMO")
    for axis in axes[::5]:
        axis.set_ylabel("LUMO")
    fig.suptitle(
        "HOMO-LUMO Maps: Benzene-only Baseline vs One Non-benzene Building Block",
        y=1.02,
        fontsize=14,
    )
    pd.DataFrame(scatter_tables).to_csv(TABLE_DIR / "figure9a_single_ring_scatter_summary.csv", index=False)
    savefig("figure9a_homo_lumo_single_ring_types")


def plot_ring_count_histograms(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.4), sharey=True)
    axes = axes.ravel()
    rows = []
    for axis, ring in zip(axes, NON_BENZENE_RINGS):
        counts = frame[ring].value_counts().sort_index()
        counts = counts[counts.index > 0]
        for count, molecules in counts.items():
            rows.append({"ring_type": ring, "ring_count": int(count), "molecule_count": int(molecules)})
        axis.bar(
            counts.index.astype(int),
            counts.values,
            color=MANUAL_COLORS.get(ring, RING_COLORS[ring]),
            edgecolor="white",
        )
        axis.set_title(RING_LABELS[ring], fontsize=9)
        axis.set_xlabel("Count in molecule")
        axis.grid(axis="y", alpha=0.2)
    for axis in axes[::5]:
        axis.set_ylabel("Molecules")
    fig.suptitle("Ring Type Count Histograms", y=1.02, fontsize=14)
    pd.DataFrame(rows).to_csv(TABLE_DIR / "ring_type_count_histograms.csv", index=False)
    savefig("ring_type_count_histograms")


def plot_antiaromatic_property_distributions(frame: pd.DataFrame) -> None:
    categories = pd.cut(
        frame["antiaromatic_ring_count"],
        bins=[-0.1, 0.1, 1.1, 2.1, frame["antiaromatic_ring_count"].max() + 1],
        labels=["0", "1", "2", "3+"],
    )
    colors = {"0": "#bdbdbd", "1": "#4c72b0", "2": "#dd8452", "3+": "#c44e52"}
    fig, axes = plt.subplots(1, len(PROPERTY_COLUMNS), figsize=(17, 3.8), sharey=False)
    table_rows = []
    for axis, property_column in zip(axes, PROPERTY_COLUMNS):
        edges = trimmed_bins(frame[property_column], bins=80)
        for label in ["0", "1", "2", "3+"]:
            values = frame.loc[categories == label, property_column]
            values = values[(values >= edges[0]) & (values <= edges[-1])]
            counts, bin_edges = np.histogram(values, bins=edges, density=True)
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            axis.plot(centers, counts, color=colors[label], lw=1.8, label=f"{label} antiaromatic")
            table_rows.append(
                {
                    "property": property_column,
                    "antiaromatic_ring_count_group": label,
                    "molecules": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                }
            )
        axis.set_title(property_column.upper())
        axis.set_xlabel("Value")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Density")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Property Distributions by Number of Antiaromatic Rings", y=1.05, fontsize=14)
    pd.DataFrame(table_rows).to_csv(TABLE_DIR / "antiaromatic_property_distribution_summary.csv", index=False)
    savefig("figure8_like_antiaromatic_property_distributions")


def plot_ring_heteroatom_heatmaps(frame: pd.DataFrame) -> None:
    hetero_by_ring = []
    for ring in NON_BENZENE_RINGS:
        subset = frame[frame[ring] > 0]
        totals = subset[HETEROATOM_COLUMNS].sum()
        frac = totals / totals.sum()
        for atom, value in frac.items():
            hetero_by_ring.append({"ring_type": ring, "heteroatom": atom, "fraction": value})
    hetero_table = pd.DataFrame(hetero_by_ring)
    hetero_table.to_csv(TABLE_DIR / "heteroatom_fraction_by_ring_presence.csv", index=False)
    heatmap = hetero_table.pivot(index="ring_type", columns="heteroatom", values="fraction").reindex(NON_BENZENE_RINGS)

    fig, axis = plt.subplots(figsize=(7.5, 7.2))
    image = axis.imshow(heatmap.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(heatmap.columns)), [col.upper() for col in heatmap.columns])
    axis.set_yticks(np.arange(len(heatmap.index)), [RING_LABELS[item] for item in heatmap.index])
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            axis.text(j, i, f"{heatmap.iloc[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(image, ax=axis, label="Fraction of heteroatoms")
    axis.set_title("Heteroatom Composition in Molecules Containing Each Ring Type")
    savefig("heteroatom_fraction_by_ring_presence")


def plot_ring_property_heatmap(frame: pd.DataFrame) -> None:
    grouped = []
    for ring in NON_BENZENE_RINGS + ["benzene_only"]:
        if ring == "benzene_only":
            subset = frame[frame["non_benzene_ring_count"] == 0]
        else:
            subset = frame[frame[ring] > 0]
        row = {"ring_type": ring, "molecules": len(subset)}
        for prop in [*PROPERTY_COLUMNS, "nfod", "heteroatom_density", "heterocycle_fraction"]:
            row[prop] = subset[prop].mean()
        grouped.append(row)
    summary = pd.DataFrame(grouped)
    summary.to_csv(TABLE_DIR / "property_means_by_ring_presence.csv", index=False)

    props = [*PROPERTY_COLUMNS, "nfod", "heteroatom_density", "heterocycle_fraction"]
    matrix = summary.set_index("ring_type")[props]
    z = (matrix - frame[props].mean()) / frame[props].std(ddof=0)
    fig, axis = plt.subplots(figsize=(10.5, 7.8))
    image = axis.imshow(z.to_numpy(), cmap="coolwarm", vmin=-1.5, vmax=1.5, aspect="auto")
    axis.set_xticks(np.arange(len(props)), [prop.upper() for prop in props], rotation=35, ha="right")
    axis.set_yticks(
        np.arange(len(z.index)),
        [RING_LABELS.get(item, "Benzene only") for item in z.index],
    )
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            axis.text(j, i, f"{z.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8)
    plt.colorbar(image, ax=axis, label="Mean z-score vs full dataset")
    axis.set_title("Ring Presence Trends Across Electronic and Structural Properties")
    savefig("ring_property_mean_zscore_heatmap")


def plot_ring_cooccurrence(frame: pd.DataFrame) -> None:
    presence = (frame[NON_BENZENE_RINGS] > 0).astype(int)
    counts = presence.T @ presence
    counts.to_csv(TABLE_DIR / "ring_type_cooccurrence_counts.csv")
    denom = np.sqrt(np.outer(np.diag(counts), np.diag(counts)))
    normalized = counts.to_numpy() / np.where(denom == 0, np.nan, denom)
    fig, axis = plt.subplots(figsize=(9.5, 8.2))
    image = axis.imshow(normalized, cmap="magma", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(NON_BENZENE_RINGS)), [RING_LABELS[r] for r in NON_BENZENE_RINGS], rotation=45, ha="right")
    axis.set_yticks(np.arange(len(NON_BENZENE_RINGS)), [RING_LABELS[r] for r in NON_BENZENE_RINGS])
    plt.colorbar(image, ax=axis, label="Cosine-normalized co-occurrence")
    axis.set_title("Non-benzene Ring Type Co-occurrence")
    savefig("ring_type_cooccurrence_heatmap")


def build_indexed_variant_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame[["name", "representation", *PROPERTY_COLUMNS, "nfod"]].itertuples(index=False):
        counts: dict[str, int] = {}
        ring_types: dict[str, str] = {}
        elements: dict[str, str] = {}
        positions: dict[str, str] = {}
        for ring_type, element, position, variant in parse_representation_variants(row.representation):
            if ring_type == "benzene":
                continue
            counts[variant] = counts.get(variant, 0) + 1
            ring_types[variant] = ring_type
            elements[variant] = element
            positions[variant] = position
        for variant, count in counts.items():
            rows.append(
                {
                    "name": row.name,
                    "ring_type": ring_types[variant],
                    "element": elements[variant],
                    "positions": positions[variant],
                    "variant": variant,
                    "variant_label": indexed_variant_label(variant),
                    "count": count,
                    **{prop: getattr(row, prop) for prop in PROPERTY_COLUMNS},
                    "nfod": row.nfod,
                }
            )
    return pd.DataFrame(rows)


def plot_indexed_variant_scatter(frame: pd.DataFrame, variant_table: pd.DataFrame) -> None:
    benzene_only = frame[frame["non_benzene_ring_count"] == 0]
    base = benzene_only.sample(min(len(benzene_only), 45000), random_state=42)
    single_type = frame[frame["mixed_non_benzene_ring_types"] == 1]

    rows = []
    for ring_type in NON_BENZENE_RINGS:
        variants = (
            variant_table[variant_table["ring_type"] == ring_type]
            .groupby("variant")
            .agg(
                molecules=("name", "nunique"),
                variant_label=("variant_label", "first"),
                element=("element", "first"),
                positions=("positions", "first"),
                mean_homo=("homo", "mean"),
                mean_lumo=("lumo", "mean"),
                mean_gap=("gap", "mean"),
                mean_aip=("aip", "mean"),
                mean_aea=("aea", "mean"),
                mean_nfod=("nfod", "mean"),
            )
            .reset_index()
            .sort_values(["molecules", "variant"], ascending=[False, True])
        )
        if variants.empty:
            continue
        variant_colors = colors_by_element(
            variants["variant"].tolist(),
            dict(zip(variants["variant"], variants["element"])),
        )
        rows.append(variants.assign(ring_type=ring_type))

        ncols = min(4, len(variants))
        nrows = int(np.ceil(len(variants) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.8 * nrows), sharex=True, sharey=True)
        axes = np.asarray(axes).reshape(-1)
        for axis, (_, variant_row) in zip(axes, variants.iterrows()):
            names = set(variant_table[variant_table["variant"] == variant_row["variant"]]["name"])
            subset = single_type[single_type["name"].isin(names)]
            axis.scatter(base["homo"], base["lumo"], s=2, alpha=0.14, color="#bdbdbd", linewidths=0)
            axis.scatter(
                subset["homo"],
                subset["lumo"],
                s=4,
                alpha=0.25,
                color=variant_colors.get(
                    variant_row["variant"],
                    MANUAL_COLORS.get(ring_type, RING_COLORS.get(ring_type, "#777777")),
                ),
                linewidths=0,
            )
            axis.set_title(f"{variant_row['variant_label']}\nn={len(subset):,}", fontsize=9)
            axis.grid(alpha=0.18)
        for axis in axes[len(variants) :]:
            axis.axis("off")
        for axis in axes[-ncols:]:
            axis.set_xlabel("HOMO")
        for axis in axes[::ncols]:
            axis.set_ylabel("LUMO")
        fig.suptitle(
            f"HOMO-LUMO Maps by Indexed {RING_LABELS.get(ring_type, ring_type)} Variant",
            y=1.02,
            fontsize=13,
        )
        savefig(f"figure9_like_indexed_{ring_type}_homo_lumo")

    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            TABLE_DIR / "indexed_ring_variant_summary.csv",
            index=False,
        )


def plot_indexed_variant_histograms(frame: pd.DataFrame, variant_table: pd.DataFrame) -> None:
    manual_rank = {variant: index for index, variant in enumerate(MANUAL_COLOR_ORDER)}
    top_variants = (
        variant_table.groupby("variant")
        .agg(
            total_count=("count", "sum"),
            molecules=("name", "nunique"),
            variant_label=("variant_label", "first"),
            ring_type=("ring_type", "first"),
            element=("element", "first"),
        )
        .sort_values("total_count", ascending=False)
        .head(24)
        .reset_index()
    )
    top_variants["manual_order"] = top_variants["variant"].map(manual_rank)
    fallback_order = pd.Series(
        len(manual_rank) + np.arange(len(top_variants)),
        index=top_variants.index,
    )
    top_variants["manual_order"] = top_variants["manual_order"].fillna(fallback_order)
    top_variants = top_variants.sort_values(["manual_order", "total_count"], ascending=[True, False])
    variants = top_variants["variant"].tolist()
    labels = dict(zip(top_variants["variant"], top_variants["variant_label"]))
    colors = colors_by_element(variants, dict(zip(top_variants["variant"], top_variants["element"])))
    all_tables = []

    fig, axes = plt.subplots(len(PROPERTY_COLUMNS), 1, figsize=(13, 13.5), sharex=False)
    for axis, property_column in zip(axes, PROPERTY_COLUMNS):
        edges = trimmed_bins(frame[property_column], bins=22)
        work = variant_table[
            (variant_table[property_column] >= edges[0])
            & (variant_table[property_column] <= edges[-1])
            & (variant_table["variant"].isin(variants))
        ].copy()
        work["bin"] = pd.cut(work[property_column], bins=edges, include_lowest=True)
        grouped = work.groupby(["bin", "variant"], observed=False)["count"].sum().unstack(fill_value=0)
        grouped = grouped.reindex(columns=variants, fill_value=0)
        fractions = grouped.div(grouped.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        table = fractions.reset_index()
        table.insert(0, "property", property_column)
        table.insert(1, "bin_midpoint", [interval.mid for interval in fractions.index])
        all_tables.append(table)

        x = np.array([interval.mid for interval in fractions.index])
        width = np.median(np.diff(x)) * 0.92 if len(x) > 1 else 0.1
        bottom = np.zeros(len(fractions))
        for variant in variants:
            values = fractions[variant].to_numpy()
            axis.bar(
                x,
                values,
                width=width,
                bottom=bottom,
                color=colors[variant],
                edgecolor="white",
                linewidth=0.2,
                label=labels[variant],
            )
            bottom += values
        axis.set_ylabel(property_column.upper())
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.2)

    axes[0].legend(ncols=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.58))
    fig.suptitle("Indexed Building-block Variant Prevalence Across Property Ranges", y=0.995, fontsize=14)
    fig.text(0.5, 0.012, "Property value, trimmed to 1st-99th percentile", ha="center")
    fig.text(0.01, 0.5, "Fraction within property bin", va="center", rotation=90)
    pd.concat(all_tables, ignore_index=True).to_csv(
        TABLE_DIR / "figure9_like_indexed_variant_stacked_property_histograms.csv",
        index=False,
    )
    top_variants.to_csv(TABLE_DIR / "indexed_ring_variant_top24.csv", index=False)
    savefig("figure9_like_indexed_variant_stacked_property_histograms")


def write_summary_tables(frame: pd.DataFrame) -> None:
    ring_summary = []
    for ring in RING_COLUMNS:
        count_sum = frame[ring].sum()
        molecules = (frame[ring] > 0).sum()
        ring_summary.append(
            {
                "ring_type": ring,
                "total_ring_instances": int(count_sum),
                "molecules_with_ring": int(molecules),
                "fraction_of_molecules": molecules / len(frame),
                "mean_count_when_present": count_sum / molecules if molecules else 0,
            }
        )
    pd.DataFrame(ring_summary).sort_values("total_ring_instances", ascending=False).to_csv(
        TABLE_DIR / "ring_type_summary.csv",
        index=False,
    )

    atom_totals = frame[HETEROATOM_COLUMNS].sum().rename("total_atoms").reset_index()
    atom_totals = atom_totals.rename(columns={"index": "heteroatom"})
    atom_totals["fraction_of_heteroatoms"] = atom_totals["total_atoms"] / atom_totals["total_atoms"].sum()
    atom_totals.to_csv(TABLE_DIR / "heteroatom_summary.csv", index=False)

    frame[
        [
            "rings",
            "aromatic_rings",
            "heteroatoms",
            "heterocycles",
            "branch",
            "non_benzene_ring_count",
            "antiaromatic_ring_count",
            "heterocycle_fraction",
            "heteroatom_density",
            *PROPERTY_COLUMNS,
            "nfod",
        ]
    ].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T.to_csv(
        TABLE_DIR / "dataset_property_structural_summary.csv"
    )


def main() -> None:
    ensure_dirs()
    frame = load_data()
    write_summary_tables(frame)
    indexed_variant_table = build_indexed_variant_table(frame)
    indexed_variant_table.to_csv(TABLE_DIR / "indexed_ring_variant_molecule_counts.csv", index=False)

    plot_stacked_property_histograms(
        frame,
        FIGURE9B_RING_ORDER,
        RING_LABELS,
        RING_COLORS,
        "figure9b_ring_type_stacked_property_histograms",
        "Building Block Prevalence Across Electronic Property Ranges",
        normalize_by_global=False,
    )
    plot_stacked_property_histograms(
        frame,
        HETEROATOM_COLUMNS,
        {atom: atom.upper() for atom in HETEROATOM_COLUMNS},
        ATOM_COLORS,
        "figure7_like_heteroatom_enrichment_property_histograms",
        "Heteroatom Enrichment Across Electronic Property Ranges",
        normalize_by_global=True,
    )
    plot_figure9_scatter(frame)
    plot_antiaromatic_property_distributions(frame)
    plot_ring_count_histograms(frame)
    plot_ring_heteroatom_heatmaps(frame)
    plot_ring_property_heatmap(frame)
    plot_ring_cooccurrence(frame)
    plot_indexed_variant_scatter(frame, indexed_variant_table)
    plot_indexed_variant_histograms(frame, indexed_variant_table)

    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
