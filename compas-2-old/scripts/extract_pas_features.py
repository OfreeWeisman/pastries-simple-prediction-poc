from __future__ import annotations

import csv
import math
import re
import argparse
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data")
INPUT_FILES = [
    DATA_DIR / "compas-2x_pastries_features_sample.csv",
    DATA_DIR / "compas-2x_pastries_features_sample_train.csv",
    DATA_DIR / "compas-2x_pastries_features_sample_test.csv",
]

HIDDEN_RING_INPUT_FILES = [
    DATA_DIR / "compas-2x_pastries_features_sample.csv",
    DATA_DIR / "compas-2x_pastries_features_sample_train.csv",
    DATA_DIR / "compas-2x_pastries_features_sample_test.csv",
]

RING_ALIASES = {
    "Bn": "benzene",
    "Pd": "pyridine",
    "Pz": "pyrazine",
    "Th": "thiophene",
    "Fu": "furan",
    "Bz": "borole",
    "Cbd": "cyclobutadiene",
    "Pr": "pyrrole",
    "DhDBn": "dhdiborinine",
    "DBn": "14diborinine",
    "Brn": "borinine",
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

TARGET_COLUMNS = [
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


def parse_float(value: str) -> float:
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
    ring_code = stripped.split("[", 1)[0]
    return RING_ALIASES.get(ring_code, ring_code.lower()), is_branch


def parse_rings(ring_part: str) -> list[tuple[str, bool]]:
    rings = []
    for token in ring_part.split(","):
        ring_type, is_branch = clean_ring_token(token)
        if ring_type:
            rings.append((ring_type, is_branch))
    return rings


def hide_ring_types(rings: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
    return [("benzene", is_branch) for _ring_type, is_branch in rings]


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
    if not subsequence:
        return 0
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


def dipole_norm(value: str) -> float:
    parts = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value or "")
    vector = [float(part) for part in parts]
    return math.sqrt(sum(item * item for item in vector)) if vector else 0.0


def extract_features(row: dict[str, str], hide_ring_identity: bool = False) -> dict[str, float | str]:
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
    if hide_ring_identity:
        rings = hide_ring_types(rings)
    total_rings = len(rings)
    ring_counts = Counter(ring_type for ring_type, _ in rings)
    branch_ring_counts = Counter(ring_type for ring_type, is_branch in rings if is_branch)
    terminal_counts = Counter()
    middle_counts = Counter()
    branch_counts = Counter()

    for index, (ring_type, _is_branch) in enumerate(rings):
        if index == 0 or index == total_rings - 1:
            terminal_counts[ring_type] += 1
        else:
            middle_counts[ring_type] += 1
        if _is_branch:
            branch_counts[ring_type] += 1

    donor_count = sum(ring_counts[ring_type] for ring_type in DONOR_RINGS)
    acceptor_count = sum(ring_counts[ring_type] for ring_type in ACCEPTOR_RINGS)
    branch_count = topology.count("(")

    def class_position_count(position_counts: Counter[str], ring_class: set[str]) -> int:
        return sum(position_counts[ring_type] for ring_type in ring_class)

    def non_benzene_count(position_counts: Counter[str]) -> int:
        return sum(count for ring_type, count in position_counts.items() if ring_type != "benzene")

    features: dict[str, float | str] = {
        "name": row["name"],
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
        "pas_donor_ring_count": donor_count,
        "pas_acceptor_ring_count": acceptor_count,
        "pas_donor_acceptor_balance": donor_count - acceptor_count,
        "pas_donor_acceptor_ratio": donor_count / (acceptor_count + 1),
        "pas_donor_fraction": donor_count / total_rings if total_rings else 0.0,
        "pas_acceptor_fraction": acceptor_count / total_rings if total_rings else 0.0,
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
        "heterocycle_fraction": parse_float(row["heterocycles"]) / parse_float(row["rings"]) if parse_float(row["rings"]) else 0.0,
        "heteroatom_density": parse_float(row["heteroatoms"]) / parse_float(row["atoms"]) if parse_float(row["atoms"]) else 0.0,
        "aromatic_ring_fraction": parse_float(row["aromatic_rings"]) / parse_float(row["rings"]) if parse_float(row["rings"]) else 0.0,
    }

    for column in SOURCE_NUMERIC_FEATURES:
        features[column] = parse_float(row[column])

    if hide_ring_identity:
        features["heterocycle_fraction"] = 0.0
        features["heteroatom_density"] = 0.0
        features["aromatic_ring_fraction"] = 1.0 if total_rings else 0.0
        features["aromatic_rings"] = total_rings
        features["heteroatoms"] = 0.0
        features["heterocycles"] = 0.0
        features["cyclobutadiene"] = 0.0
        features["pyrrole"] = 0.0
        features["borole"] = 0.0
        features["furan"] = 0.0
        features["thiophene"] = 0.0
        features["dhdiborinine"] = 0.0
        features["14diborinine"] = 0.0
        features["pyrazine"] = 0.0
        features["pyridine"] = 0.0
        features["borinine"] = 0.0
        features["benzene"] = float(total_rings)
        features["b"] = 0.0
        features["s"] = 0.0
        features["o"] = 0.0
        features["n"] = 0.0

    for ring_type in RING_TYPES:
        count = ring_counts[ring_type]
        features[f"pas_{ring_type}_count"] = count
        features[f"pas_{ring_type}_fraction"] = count / total_rings if total_rings else 0.0
        features[f"pas_{ring_type}_terminal_count"] = terminal_counts[ring_type]
        features[f"pas_{ring_type}_middle_count"] = middle_counts[ring_type]
        features[f"pas_{ring_type}_branch_count"] = branch_ring_counts[ring_type]

    for target in TARGET_COLUMNS:
        if target == "dipole_norm":
            features[target] = dipole_norm(row.get("dipole", ""))
        else:
            features[target] = parse_float(row.get(target, ""))

    return features


def output_path(input_path: Path, suffix: str = "_feature_vectors") -> Path:
    return input_path.with_name(input_path.stem + f"{suffix}.csv")


def process_file(input_path: Path, hide_ring_identity: bool = False) -> None:
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [extract_features(row, hide_ring_identity=hide_ring_identity) for row in reader]

    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    suffix = "_hidden_ring_types_feature_vectors" if hide_ring_identity else "_feature_vectors"
    path = output_path(input_path, suffix=suffix)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PAS feature vectors.")
    parser.add_argument(
        "--hide-ring-types",
        action="store_true",
        help="Convert every parsed ring token to benzene and neutralize ring identity columns.",
    )
    args = parser.parse_args()

    for input_file in INPUT_FILES:
        process_file(input_file, hide_ring_identity=args.hide_ring_types)


if __name__ == "__main__":
    main()
