"""Shared feature lists for PAS extraction and RF training."""

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

# PAS columns identical to a paper LALAS column (verified on COMPAS-2 sample).
PAS_EXACT_DUPLICATE_OF_PAPER = {
    "pas_total_rings": "paper_lalas_number_of_rings",
    "pas_branch_count": "paper_lalas_number_of_branching_points",
    "pas_l_fraction": "paper_lalas_l_ratio",
    "pas_longest_l_run": "paper_lalas_longest_l",
    "pas_longest_a_run": "paper_lalas_longest_a_case_insensitive",
}

PAS_COLUMNS_SUPERSEDED_BY_PAPER = frozenset(PAS_EXACT_DUPLICATE_OF_PAPER)
