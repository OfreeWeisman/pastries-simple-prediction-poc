# COMPAS-2 Pastries Data Trend Analysis

Generated from `data/compas-2x_pastries_features.csv`.

## Article-style figures

- `figures/figure9a_homo_lumo_single_ring_types.png`: Figure 9A-style HOMO/LUMO scatter maps. Gray points are benzene-only molecules; colored points contain benzene plus one non-benzene building-block type.
- `figures/figure9b_ring_type_stacked_property_histograms.png`: Figure 9B-style stacked histograms of non-benzene building-block prevalence across HOMO, LUMO, Gap, AIP, and AEA bins.
- `figures/figure7_like_heteroatom_enrichment_property_histograms.png`: Figure 7-style heteroatom enrichment across the same property bins, normalized against global heteroatom prevalence.
- `figures/figure8_like_antiaromatic_property_distributions.png`: Figure 8-style property distributions grouped by the number of antiaromatic rings.

## Additional diagnostics

- `figures/ring_type_count_histograms.png`: Histograms of how often each ring type appears per molecule.
- `figures/ring_property_mean_zscore_heatmap.png`: Mean property shifts for molecules containing each ring type.
- `figures/heteroatom_fraction_by_ring_presence.png`: Higher-resolution heteroatom composition for molecules containing each ring type.
- `figures/ring_type_cooccurrence_heatmap.png`: Co-occurrence of non-benzene ring types.
- `figures/figure9_like_indexed_*_homo_lumo.png`: Figure 9A-like HOMO/LUMO maps split by indexed heteroatom variants, for example `Pyridine (Pd) [N:1]` vs `Pyridine (Pd) [N:4]`.
- `figures/figure9_like_indexed_variant_stacked_property_histograms.png`: Figure 9B-like stacked property histograms split by the most common indexed ring variants.

Each figure is also saved as SVG. The matching numeric data is in `tables/`.

## Color convention

The plots use grouped color families: B-containing rings are orange/red shades, N-containing rings are blue shades, O-containing rings are green shades, S-containing rings are purple shades, and carbon-only rings are gray. Indexed variants within the same heteroatom family use different shades of that same family color.

## Ring code mapping

The raw `representation` column uses short ring codes. The indexed plots parse these codes directly:

| Code | Ring type |
|---|---|
| `Bn` | Benzene (Bn) |
| `Cbd` | Cyclobutadiene (Cbd) |
| `Py` | Pyrrole (Py) |
| `Bl` | Borole (Bl) |
| `Fu` | Furan (Fu) |
| `Th` | Thiophene (Th) |
| `DhDBn` | Dhydrodiborinine (DhDBn) |
| `DBn` | DiBorinine (DBn) |
| `Pz` | Pyrazine (Pz) |
| `Pd` | Pyridine (Pd) |
| `Bz` | Borazine (Bz) |

## Main trends

- Benzene (Bn) appears in almost every molecule, so the non-benzene ring plots exclude Benzene (Bn) from the stacked composition histograms.
- Borole (Bl) and Dhydrodiborinine (DhDBn) are shifted toward lower LUMO and lower gap, with higher NFOD.
- DiBorinine (DBn) and Borazine (Bz) shift HOMO upward compared with the Benzene (Bn)-only baseline.
- Furan (Fu) and Thiophene (Th) follow a narrower HOMO/LUMO band closer to the Benzene (Bn)-only baseline.
- Pyrazine (Pz) and Pyridine (Pd) broaden the HOMO/LUMO space and shift toward lower LUMO.

## Suggested next analyses

- Split the Figure 9 histograms into single-ring-only molecules versus mixed-heterocycle molecules to isolate clean ring effects from mixing effects.
- Add 2D density contours to the HOMO/LUMO maps so dense regions are easier to compare quantitatively.
- Fit simple linear or random-forest models per property using ring counts only, then compare residuals against mixed-ring counts and topology descriptors.
- Build pairwise ring-interaction tables: expected property shift from ring A plus ring B versus observed shift in molecules containing both.
