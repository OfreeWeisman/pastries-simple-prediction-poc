# COMPAS-2 Experiment

This folder contains the 10k PAS sample experiment comparing Random Forest prediction with and without ring-type information.

## Data

`data/` contains:

- 10k sample, 80/20 train/test split
- normal feature vectors with ring-type information
- hidden-ring feature vectors where all rings are treated as benzene
- single 80/20 RF ablation metrics/importances
- 5-fold RF ablation metrics/importances
- top-20 ranked RF features
- extracted text from the LALAS paper used for comparison

## Results

`results/` contains the 5-fold comparison plots:

- error percentage
- paper-style normalized MAE
- R2

## Scripts

`scripts/` contains the scripts used to regenerate the sample, features, RF ablation, and paper-feature comparison.

The main 5-fold experiment command is:

```bash
python scripts/run_rf_ringtype_ablation_cv.py --n-estimators 100
```
