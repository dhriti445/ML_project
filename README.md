# Detecting Heart Abnormality Using ECG Features with CART

UE24CS352A – Machine Learning mini-project. Team: Dhriti Kamani - PES1UG24AM085, Dhruv R - PES1UG24AM086

## Status

| Item | Status |
|---|---|
| Source code (`main.py`), executed notebook, README | ✅ done |
| Real UCI dataset | ✅ used (452 × 279; class counts and missing values match the UCI documentation; `arrhythmia.data` is not redistributed – add your own copy) |
| Real experiment | ✅ executed once (seed 42); outputs in `results.txt` and the three PNG figures |

## Individual contributions

The assignment requires honest documentation of each member's work. This table is a template: nothing below has been filled in on your behalf.

| Member (name / ID) | Contribution |
|---|---|
| PES1UG24AM085 - Dhriti Kamani | `data validation, preprocessing` |
| PES1UG24AM086 - Dhruv R | `tuning/pruning, notebook` |

## Overview

Multiclass classification of patients in the UCI Arrhythmia data (452 patients, 279 features, class label 1–16; class 1 = normal) with a CART decision tree (`sklearn.tree.DecisionTreeClassifier`), following Zhou, Garcia & Rajbhandary, *Detecting Heart Abnormality using ECG with CART* (Stanford; R/rpart; ~80 % accuracy reported – their result, not ours). A Random Forest (`RandomForestClassifier`) is added as a comparison model.

Objectives: validate the data; leak-free preprocessing; tune and prune CART on training data only; compare it with a Random Forest; evaluate on a held-out test set and by 10-fold CV; interpret features; compare honestly with the paper; state limitations.

## Dataset

Official source: https://archive.ics.uci.edu/dataset/5/arrhythmia → `arrhythmia.data` (comma-separated, no header, `?` = missing, 280 columns: 279 predictors + class label last). Put the file next to `main.py` (or in a `data/` folder). Features are named `f1`…`f279` (UCI attribute order; e.g. `f15` = heart rate). The model uses precomputed features, not raw waveforms.

## Technologies

Python 3.11+ (tested 3.13), pandas, NumPy, scikit-learn, Matplotlib, Jupyter (for the notebook).

## Repository structure

```
ecg-cart-simple/
├── main.py                    the whole experiment (run this)
├── ECG_CART_Analysis.ipynb    same experiment, step by step (saved executed on the real data)
├── README.md  requirements.txt
├── data
      -arrhythmia.data            you add this (UCI file)
├── results.txt                metrics table + disclaimer (created by main.py)
└── output
      -confusion_matrix.png  feature_importance.png  decision_tree.png   (created by main.py)
```

## Installation (Windows PowerShell)

```
cd path\to\ecg-cart-simple
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
```
macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## Get the data

Download `arrhythmia.data` manually from the UCI link above and copy it next to `main.py`. `main.py` stops with a clear message if the file is missing or does not have shape (452, 280).

## Run

```
python main.py                                        # takes a short while (RF + 10-fold CV): load -> tune CART -> Random Forest -> evaluate -> figures
pip install jupyter                                   # only if you want the notebook
jupyter notebook ECG_CART_Analysis.ipynb              # Kernel -> Restart & Run All
```

## Methodology

- **Validate:** 452 rows × 280 columns, `?` read as missing, target = last column and removed from X.
- **Split:** stratified 70/30 (seed 42) → 316 train / 136 test. Test rows are not used for any tuning.
- **Preprocess** (inside one scikit-learn `Pipeline`, so it is learned from training rows only – and re-learned inside every CV fold): drop features with > 50 % missing training values (the paper's feature 14, `j_angle`, is 83.2 % missing; we decide from measured data, never position); median imputation for the rest. Difference from the paper: rpart-based handling → simple train-statistic imputation.
- **CART:** Gini/entropy splits. Tuning on training rows only: criterion {gini, entropy} × max_depth {4, 6, 8, None} × min_samples_leaf {1, 3, 5} × ccp_alpha {0, 0.005, 0.01, 0.02} = 96 combinations × 5-fold stratified CV, metric = accuracy. `ccp_alpha` is cost-complexity pruning; sklearn `ccp_alpha` and rpart `cp` are related but not interchangeable, so the paper's 0.0001 / 0.027 are not reused.
- **Random Forest:** 300 trees, default settings, seed 42.
- **Evaluate:** held-out test accuracy, balanced accuracy, macro-F1; plus 10-fold stratified CV accuracy (3 repeats) on all 452 rows – the paper's CV protocol.

## Experimental results

Single stratified 70/30 split (316 train / 136 test), seed 42. Selected CART: criterion gini, max_depth 6, min_samples_leaf 3, ccp_alpha 0.02 (8 leaves; training 5-fold CV 0.684); `j_angle` (f14) dropped.

| Model | Test accuracy | Test balanced acc. | Test macro-F1 | 10-fold CV accuracy (all rows) |
|---|---|---|---|---|
| CART (pruned) | 0.632 | 0.261 | 0.237 | 0.669 |
| Random Forest | 0.728 | 0.360 | 0.369 | 0.739 |

Always predicting class 1 (normal) scores 0.542. Random Forest class 1 recall is 0.97 (72 of 74), but classes 5, 7, 8, 14, 15 and 16 are never recovered on the test set, so macro-F1 is far lower than accuracy. The paper's ~80 % is **not** our result and is not reproduced: its protocol differs (R/rpart, other splits), and we did not tune toward that number. The CV column and the test columns use different protocols and are not directly comparable. With only 136 test rows and classes of 1–8 test cases, a single split is noisy; differences of a few points between models can be within noise.

## Reproducibility

Fixed seed (42) and deterministic split; results can differ slightly across scikit-learn/pandas versions. Tested with scikit-learn 1.9.1, pandas 3.0.5, NumPy 2.5.3, Matplotlib 3.11.2.

## Limitations

Small dataset; rare classes (1–8 test rows) → unstable metrics; single split; simple median imputation; feature importance is not causal; precomputed features, no waveforms; historical single-source data; no external validation; no automated tests in this version.

## Publishing to a private GitHub repository

```
git init
git status
echo arrhythmia.data >> .gitignore
echo .venv/ >> .gitignore
git add .
git status                                   # review the staged list (no .venv, no data)
git commit -m "Initial commit: ECG CART mini-project"
git branch -M main
# create an EMPTY PRIVATE repo on github.com first (do not tick "add README"), then:
git remote add origin https://github.com/<your-username>/ecg-cart-simple.git
git push -u origin main
git remote -v ; git log --oneline -1         # verify; then open the repo page and check "Private"
```
Share: GitHub → repo → Settings → Collaborators → Add people → add faculty and TAs (usernames/emails from the course). Confirm Settings → General shows Private.

## References

1. B. Zhou, G. Garcia Jr, P. Rajbhandary, *Detecting Heart Abnormality using ECG with CART*, Stanford (course report).
2. UCI Machine Learning Repository, Arrhythmia data set, https://archive.ics.uci.edu/dataset/5/arrhythmia
