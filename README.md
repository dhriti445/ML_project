# ECG Arrhythmia Classification – CART and Random Forest

UE24CS352A Machine Learning mini-project · 
Team: Dhriti Kamani - PES1UG24AM085, Dhruv R - PES1UG24AM086


## What it does
Predicts the heart-condition class (UCI labels 1–16; class 1 = normal; 13 classes occur) of a patient from 279 ECG/patient features, using the UCI Arrhythmia dataset (452 patients). Based on the Stanford paper *Detecting Heart Abnormality using ECG with CART* (Zhou, Garcia, Rajbhandary), which used R/rpart and reports ~80 % accuracy (its own protocol – **not our result**).

Two models are compared: a pruned **CART** decision tree (the paper's method, interpretable) and a **Random Forest** (many trees voting).

## Files
| File | Purpose |
|---|---|
| `main.py` | the whole experiment (load → split → preprocess → train → evaluate → plots) |
| `requirements.txt` | packages |
| `arrhythmia.data` | **you add this** (UCI: https://archive.ics.uci.edu/dataset/5/arrhythmia), next to `main.py` or in a `data/` folder |

## Run (Windows PowerShell)
```powershell
cd path\to\ecg-cart-simple
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
python main.py                        # ~1 minute
```
Creates `results.txt`, `confusion_matrix.png`, `feature_importance.png`, `decision_tree.png`.

## Method
1. **Load**: header-less CSV, `?` = missing; last column is the label and is never in the features.
2. **Split**: 70 % train / 30 % test, stratified, seed 42. The test set is used once, only for reporting.
3. **Preprocessing** (inside the model, learned from training rows only – no leakage): drop features with > 50 % missing in the training rows (this removes attribute 14, 83.2 % missing, as the paper did), then fill other gaps with the training median.
4. **CART**: grid search over criterion, `max_depth`, `min_samples_leaf` and `ccp_alpha` (cost-complexity **pruning**) with 5-fold cross-validation on the training rows.
5. **Random Forest**: 300 trees, default settings. Chosen over other candidates using training-set cross-validation only.
6. **Evaluation**: test-set accuracy, balanced accuracy, macro-F1, and 10-fold cross-validation accuracy on all 452 rows (3 repeats) – the same kind of CV the paper reports.

## Results (actual run, seed 42)
| Model | Test accuracy | Test balanced acc. | Test macro-F1 | 10-fold CV accuracy |
|---|---|---|---|---|
| CART (pruned) | 0.632 | 0.261 | 0.237 | 0.669 |
| Random Forest | **0.728** | 0.360 | 0.369 | **0.739** |

Always predicting "normal" would score 54.2 %. **Neither model reaches the paper's ~80 %.** The paper used R/rpart, other imputation/pruning and its own evaluation settings, so its number is not directly comparable and this is not an exact reproduction. The test set has only 136 patients and some classes have 1–4, so per-class results are noisy; macro-F1 (0.24–0.37) shows rare classes are mostly missed.

## Limitations
Small dataset, rare classes, one split, simple imputation, feature importance is not causal, precomputed features (no raw ECG), historical single-source data, no external validation.

## Reference
Zhou, Garcia, Rajbhandary, *Detecting Heart Abnormality using ECG with CART*, Stanford. UCI Machine Learning Repository, Arrhythmia data set.

## Publish to a private GitHub repo
```powershell
git init
git add main.py requirements.txt README.md      # do NOT add arrhythmia.data
git commit -m "ECG arrhythmia classification"
git branch -M main
git remote add origin https://github.com/<you>/ecg-cart-simple.git
git push -u origin main
```
#
