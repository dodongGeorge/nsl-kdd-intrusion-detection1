# Network Intrusion Detection Using Machine Learning Models

**Group 14 · University of Mindanao · CCE105**

A Streamlit application that classifies NSL-KDD network connection records into five classes: `Normal`, `DoS`, `Probe`, `R2L`, and `U2R`. It uses a Random Forest model with engineered login/privilege features, per-class feature selection, training-only class balancing, and an R2L specialist classifier.

## Features

- Try predictions on sample NSL-KDD records.
- Upload a CSV containing the 41 standard NSL-KDD feature columns.
- View saved KDDTest+ evaluation metrics, confusion matrices, and before/after comparisons.
- Explore one-vs-rest feature importance for each class.
- Use a separate binary R2L-versus-rest Random Forest specialist. Its threshold is selected on a held-out split from the training data; KDDTest+ is not used to tune the threshold.

## Results in the included evaluation

The following are the saved benchmark results included with this project, evaluated on KDDTest+:

| Metric | Original model | Improved model |
|---|---:|---:|
| Accuracy | 73.97% | 82.01% |
| Macro-F1 | 48.71% | 69.59% |
| R2L recall | 0.35% | 39.62% |
| U2R recall | 4.48% | 26.87% |
| Normal false-positive rate | 2.70% | 3.46% |

These are benchmark results, not a guarantee of real-world performance. Higher recall for rare attacks comes with a modest increase in Normal traffic flagged as an attack. The test set contains attack types not present in training, which limits generalization, especially for R2L and U2R.

## Project files

- `app.py` — final Streamlit user interface and prediction flow, adapted from the final ZIP project while preserving this draft's project folder structure.
- `Group14_NSL_KDD_Random_Forest.ipynb` — original draft notebook retained for reference; the final app and saved model use the improved pipeline files (`common.py`, `train_model.py`, and `nsl_kdd_random_forest.joblib`).
- `common.py` — shared feature definitions, label mapping, feature engineering, and preprocessing.
- `train_model.py` — training, cross-validation grid search, threshold tuning, evaluation, and figure generation.
- `nsl_kdd_random_forest.joblib` — pre-trained main model and R2L specialist, with selected features and threshold.
- `results.json`, `results_summary.md` — saved benchmark metrics.
- `figures/` — feature importance, confusion matrices, and per-class metric charts.
- `data/` — NSL-KDD training and test files used by the training script.
- `sample_connections.csv` — sample records for the app and download button.

## Run locally

Use Python compatible with the package versions in `requirements.txt`.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The pre-trained artifact and evaluation outputs are included, so **you do not need to run training to launch the app**.

To retrain and regenerate the model, metrics, and figures, run from the project root:

```bash
python train_model.py
```

Training is compute-intensive and can take several minutes or longer depending on the computer. Retraining overwrites the included model and evaluation files.

## Deploy to Streamlit Community Cloud

1. Push the contents of this project folder to a GitHub repository.
2. In Streamlit Community Cloud, create or select an app connected to that repository.
3. Set the main file path to `app.py` (or `nsl-kdd-ids/app.py` if you committed the parent folder instead of its contents).
4. Deploy. Keep `app.py`, `common.py`, `requirements.txt`, `nsl_kdd_random_forest.joblib`, `results.json`, `sample_connections.csv`, and the `figures/` folder at the paths shown in this repository.
5. If deployment fails, inspect **Manage app → Logs** and verify the repository path and dependencies.

## Expected CSV format

Upload a CSV with a header row containing all 41 standard NSL-KDD feature names. The target label is not required for prediction. `sample_connections.csv` is available from the app as an example.

## Limitations

This is an offline benchmark project, not a live network sensor or an intrusion-prevention system. Predictions depend on NSL-KDD's feature set and distribution. The probability/confidence displayed by the app is not a guarantee that a prediction is correct. Use only datasets you are authorized to analyze.
