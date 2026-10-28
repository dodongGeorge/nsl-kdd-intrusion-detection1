# Streamlit redeployment checklist

1. Upload the **contents** of this project folder to the root of the GitHub repository connected to Streamlit.
2. `app.py`, `common.py`, `results.json`, `nsl_kdd_random_forest.joblib`, and `sample_connections.csv` must be in the same folder.
3. Commit the entire `figures/` folder, especially `figures/per_class_importance.csv`.
4. In Streamlit Cloud, set the Main file path to `app.py` if the repository root contains `app.py`. If you keep the project nested, set it to `nsl-kdd-intrusion-detection/app.py` and preserve the matching nested assets.
5. Commit and push; Streamlit Cloud should redeploy.

The app generates fallback confusion-matrix and per-class recall/F1 charts from `results.json` when the PNGs are absent. Class-specific feature importance uses the real CSV included in this package.
