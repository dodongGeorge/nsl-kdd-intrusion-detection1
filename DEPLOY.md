# Streamlit deployment checklist

1. Push this project folder to GitHub, including `app.py`, `common.py`, `requirements.txt`, `nsl_kdd_random_forest.joblib`, `results.json`, `sample_connections.csv`, and the complete `figures/` directory.
2. In Streamlit Community Cloud, select the correct repository and branch.
3. Set the app's main file path to `app.py` if the files are at repository root, or `nsl-kdd-ids/app.py` if the folder itself is inside the repository.
4. Deploy and inspect **Manage app → Logs** if startup fails.
5. Do not delete or rename `figures/per_class_importance.csv`; the Feature Importance tab reads it. The app now handles a missing file gracefully, but the chart requires it.
6. The included model is already trained. Running `python train_model.py` is optional and will regenerate model artifacts, results, and figures.

## Quick local smoke test

```bash
python -m pip install -r requirements.txt
python -m py_compile app.py common.py train_model.py
streamlit run app.py
```
