import json
from pathlib import Path
import numpy as np, pandas as pd, joblib, streamlit as st
from common import CLASSES, CAT, FEATURES, R2L_BOOST, engineer, build_matrix

st.set_page_config(page_title="Network Intrusion Detection", page_icon="🛡️", layout="wide")

BASE = Path(__file__).resolve().parent   # resolve files relative to app.py


def find_file(rel):
    """Find a project asset at the expected path or in a nested deployment folder."""
    expected = BASE / rel
    if expected.is_file():
        return expected
    # This also handles repositories accidentally uploaded with an extra nested project folder.
    matches = list(BASE.glob("**/" + Path(rel).name))
    return matches[0] if matches else expected


def show_img(rel, caption=None):
    """Show a saved figure if present; otherwise report its expected path."""
    p = find_file(rel)
    if p.is_file():
        st.image(str(p), caption=caption)
        return True
    st.warning(f"Figure not found: {rel}. Commit this file to the GitHub repository to display it.")
    return False


def show_generated_confusion_matrices():
    """Build confusion matrices from metrics saved in results.json."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in zip(axes, ["before", "after"], ["Original model", "Improved model"]):
        cm = np.asarray(R[key]["cm"])
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
            ax=ax, values_format="d", xticks_rotation=45, colorbar=False
        )
        ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_generated_before_after():
    """Build per-class recall/F1 comparison from results.json."""
    import matplotlib.pyplot as plt
    x = np.arange(len(CLASSES))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, version, metric, offset in [
        ("Recall before", "before", "recall", -1.5 * width),
        ("Recall after", "after", "recall", -0.5 * width),
        ("F1 before", "before", "f1", 0.5 * width),
        ("F1 after", "after", "f1", 1.5 * width),
    ]:
        ax.bar(x + offset, [R[version][metric][c] for c in CLASSES], width, label=label)
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Per-class recall and F1 before and after")
    ax.legend(ncol=2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

@st.cache_resource
def load():
    model_path = find_file("nsl_kdd_random_forest.joblib")
    if not model_path.is_file():
        raise FileNotFoundError("nsl_kdd_random_forest.joblib is missing from the repository.")
    b = joblib.load(model_path)
    return b["model"], b["r2l_model"], float(b["r2l_threshold"]), b["selected_columns"], pd.Index(b["dummy_cols"])

model, r2l_model, r2l_threshold, selected, dummy_cols = load()
results_path = find_file("results.json")
if not results_path.is_file():
    st.error("results.json is missing. Add it to the same project folder as app.py.")
    st.stop()
with open(results_path, encoding="utf-8") as results_file:
    R = json.load(results_file)

def predict(df):
    df = df[FEATURES].copy()
    for c in CAT: df[c] = df[c].astype(str).str.strip().str.lower()
    X = build_matrix(engineer(df), dummy_cols).reindex(columns=selected, fill_value=0)
    p = model.predict_proba(X)
    # Retain the existing R2L probability boost, then let the specialist add
    # R2L overrides when its independently tuned threshold is met.
    p = p * np.where(np.asarray(model.classes_) == "R2L", R2L_BOOST, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    out = pd.DataFrame(p, columns=model.classes_)[CLASSES]
    base_pred = model.classes_[p.argmax(axis=1)]
    r2l_positive_index = list(r2l_model.classes_).index(1)
    r2l_prob = r2l_model.predict_proba(X)[:, r2l_positive_index]
    r2l_override = r2l_prob >= r2l_threshold
    pred = np.where(r2l_override, "R2L", base_pred)
    confidence = p.max(axis=1).copy()
    # For records overridden by the specialist, show its binary probability.
    confidence[r2l_override] = r2l_prob[r2l_override]
    out.insert(0, "Predicted class", pred)
    out.insert(1, "Confidence", confidence)
    out.insert(2, "R2L specialist probability", r2l_prob)
    return out

st.title("🛡️ Network Intrusion Detection")
st.caption("NSL-KDD Random Forest with engineered features, class balancing, an R2L probability boost, and a validation-tuned specialist")
t1, t2, t3, t4 = st.tabs(["Try a sample connection", "Upload connections (CSV)", "Model performance", "Features each class needs"])

with t1:
    sample_path = find_file("sample_connections.csv")
    if not sample_path.is_file():
        st.error("sample_connections.csv is missing from the repository.")
        st.stop()
    sample = pd.read_csv(sample_path)
    pick = st.selectbox("Actual class to browse", CLASSES, index=3)
    sub = sample[sample.actual_class == pick].reset_index(drop=True)
    i = st.selectbox("Sample connection", sub.index, format_func=lambda k: f"#{k}: {sub.protocol_type[k]} / {sub.service[k]} / {sub.flag[k]} (src_bytes={sub.src_bytes[k]})")
    row = sub.iloc[[i]]
    res = predict(row).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted class", res["Predicted class"]); c2.metric("Confidence", f"{res['Confidence']*100:.1f}%"); c3.metric("Actual class", pick)
    (st.success if res["Predicted class"] == pick else st.warning)("Correct prediction" if res["Predicted class"] == pick else "Incorrect prediction (R2L is still often missed; see Model performance)")
    st.bar_chart(res[CLASSES].astype(float))
    with st.expander("Show the 41 feature values"): st.dataframe(row[FEATURES].T.astype(str))

with t2:
    st.write("Upload a CSV with the 41 NSL-KDD feature columns (header row with the standard feature names).")
    sample_path = find_file("sample_connections.csv")
    if sample_path.is_file():
        st.download_button("Download a sample CSV", pd.read_csv(sample_path).to_csv(index=False), "sample_connections.csv")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        df = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing: st.error(f"Missing columns: {missing}")
        else:
            out = pd.concat([predict(df), df.reset_index(drop=True)], axis=1)
            st.dataframe(out.head(500)); st.bar_chart(out["Predicted class"].value_counts())

with t3:
    A, B = R["after"], R["before"]
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{A['accuracy']*100:.2f}%", f"{(A['accuracy']-B['accuracy'])*100:+.2f} pts")
    b.metric("Macro F1", f"{A['macro_f1']*100:.2f}%", f"{(A['macro_f1']-B['macro_f1'])*100:+.2f} pts")
    c.metric("Attacks detected", f"{A['attacks_detected']*100:.2f}%", f"{(A['attacks_detected']-B['attacks_detected'])*100:+.2f} pts")
    d.metric("Normal flagged as attack", f"{A['normal_flagged_as_attack']*100:.2f}%", f"{(A['normal_flagged_as_attack']-B['normal_flagged_as_attack'])*100:+.2f} pts", delta_color="inverse")
    st.table(pd.DataFrame({f"{m} {w}": [f"{R[w.lower()][k][cl]*100:.2f}%" for cl in CLASSES]
                           for k, m in [("recall", "Recall"), ("f1", "F1")] for w in ["Before", "After"]}, index=CLASSES))
    if not show_img("figures/confusion_matrices.png", caption="Confusion matrices on KDDTest+"):
        show_generated_confusion_matrices()
    if not show_img("figures/per_class_before_after.png", caption="Per-class recall and F1 before and after"):
        show_generated_before_after()
    bp = R["best_params"]
    st.info(f"Grid search (macro-F1, 3-fold): rare-class target {int(bp['rare_n'])}, large-class cap {int(bp['max_n'])}, min_samples_leaf {int(bp['min_samples_leaf'])}. "
            f"R2L probability boost: {R2L_BOOST:g}x; specialist threshold: {r2l_threshold:.2f}. R2L/U2R remain difficult, especially for attack types absent from training.")

with t4:
    importance_path = find_file("figures/per_class_importance.csv")
    if importance_path.is_file():
        imp = pd.read_csv(importance_path, index_col=0)
        missing_classes = [c for c in CLASSES if c not in imp.columns]
        if missing_classes:
            st.error(f"The feature-importance CSV is missing these columns: {missing_classes}.")
        else:
            cls = st.selectbox("Class", CLASSES, index=3)
            st.subheader(f"Most important features for {cls}")
            st.bar_chart(imp[cls].nlargest(12))
            st.dataframe(imp.loc[imp.max(axis=1).nlargest(25).index].round(4))
            show_img("figures/per_class_feature_importance.png", caption="Top features per class (one-vs-rest forests)")
    else:
        st.error("Missing figures/per_class_importance.csv. This file is included in the provided corrected project ZIP. Upload/commit the complete figures folder, not just app.py.")
        st.info("The app will not invent class-specific rankings because the paper's per-class analysis requires the original one-vs-rest importance data.")
