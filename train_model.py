"""Improved NSL-KDD Random Forest: richer preprocessing, per-class feature selection, SMOTE + under-sampling,
grid search scored by macro-F1, and a before/after comparison on KDDTest+.   Run: python train_model.py"""
import itertools, json, warnings
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from common import *

warnings.filterwarnings("ignore")
SEED = 42

# ---------------------------------------------------------------- 1. load + clean
train = clean(pd.read_csv("data/KDDTrain+.txt", names=RAW_COLUMNS))
test = clean(pd.read_csv("data/KDDTest+.txt", names=RAW_COLUMNS))
assert train.attack_class.notna().all() and test.attack_class.notna().all()
y_train, y_test = train.attack_class, test.attack_class
dummy_cols = pd.get_dummies(pd.concat([train[CAT], test[CAT]])).columns   # same one-hot columns for both files
unseen_mask = ~test.label.isin(set(train.label)) & (test.label != "normal")
print(f"train {train.shape}, test {test.shape}; test records of attack types unseen in training: {unseen_mask.sum()}")

# ---------------------------------------------------------------- 2. features
train_e, test_e = engineer(train), engineer(test)
numeric = [c for c in FEATURES if c not in CAT + CONSTANT]
corr = train[numeric].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
corr_drop = [c for c in upper.columns if (upper[c] > 0.95).any() and c not in PROTECTED]
drop = CONSTANT + corr_drop
print("dropped by correlation filter (protected features kept):", corr_drop)
X_tr_all, X_te_all = build_matrix(train_e, dummy_cols, drop), build_matrix(test_e, dummy_cols, drop)

# per-class importance: one one-vs-rest forest per class
imp = {}
for cls in CLASSES:
    rf = RandomForestClassifier(n_estimators=60, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    rf.fit(X_tr_all, (y_train == cls).astype(int))
    imp[cls] = pd.Series(rf.feature_importances_, index=X_tr_all.columns)
imp = pd.DataFrame(imp)
keep = set()
for cls in CLASSES:
    keep |= set(imp[cls].nlargest(15).index) | set(imp.index[imp[cls] >= 0.01])
selected = [c for c in X_tr_all.columns if c in keep]
print(f"selected {len(selected)} of {X_tr_all.shape[1]} columns")
imp.round(4).to_csv("figures/per_class_importance.csv")
X_tr, X_te = X_tr_all[selected], X_te_all[selected]

# ---------------------------------------------------------------- 2b. R2L specialist + validation-only threshold tuning
# Learn R2L vs all other classes separately. Threshold is chosen on a held-out
# split of TRAINING data; KDDTest+ is not used to tune it.
y_r2l_binary = (y_train == "R2L").astype(int)
X_sp_train, X_sp_val, y_sp_train, y_sp_val = train_test_split(
    X_tr, y_r2l_binary, test_size=0.20, random_state=SEED, stratify=y_r2l_binary
)
r2l_val_model = RandomForestClassifier(
    n_estimators=100, min_samples_leaf=2, class_weight="balanced_subsample",
    n_jobs=-1, random_state=SEED
)
r2l_val_model.fit(X_sp_train, y_sp_train)
r2l_val_prob = r2l_val_model.predict_proba(X_sp_val)[:, list(r2l_val_model.classes_).index(1)]
threshold_rows = []
for threshold in np.arange(0.05, 0.96, 0.01):
    val_pred = (r2l_val_prob >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_sp_val, val_pred, average="binary", zero_division=0
    )
    threshold_rows.append((float(f), float(threshold), float(p), float(r)))
# Maximise R2L F1 on validation data; if tied, prefer the higher threshold.
_, R2L_THRESHOLD, r2l_val_precision, r2l_val_recall = max(
    threshold_rows, key=lambda row: (row[0], row[1])
)
print(f"R2L specialist validation threshold={R2L_THRESHOLD:.2f}, "
      f"precision={r2l_val_precision:.3f}, recall={r2l_val_recall:.3f}")
# Refit specialist on all training records after selecting the threshold.
r2l_model = RandomForestClassifier(
    n_estimators=100, min_samples_leaf=2, class_weight="balanced_subsample",
    n_jobs=-1, random_state=SEED
).fit(X_tr, y_r2l_binary)

# ---------------------------------------------------------------- 3. class balancing
binary_cols = [c for c in selected if set(X_tr[c].unique()) <= {0.0, 1.0}]

def balance(X, y, rare_n, max_n):
    """Under-sample Normal/DoS to max_n, then SMOTE R2L/U2R up to rare_n (SMOTE runs on standardised data)."""
    counts = y.value_counts()
    under = {c: max_n for c in ["Normal", "DoS"] if counts.get(c, 0) > max_n}
    if under:
        X, y = RandomUnderSampler(sampling_strategy=under, random_state=SEED).fit_resample(X, y)
    counts = y.value_counts()
    over = {c: rare_n for c in ["R2L", "U2R"] if counts.get(c, 0) < rare_n}
    k = max(1, min(5, int(min(counts.get("R2L", 6), counts.get("U2R", 6))) - 1))
    sc = StandardScaler().fit(X)
    Xs, y2 = SMOTE(sampling_strategy=over, k_neighbors=k, random_state=SEED).fit_resample(sc.transform(X), y)
    X2 = pd.DataFrame(sc.inverse_transform(Xs), columns=X.columns)
    X2[binary_cols] = X2[binary_cols].round().clip(0, 1)
    return X2, y2

# ---------------------------------------------------------------- 4. grid search (macro-F1, stratified 3-fold)
rng = np.random.RandomState(SEED)
sub = np.concatenate([rng.choice(np.where(y_train == c)[0], min(25000, (y_train == c).sum()), replace=False)
                      if c in ("Normal", "DoS") else np.where(y_train == c)[0] for c in CLASSES])
Xs_, ys_ = X_tr.iloc[sub].reset_index(drop=True), y_train.iloc[sub].reset_index(drop=True)
grid = list(itertools.product([1000, 3000], [8000, 15000], [1, 2]))    # rare_n, max_n, min_samples_leaf
skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
rows = []
for rare_n, max_n, leaf in grid:
    scores, r2l, u2r = [], [], []
    for a, b in skf.split(Xs_, ys_):
        Xb, yb = balance(Xs_.iloc[a], ys_.iloc[a], rare_n, max_n)
        m = RandomForestClassifier(n_estimators=60, min_samples_leaf=leaf, class_weight="balanced",
                                   n_jobs=-1, random_state=SEED).fit(Xb, yb)
        p = m.predict(Xs_.iloc[b])
        scores.append(f1_score(ys_.iloc[b], p, average="macro"))
        f = f1_score(ys_.iloc[b], p, labels=CLASSES, average=None, zero_division=0)
        r2l.append(f[3]); u2r.append(f[4])
    rows.append(dict(rare_n=rare_n, max_n=max_n, min_samples_leaf=leaf, cv_macro_f1=np.mean(scores),
                     cv_f1_R2L=np.mean(r2l), cv_f1_U2R=np.mean(u2r)))
    print(rows[-1])
gs = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False).reset_index(drop=True)
best = gs.iloc[0]
gs.round(4).to_csv("figures/grid_search_results.csv", index=False)
print("best:", dict(best))

# ---------------------------------------------------------------- 5. final models: tuned vs original
def metrics(y_true, pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, labels=CLASSES, zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=CLASSES)
    act, prd = (y_true != "Normal").values, (pd.Series(pred, index=y_true.index) != "Normal").values
    return dict(accuracy=accuracy_score(y_true, pred), macro_f1=f1_score(y_true, pred, average="macro"),
                precision=dict(zip(CLASSES, p)), recall=dict(zip(CLASSES, r)), f1=dict(zip(CLASSES, f)),
                normal_flagged_as_attack=float((~act & prd).sum() / (~act).sum()),
                attacks_detected=float((act & prd).sum() / act.sum()), cm=cm.tolist())

# original configuration (before): all non-constant/non-correlated columns, 9 old features, global importance filter
tr_o, te_o = engineer(train, new=False), engineer(test, new=False)
Xo_tr_all, Xo_te_all = build_matrix(tr_o, dummy_cols, CONSTANT + corr_drop), build_matrix(te_o, dummy_cols, CONSTANT + corr_drop)
pre = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED).fit(Xo_tr_all, y_train)
sel_o = pd.Series(pre.feature_importances_, index=Xo_tr_all.columns)
sel_o = sel_o[sel_o >= 0.001].index.tolist()
rf_o = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED).fit(Xo_tr_all[sel_o], y_train)
pred_o = rf_o.predict(Xo_te_all[sel_o])

# tuned configuration (after)
Xb, yb = balance(X_tr, y_train, int(best.rare_n), int(best.max_n))
print("balanced training counts:", yb.value_counts().to_dict())
rf_t = RandomForestClassifier(n_estimators=200, min_samples_leaf=int(best.min_samples_leaf), class_weight="balanced",
                              n_jobs=-1, random_state=SEED).fit(Xb, yb)
# Apply the existing R2L probability boost, then the validation-tuned specialist.
proba_t = rf_t.predict_proba(X_te)
proba_t = proba_t * np.where(np.asarray(rf_t.classes_) == "R2L", R2L_BOOST, 1.0)
proba_t = proba_t / proba_t.sum(axis=1, keepdims=True)
pred_t = rf_t.classes_[proba_t.argmax(axis=1)]
r2l_test_prob = r2l_model.predict_proba(X_te)[:, list(r2l_model.classes_).index(1)]
pred_t = np.where(r2l_test_prob >= R2L_THRESHOLD, "R2L", pred_t)

before, after = metrics(y_test, pred_o), metrics(y_test, pred_t)
seen_unseen = {}
for name, pred in [("before", pred_o), ("after", pred_t)]:
    for cls in ["R2L", "U2R"]:
        for tag, mk in [("seen", ~unseen_mask), ("unseen", unseen_mask)]:
            m = ((y_test == cls) & mk).values
            seen_unseen[f"{name}_{cls}_{tag}"] = (int(m.sum()), float((pred[m] == cls).mean()) if m.sum() else None)

# ---------------------------------------------------------------- 6. outputs
joblib.dump(dict(model=rf_t, r2l_model=r2l_model, r2l_threshold=float(R2L_THRESHOLD), selected_columns=selected, dummy_cols=list(dummy_cols), classes=list(rf_t.classes_)),
            "nsl_kdd_random_forest.joblib", compress=3)
json.dump(dict(before=before, after=after, best_params={k: float(v) for k, v in best.items()},
               seen_unseen=seen_unseen, r2l_threshold=float(R2L_THRESHOLD), r2l_validation_precision=float(r2l_val_precision), r2l_validation_recall=float(r2l_val_recall), n_selected=len(selected), corr_dropped=corr_drop,
               balanced_counts={k: int(v) for k, v in yb.value_counts().items()}),
          open("results.json", "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, (t, r) in zip(ax, [("Before (original)", before), ("After (improved)", after)]):
    cm = np.array(r["cm"]); cmn = cm / cm.sum(1, keepdims=True)
    a.imshow(cmn, cmap="Blues", vmin=0, vmax=1); a.set_title(t)
    a.set_xticks(range(5)); a.set_yticks(range(5)); a.set_xticklabels(CLASSES); a.set_yticklabels(CLASSES)
    for i in range(5):
        for j in range(5): a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cmn[i, j] > .5 else "black", fontsize=8)
    a.set_xlabel("Predicted"); a.set_ylabel("Actual")
plt.tight_layout(); plt.savefig("figures/confusion_matrices.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); x = np.arange(5)
for a, key in zip(ax, ["recall", "f1"]):
    a.bar(x - .2, [before[key][c] for c in CLASSES], .4, label="Before", color="#999"); a.bar(x + .2, [after[key][c] for c in CLASSES], .4, label="After", color="#9b1c2e")
    a.set_xticks(x); a.set_xticklabels(CLASSES); a.set_title(key.upper() + " per class"); a.set_ylim(0, 1); a.legend()
plt.tight_layout(); plt.savefig("figures/per_class_before_after.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 5, figsize=(17, 4.5))
for a, cls in zip(ax, CLASSES):
    t = imp[cls].nlargest(8)[::-1]; a.barh(t.index, t.values, color="#9b1c2e"); a.set_title(cls); a.tick_params(labelsize=7)
plt.tight_layout(); plt.savefig("figures/per_class_feature_importance.png", dpi=130); plt.close()

def tbl(h, rows): return "| " + " | ".join(h) + " |\n|" + "---|" * len(h) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows)
pc = lambda v: f"{v*100:.2f}%"
md = ["# Results (KDDTest+)\n", "## Overall\n", tbl(["Metric", "Before", "After"], [
    ["Accuracy", pc(before["accuracy"]), pc(after["accuracy"])], ["Macro-F1", pc(before["macro_f1"]), pc(after["macro_f1"])],
    ["Attacks detected", pc(before["attacks_detected"]), pc(after["attacks_detected"])],
    ["Normal flagged as attack", pc(before["normal_flagged_as_attack"]), pc(after["normal_flagged_as_attack"])]]),
    "\n## Per class\n", tbl(["Class", "Recall before", "Recall after", "F1 before", "F1 after", "Precision before", "Precision after"],
        [[c, pc(before["recall"][c]), pc(after["recall"][c]), pc(before["f1"][c]), pc(after["f1"][c]),
          pc(before["precision"][c]), pc(after["precision"][c])] for c in CLASSES]),
    "\n## Best grid-search settings\n", tbl(["rare_n", "max_n", "min_samples_leaf", "CV macro-F1"],
        [[str(int(best.rare_n)), str(int(best.max_n)), str(int(best.min_samples_leaf)), f"{best.cv_macro_f1:.4f}"]]),
    "\n## R2L / U2R recall on attack types seen vs unseen in training\n", tbl(["Class", "Type", "Records", "Recall before", "Recall after"],
        [[c, t, str(seen_unseen[f'after_{c}_{t}'][0]), pc(seen_unseen[f'before_{c}_{t}'][1] or 0), pc(seen_unseen[f'after_{c}_{t}'][1] or 0)]
         for c in ["R2L", "U2R"] for t in ["seen", "unseen"]]),
    "\n## Top 5 features per class (one-vs-rest importance)\n", tbl(["Class", "Top features"],
        [[c, ", ".join(imp[c].nlargest(5).index)] for c in CLASSES])]
open("results_summary.md", "w").write("\n".join(md))
print("\n".join(md))
