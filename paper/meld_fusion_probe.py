import json
import numpy as np

LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
idx = {e: i for i, e in enumerate(LABELS)}

recs = [json.loads(l) for l in open("paper/meld_test_scores.jsonl", encoding="utf-8") if l.strip()]

def vec(r):
    return [r["voice"].get(e, 0.0) for e in LABELS] + [r["text_scores"].get(e, 0.0) for e in LABELS]

X = np.array([vec(r) for r in recs])
y = np.array([idx[r["gold"]] for r in recs])

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
except ImportError:
    raise SystemExit("need scikit-learn:  .venv/bin/pip install scikit-learn")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

def report(name, clf):
    pred = cross_val_predict(clf, X, y, cv=cv)
    print(f"{name:32s} WF1={f1_score(y, pred, average='weighted')*100:5.2f}  "
          f"MacroF1={f1_score(y, pred, average='macro')*100:5.2f}  "
          f"Acc={accuracy_score(y, pred)*100:5.2f}")

print("Reference: text-only Acc=49.85 WF1=51.55 | naive Acc=51.69 | heuristic Acc=53.75 WF1=51.58")
print("Oracle ceiling Acc=67.78\n")
print("Learned fusion (5-fold cross_val_predict on 14-dim voice+text scores):")
report("LogReg", LogisticRegression(max_iter=2000))
report("LogReg (balanced)", LogisticRegression(max_iter=2000, class_weight="balanced"))
report("HistGradientBoosting", HistGradientBoostingClassifier(random_state=0))