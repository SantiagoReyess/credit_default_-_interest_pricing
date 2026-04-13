"""
1_modelo_pd.py
Entrena CatBoostClassifier sobre el dataset de creditos hipotecarios,
selecciona hiperparametros con un grid curado y exporta el modelo como pickle.

Inputs:
    home_loans.csv

Outputs:
    modelo_pd.pkl       modelo entrenado listo para scoring
    metricas_modelo.csv metricas de evaluacion del test set
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from scipy.stats import ks_2samp


# 1. DATOS

df = pd.read_csv("home_loans.csv")
print(f"Registros: {len(df):,}  |  Default rate: {df['Default'].mean():.3f}")

DROP = [
    "Default",
    "ScoreBin",     
    "LoanPurpose",   
    "InterestRate",  
]
cat_features = [
    "Education", "EmploymentType", "MaritalStatus",
    "HasMortgage", "HasDependents", "HasCoSigner",
]

X = df.drop(columns=DROP)
y = df["Default"]
cat_idx = [X.columns.get_loc(c) for c in cat_features]

print(f"Features: {list(X.columns)}")


# 2. SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = neg / pos

print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"Scale pos weight: {spw:.2f}  (neg={neg:,}, pos={pos:,})")


# 3. SELECCION DE HIPERPARAMETROS

GRID = [
    {"depth": 4, "learning_rate": 0.10, "l2_leaf_reg": 5},
    {"depth": 5, "learning_rate": 0.08, "l2_leaf_reg": 5},
    {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3},
    {"depth": 6, "learning_rate": 0.08, "l2_leaf_reg": 7},
]

X_tr_g, X_val_g, y_tr_g, y_val_g = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
)

print("\nSeleccionando hiperparametros...")
best_score, best = -1, GRID[0]

for params in GRID:
    m = CatBoostClassifier(
        **params,
        iterations=800,
        scale_pos_weight=spw,
        loss_function="Logloss",
        eval_metric="AUC",
        cat_features=cat_idx,
        early_stopping_rounds=40,
        random_seed=42,
        verbose=False,
    )
    m.fit(
        Pool(X_tr_g, y_tr_g, cat_features=cat_idx),
        eval_set=Pool(X_val_g, y_val_g, cat_features=cat_idx),
    )
    score = roc_auc_score(y_val_g, m.predict_proba(X_val_g)[:, 1])
    print(f"  depth={params['depth']}  lr={params['learning_rate']}"
          f"  l2={params['l2_leaf_reg']}  AUC={score:.4f}")
    if score > best_score:
        best_score, best = score, params

print(f"\nMejores parametros: {best}  (AUC val={best_score:.4f})")


# 4. MODELO FINAL

model = CatBoostClassifier(
    **best,
    iterations=800,
    scale_pos_weight=spw,
    loss_function="Logloss",
    eval_metric="AUC",
    cat_features=cat_idx,
    early_stopping_rounds=40,
    random_seed=42,
    verbose=100,
)
model.fit(
    Pool(X_train, y_train, cat_features=cat_idx),
    eval_set=Pool(X_test, y_test, cat_features=cat_idx),
)


# 5. EVALUACION

prob_test = model.predict_proba(X_test)[:, 1]

prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, prob_test)
f1_arr   = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
best_thr = float(thr_arr[np.argmax(f1_arr)])
y_pred   = (prob_test >= best_thr).astype(int)

auc = roc_auc_score(y_test, prob_test)
f1  = f1_score(y_test, y_pred)
ks  = ks_2samp(prob_test[y_test == 1], prob_test[y_test == 0]).statistic

print(f"\nAUC-ROC : {auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"KS Stat : {ks:.4f}")
print(f"Umbral  : {best_thr:.3f}")
print()
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

pd.DataFrame([{
    "AUC":    round(auc, 4),
    "F1":     round(f1, 4),
    "KS":     round(ks, 4),
    "Umbral": round(best_thr, 3),
}]).to_csv("metricas_modelo.csv", index=False)


# 6. EXPORTAR MODELO

with open("modelo_pd.pkl", "wb") as f:
    pickle.dump({
        "model":          model,
        "cat_features":   cat_features,
        "cat_idx":        cat_idx,
        "feature_names":  list(X.columns),
        "best_threshold": best_thr,
        "spw":            spw,
    }, f)

print("modelo_pd.pkl exportado")
print("metricas_modelo.csv exportado")