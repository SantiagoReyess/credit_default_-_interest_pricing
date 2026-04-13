"""
2_pricing_tasa.py
Carga el modelo exportado por 1_modelo_pd.py y construye una tasa de interes
individualizada para cada acreditado siguiendo el framework de Expected Loss:

    r_i = TasaBase + PrimaInflacion + PrimaRiesgo_i
          + PrimaLiquidez_i + CostosAdmin + MargenUtilidad_i

Inputs:
    modelo_pd.pkl   exportado por 1_modelo_pd.py
    home_loans.csv  dataset de creditos hipotecarios

Outputs:
    pricing_clientes.csv  tabla con PD y componentes de tasa por cliente
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from catboost import Pool
from scipy.stats import pearsonr, spearmanr


# 1. CARGAR MODELO Y DATOS

with open("modelo_pd.pkl", "rb") as f:
    meta = pickle.load(f)

model        = meta["model"]
cat_features = meta["cat_features"]
cat_idx      = meta["cat_idx"]
feat_names   = meta["feature_names"]
best_thr     = meta["best_threshold"]

df = pd.read_csv("home_loans.csv")
X  = df[feat_names]


# 2. SCORING — PD por cliente

PD = model.predict_proba(Pool(X, cat_features=cat_idx))[:, 1]

print(f"PD  media: {PD.mean():.4f}  mediana: {np.median(PD):.4f}")
print(f"    p10:   {np.percentile(PD, 10):.4f}  p90: {np.percentile(PD, 90):.4f}")


# 3. PARAMETROS FIJOS

TASA_BASE       = 0.0675  # TIIE 
PRIMA_INFLACION = 0.0175  # expectativa inflacion 
LGD             = 0.28    # Loss Given Default hipotecario con colateral inmobiliario
COSTOS_ADMIN    = 0.0075  # amortizado sobre saldo promedio SHF


# 4. COMPONENTES VARIABLES

# Prima de Riesgo: EL = PD x LGD + overlays de politica crediticia
prima_riesgo_base = PD * LGD
overlay = np.zeros(len(df))
overlay += np.where(df["EmploymentType"] == "Unemployed",    0.0015, 0.0)
overlay += np.where(df["EmploymentType"] == "Self-employed", 0.0010, 0.0)
overlay += np.where(df["HasMortgage"]    == "Yes",           0.0010, 0.0)  # deuda hipotecaria previa
overlay += np.where(df["HasCoSigner"]    == "Yes",          -0.0010, 0.0)  # mitigacion por coacreditado
overlay += np.where(df["DTIRatio"]       >  0.40,            0.0010, 0.0)  # estres financiero
prima_riesgo = np.clip(prima_riesgo_base + overlay, 0.0050, 0.0250)

# Prima de Liquidez: interpolacion lineal sobre LoanTerm
term_min, term_max = df["LoanTerm"].min(), df["LoanTerm"].max()
prima_liquidez = 0.0050 + (
    (df["LoanTerm"] - term_min) / (term_max - term_min)
) * (0.0100 - 0.0050)

# Margen de Utilidad: escala inversa logaritmica sobre LoanAmount
log_amt = np.log(df["LoanAmount"])
margen_util = 0.0200 - (
    (log_amt - log_amt.min()) / (log_amt.max() - log_amt.min())
) * (0.0200 - 0.0100)

# Tasa final
SEG_ORDER = ["Bajo (<5%)", "Medio (5-15%)", "Alto (15-30%)", "Muy Alto (>30%)"]

df["PD"]             = PD
df["PrimaRiesgo"]    = prima_riesgo
df["PrimaLiquidez"]  = prima_liquidez
df["MargenUtilidad"] = margen_util
df["TasaConstruida"] = (
    TASA_BASE + PRIMA_INFLACION + prima_riesgo + prima_liquidez
    + COSTOS_ADMIN + margen_util
)
df["SegmentoRiesgo"] = pd.cut(
    df["PD"],
    bins=[0, 0.05, 0.15, 0.30, 1.0],
    labels=SEG_ORDER,
)


# 5. RESUMEN

r_p, _ = pearsonr(df["TasaConstruida"], df["InterestRate"])
r_s, _ = spearmanr(df["TasaConstruida"], df["InterestRate"])

print()
print(f"  Tasa Base (fija)              {TASA_BASE*100:>6.3f}%")
print(f"  Prima Inflacion (fija)        {PRIMA_INFLACION*100:>6.3f}%")
print(f"  Prima Riesgo (media)          {prima_riesgo.mean()*100:>6.3f}%")
print(f"  Prima Liquidez (media)        {prima_liquidez.mean()*100:>6.3f}%")
print(f"  Costos Admin (fijos)          {COSTOS_ADMIN*100:>6.3f}%")
print(f"  Margen Utilidad (media)       {margen_util.mean()*100:>6.3f}%")
print(f"  Tasa Construida media         {df['TasaConstruida'].mean()*100:>6.3f}%")
print(f"  Tasa Construida rango         "
      f"{df['TasaConstruida'].min()*100:.3f}% - {df['TasaConstruida'].max()*100:.3f}%")
print(f"  InterestRate dataset media    {df['InterestRate'].mean():>6.3f}%")
print(f"  Pearson  (construida/dataset) {r_p:>7.4f}")
print(f"  Spearman (construida/dataset) {r_s:>7.4f}")
print()
print(df["SegmentoRiesgo"].value_counts().sort_index().to_string())
print()


# 6. EXPORTAR CSV

out_cols = [
    "Age", "Income", "LoanAmount", "CreditScore", "LoanTerm", "DTIRatio",
    "EmploymentType", "HasMortgage", "HasCoSigner",
    "PD", "SegmentoRiesgo", "PrimaRiesgo", "PrimaLiquidez",
    "MargenUtilidad", "TasaConstruida", "InterestRate", "Default",
]
df_out = df[out_cols].copy()
for col in ["PD", "PrimaRiesgo", "PrimaLiquidez", "MargenUtilidad", "TasaConstruida"]:
    df_out[col] = (df_out[col] * 100).round(4)
df_out.rename(columns={
    "PD":             "PD_%",
    "PrimaRiesgo":    "PrimaRiesgo_%",
    "PrimaLiquidez":  "PrimaLiquidez_%",
    "MargenUtilidad": "MargenUtilidad_%",
    "TasaConstruida": "TasaConstruida_%",
}, inplace=True)
df_out.to_csv("pricing_clientes.csv", index=False)

print("pricing_clientes.csv exportado")