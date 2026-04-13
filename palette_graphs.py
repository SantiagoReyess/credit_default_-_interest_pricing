"""
3_graficas_paleta.py — Gráficas con nueva paleta de colores
============================================================
Lee modelo_pd.pkl y home_loans.csv, reconstruye en memoria todo
lo necesario y genera las 17 gráficas sin reentrenar el modelo.

Inputs necesarios (mismo directorio):
    modelo_pd.pkl        — exportado por 1_modelo_pd.py
    home_loans.csv       — dataset de créditos hipotecarios
    pricing_clientes.csv — exportado por 2_pricing_tasa.py


Outputs → carpeta graficas_paleta/
"""

import pickle, shap, warnings, os
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)
from scipy.stats import ks_2samp, pearsonr, spearmanr

# 1. PALETA Y HELPERS
C = {
    "bg":      "#F5F2EF",
    "card":    "#FDFCFB",
    "blue":    "#808D94",
    "beige":   "#E6DDD3",
    "warm":    "#E6CFB8",
    "brown":   "#8F6F65",
    "green":   "#3F493D",
    "gray":    "#D6CBC6",
    "text":    "#2C2C2C",
    "subtext": "#6B6460",
    "white":   "#FDFCFB",
    "border":  "#D6CBC6",
}
SEG_COLORS = ["#3F493D", "#808D94", "#E6CFB8", "#8F6F65"]
SEG_ORDER  = ["Bajo (<5%)", "Medio (5-15%)", "Alto (15-30%)", "Muy Alto (>30%)"]
FONT       = dict(family="Georgia, 'Times New Roman', serif", color=C["text"])

def base_layout(title, xt="", yt="", w=820, h=560):
    AX = dict(
        color=C["subtext"], gridcolor=C["gray"], showgrid=True,
        zeroline=False, tickfont=dict(color=C["subtext"]),
        linecolor=C["gray"], linewidth=1,
    )
    return dict(
        title=dict(text=title,
                   font=dict(size=17, color=C["text"], family="Georgia, serif"),
                   x=0.05),
        paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
        font=FONT, width=w, height=h,
        xaxis={**AX, "title": xt},
        yaxis={**AX, "title": yt},
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"],
                    borderwidth=1, font=dict(color=C["text"])),
        margin=dict(t=75, b=65, l=75, r=45),
    )

os.makedirs("graficas_paleta", exist_ok=True)

def save(fig, name):
    fig.write_html(f"graficas_paleta/{name}.html", include_plotlyjs="cdn")
    print(f"  ✅ {name}.html")

# 2. CARGAR MODELO Y DATOS
print("Cargando modelo y datos…")
with open("modelo_pd.pkl", "rb") as f:
    meta = pickle.load(f)

model        = meta["model"]
cat_features = meta["cat_features"]
cat_idx      = meta["cat_idx"]
feat_names   = meta["feature_names"]
best_thr     = meta["best_threshold"]

df_raw = pd.read_csv("home_loans.csv")
X      = df_raw[feat_names]
y      = df_raw["Default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"  Test set: {len(X_test):,} registros")

# 3. PREDICCIONES Y MÉTRICAS
print("Calculando predicciones…")
pool_test  = Pool(X_test, cat_features=cat_idx)
prob_test  = model.predict_proba(pool_test)[:, 1]
y_pred     = (prob_test >= best_thr).astype(int)

auc = roc_auc_score(y_test, prob_test)
f1  = f1_score(y_test, y_pred)
ks  = ks_2samp(prob_test[y_test == 1], prob_test[y_test == 0]).statistic
cm  = confusion_matrix(y_test, y_pred)
fpr, tpr, _       = roc_curve(y_test, prob_test)
prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, prob_test)
f1_arr   = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
best_idx = int(np.argmax(f1_arr))

print(f"  AUC={auc:.4f}  KS={ks:.4f}  F1={f1:.4f}  Umbral={best_thr:.3f}")

# 4. SHAP VALUES
print("Calculando SHAP values (~30 seg)…")
shap_sample = X_test.sample(min(2000, len(X_test)), random_state=42)
explainer   = shap.TreeExplainer(model)
shap_vals   = explainer.shap_values(Pool(shap_sample, cat_features=cat_idx))


# 5. DATOS DE PRICING
print("Cargando pricing_clientes.csv…")
df_p = pd.read_csv("pricing_clientes.csv")
df_p["PD"]            = df_p["PD_%"] / 100
df_p["PrimaRiesgo"]   = df_p["PrimaRiesgo_%"] / 100
df_p["PrimaLiquidez"] = df_p["PrimaLiquidez_%"] / 100
df_p["MargenUtilidad"]= df_p["MargenUtilidad_%"] / 100
df_p["TasaConstruida"]= df_p["TasaConstruida_%"] / 100

TASA_BASE       = 0.0675
PRIMA_INFLACION = 0.0175
COSTOS_ADMIN    = 0.0075
FIXED           = TASA_BASE + PRIMA_INFLACION + COSTOS_ADMIN

seg_means = (df_p.groupby("SegmentoRiesgo", observed=True)
               [["PrimaRiesgo", "PrimaLiquidez", "MargenUtilidad"]].mean())
sample    = df_p.sample(min(4000, len(df_p)), random_state=42)
r_p, _    = pearsonr(df_p["TasaConstruida"], df_p["InterestRate"])
r_s, _    = spearmanr(df_p["TasaConstruida"], df_p["InterestRate"])

print(f"  Registros pricing: {len(df_p):,}\n")

# ROC 
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=fpr, y=tpr, mode="lines", name=f"CatBoost (AUC = {auc:.4f})",
    line=dict(color=C["blue"], width=2.5),
    fill="tozeroy", fillcolor="rgba(128,141,148,0.10)",
))
fig.add_trace(go.Scatter(
    x=[0,1], y=[0,1], mode="lines", name="Clasificador aleatorio",
    line=dict(color=C["gray"], width=1.5, dash="dash"),
))
fig.add_annotation(
    x=0.72, y=0.18,
    text=(f"<b>AUC-ROC = {auc:.4f}</b><br>"
          f"KS Stat  = {ks:.4f}<br>"
          f"F1 Score = {f1:.4f}"),
    showarrow=False, bgcolor=C["beige"], bordercolor=C["blue"],
    borderwidth=1.5, font=dict(size=12, color=C["text"]),
    align="left", xanchor="left",
)
fig.update_layout(**base_layout(
    "Curva ROC — CatBoost Default Hipotecario",
    "Tasa de Falsos Positivos (FPR)",
    "Tasa de Verdaderos Positivos (TPR)"))
save(fig, "fig_roc")

# Precision-Recall 
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=rec_arr, y=prec_arr, mode="lines", name="Curva P-R",
    line=dict(color=C["green"], width=2.5),
    fill="tozeroy", fillcolor="rgba(63,73,61,0.08)",
))
fig.add_trace(go.Scatter(
    x=[rec_arr[best_idx]], y=[prec_arr[best_idx]],
    mode="markers", name=f"Umbral óptimo = {best_thr:.3f}",
    marker=dict(color=C["brown"], size=13, symbol="circle",
                line=dict(color=C["white"], width=2)),
))
fig.add_annotation(
    x=rec_arr[best_idx] + 0.04, y=prec_arr[best_idx],
    text=f"Umbral = {best_thr:.3f}<br>F1 = {f1:.4f}",
    showarrow=True, arrowcolor=C["brown"], arrowwidth=1.5,
    ax=55, ay=-35, font=dict(size=12, color=C["text"]),
    bgcolor=C["beige"], bordercolor=C["brown"], borderwidth=1,
)
fig.update_layout(**base_layout(
    "Curva Precision-Recall — Umbral Óptimo por Max-F1",
    "Recall", "Precision"))
save(fig, "fig_pr")

# KS 
xg   = np.linspace(0, 1, 400)
cdf0 = np.array([(prob_test[y_test == 0] <= t).mean() for t in xg])
cdf1 = np.array([(prob_test[y_test == 1] <= t).mean() for t in xg])
ks_xi = int(np.argmax(np.abs(cdf0 - cdf1)))
ks_x  = float(xg[ks_xi])

fig = go.Figure()
fig.add_trace(go.Scatter(x=xg, y=cdf0, mode="lines", name="No Default",
    line=dict(color=C["green"], width=2.5)))
fig.add_trace(go.Scatter(x=xg, y=cdf1, mode="lines", name="Default",
    line=dict(color=C["brown"], width=2.5)))
fig.add_trace(go.Scatter(
    x=[ks_x, ks_x], y=[float(cdf1[ks_xi]), float(cdf0[ks_xi])],
    mode="lines+markers", name=f"KS = {ks:.4f}",
    line=dict(color=C["blue"], width=3, dash="dot"),
    marker=dict(color=C["blue"], size=10),
))
fig.add_annotation(
    x=ks_x + 0.03, y=(float(cdf0[ks_xi]) + float(cdf1[ks_xi])) / 2,
    text=f"<b>KS = {ks:.4f}</b>",
    showarrow=False, font=dict(size=13, color=C["blue"]),
    bgcolor=C["beige"], bordercolor=C["blue"], borderwidth=1.5,
)
fig.update_layout(**base_layout(
    "KS Statistic — Separación entre Distribuciones de Score",
    "Probabilidad Predicha (Score)", "CDF Acumulada"))
save(fig, "fig_ks")

# Confusion Matrix 
pcts   = cm / cm.sum() * 100
labels = [["TN", "FP"], ["FN", "TP"]]
z_text = [[f"<b>{labels[i][j]}</b><br>{cm[i,j]:,}<br>({pcts[i,j]:.1f}%)"
           for j in range(2)] for i in range(2)]
fig = go.Figure(go.Heatmap(
    z=cm, text=z_text, texttemplate="%{text}",
    colorscale=[[0, C["beige"]], [0.5, C["warm"]], [1, C["blue"]]],
    showscale=False, textfont=dict(size=20, color=C["text"]),
    xgap=5, ygap=5,
))
fig.update_layout(
    title=dict(text="Confusion Matrix",
               font=dict(size=17, color=C["text"], family="Georgia, serif"), x=0.05),
    paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
    font=FONT, width=600, height=520,
    margin=dict(t=75, b=65, l=75, r=45),
    xaxis=dict(title="Predicción", tickvals=[0,1],
               ticktext=["No Default (0)", "Default (1)"],
               tickfont=dict(color=C["text"], size=12),
               color=C["subtext"], gridcolor="rgba(0,0,0,0)"),
    yaxis=dict(title="Valor Real", tickvals=[0,1],
               ticktext=["No Default (0)", "Default (1)"],
               tickfont=dict(color=C["text"], size=12),
               color=C["subtext"], gridcolor="rgba(0,0,0,0)",
               autorange="reversed"),
)
save(fig, "fig_confusion")

# Score Distribution 
bins    = np.linspace(0, 1, 55)
centers = (bins[:-1] + bins[1:]) / 2
h0, _   = np.histogram(prob_test[y_test == 0], bins=bins, density=True)
h1, _   = np.histogram(prob_test[y_test == 1], bins=bins, density=True)

fig = go.Figure()
fig.add_trace(go.Bar(x=centers, y=h0, name="No Default (0)",
    marker_color=C["green"], opacity=0.75,
    hovertemplate="Score: %{x:.3f}<br>Densidad: %{y:.2f}<extra>No Default</extra>"))
fig.add_trace(go.Bar(x=centers, y=h1, name="Default (1)",
    marker_color=C["brown"], opacity=0.75,
    hovertemplate="Score: %{x:.3f}<br>Densidad: %{y:.2f}<extra>Default</extra>"))
fig.add_vline(x=best_thr, line_color=C["blue"], line_width=2.5, line_dash="dash",
    annotation_text=f"  Umbral = {best_thr:.3f}",
    annotation_font_color=C["blue"], annotation_bgcolor=C["beige"],
    annotation_bordercolor=C["blue"], annotation_font_size=12)
fig.update_layout(**base_layout(
    "Distribución de Scores Predichos por Clase",
    "Probabilidad Predicha", "Densidad", w=880, h=560),
    barmode="overlay")
save(fig, "fig_score_dist")

# Feature Importance 
fi = pd.Series(model.get_feature_importance(), index=feat_names).sort_values()
bar_colors = [C["blue"] if v >= fi.median() else C["gray"] for v in fi.values]

fig = go.Figure(go.Bar(
    x=fi.values, y=fi.index, orientation="h",
    marker_color=bar_colors,
    text=[f"{v:.1f}" for v in fi.values],
    textposition="outside", textfont=dict(color=C["text"], size=11),
    hovertemplate="%{y}: %{x:.2f}<extra></extra>",
))
fig.update_layout(**base_layout(
    "Feature Importance — CatBoost (Predictor Gain)",
    "Importancia", "", w=780, h=560))
fig.update_xaxes(range=[0, fi.max() * 1.20])
save(fig, "fig_feature_imp")

#  SHAP Importance 
mean_shap   = pd.Series(np.abs(shap_vals).mean(axis=0),
                        index=feat_names).sort_values()
shap_colors = [C["blue"] if v >= mean_shap.median() else C["gray"]
               for v in mean_shap.values]

fig = go.Figure(go.Bar(
    x=mean_shap.values, y=mean_shap.index, orientation="h",
    marker_color=shap_colors,
    text=[f"{v:.4f}" for v in mean_shap.values],
    textposition="outside", textfont=dict(color=C["text"], size=11),
    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
))
fig.update_layout(**base_layout(
    "SHAP — Importancia Media por Variable (Mean |SHAP|)",
    "Mean |SHAP value|", "", w=780, h=560))
fig.update_xaxes(range=[0, mean_shap.max() * 1.20])
save(fig, "fig_shap_importance")

# SHAP Scatter Top 6 
top6 = mean_shap.nlargest(6).index.tolist()
fig  = make_subplots(rows=2, cols=3,
    subplot_titles=[f"<b>{f}</b>" for f in top6],
    vertical_spacing=0.22, horizontal_spacing=0.10)

for i, feat in enumerate(top6):
    r, c   = divmod(i, 3)
    fi_idx = feat_names.index(feat)
    fvals  = shap_sample[feat].values
    if feat in cat_features:
        uniq   = sorted(set(fvals))
        fnum   = np.array([uniq.index(v) for v in fvals], dtype=float)
        cscale = [[0, C["beige"]], [0.5, C["warm"]], [1, C["brown"]]]
    else:
        fnum   = fvals.astype(float)
        cscale = [[0, C["green"]], [0.5, C["beige"]], [1, C["brown"]]]

    fig.add_trace(go.Scatter(
        x=fnum, y=shap_vals[:, fi_idx], mode="markers",
        marker=dict(color=fnum, colorscale=cscale,
                    size=4, opacity=0.50, showscale=False),
        name=feat,
        hovertemplate=f"{feat}: %{{x:.2f}}<br>SHAP: %{{y:.4f}}<extra></extra>",
    ), row=r+1, col=c+1)
    fig.update_xaxes(
        title_text="Valor de feature",
        title_font=dict(size=10, color=C["subtext"]),
        gridcolor=C["gray"], tickfont=dict(color=C["subtext"], size=9),
        linecolor=C["gray"], row=r+1, col=c+1)
    fig.update_yaxes(
        title_text="SHAP value",
        title_font=dict(size=10, color=C["subtext"]),
        gridcolor=C["gray"], tickfont=dict(color=C["subtext"], size=9),
        zeroline=True, zerolinecolor=C["gray"], zerolinewidth=1.5,
        row=r+1, col=c+1)

fig.update_layout(
    title=dict(text="SHAP — Efecto Individual por Observación (Top 6 Features)",
               font=dict(size=17, color=C["text"], family="Georgia, serif"), x=0.04),
    paper_bgcolor=C["bg"], plot_bgcolor=C["card"],
    font=FONT, showlegend=False,
    width=1000, height=640,
    margin=dict(t=85, b=50, l=65, r=30),
)
for ann in fig.layout.annotations:
    ann.font.color  = C["text"]
    ann.font.size   = 13
    ann.font.family = "Georgia, serif"
save(fig, "fig_shap_scatter")


# Barras apiladas por segmento 
comp_defs = [
    ("Componentes Fijos (TIIE + Inflación + Costos)",
     [FIXED] * 4, C["blue"]),
    ("Prima de Riesgo",
     [seg_means.loc[s, "PrimaRiesgo"]    if s in seg_means.index else 0 for s in SEG_ORDER],
     C["brown"]),
    ("Prima de Liquidez",
     [seg_means.loc[s, "PrimaLiquidez"] if s in seg_means.index else 0 for s in SEG_ORDER],
     C["warm"]),
    ("Margen de Utilidad",
     [seg_means.loc[s, "MargenUtilidad"]if s in seg_means.index else 0 for s in SEG_ORDER],
     C["green"]),
]
fig = go.Figure()
running = np.zeros(4)
for name, vals, color in comp_defs:
    fig.add_trace(go.Bar(
        name=name, x=SEG_ORDER, y=[v*100 for v in vals],
        base=[r*100 for r in running],
        marker_color=color, opacity=0.88,
        hovertemplate="<b>%{x}</b><br>" + name + ": %{y:.3f}%<extra></extra>",
    ))
    running += np.array(vals)
for i, seg in enumerate(SEG_ORDER):
    fig.add_annotation(x=seg, y=running[i]*100 + 0.05,
        text=f"<b>{running[i]*100:.2f}%</b>",
        showarrow=False, font=dict(size=12, color=C["text"]), yanchor="bottom")
fig.update_layout(**base_layout(
    "Descomposición de Tasa por Segmento de Riesgo",
    "Segmento de Riesgo (PD)", "Tasa Anual (%)", w=900, h=580),
    barmode="stack")
fig.update_xaxes(tickfont=dict(color=C["text"], size=12))
save(fig, "fig_precio_segmento")

# Clientes por segmento 
counts = df_p["SegmentoRiesgo"].value_counts().reindex(SEG_ORDER).fillna(0).astype(int)
pct_c  = counts / counts.sum() * 100

fig = go.Figure(go.Bar(
    x=SEG_ORDER, y=counts.values, marker_color=SEG_COLORS, opacity=0.88,
    text=[f"{v:,}<br>({p:.1f}%)" for v, p in zip(counts.values, pct_c)],
    textposition="outside", textfont=dict(color=C["text"], size=12),
    hovertemplate="<b>%{x}</b><br>Clientes: %{y:,}<extra></extra>",
))
fig.update_layout(**base_layout(
    "Distribución de Clientes por Segmento de Riesgo",
    "Segmento", "N° de Clientes", w=780, h=530))
fig.update_xaxes(tickfont=dict(color=C["text"], size=12))
fig.update_yaxes(range=[0, counts.max() * 1.18])
save(fig, "fig_clientes_segmento")

# PD vs Prima de Riesgo 
fig = go.Figure()
for seg, color in zip(SEG_ORDER, SEG_COLORS):
    mask = sample["SegmentoRiesgo"] == seg
    fig.add_trace(go.Scatter(
        x=sample.loc[mask, "PD"] * 100,
        y=sample.loc[mask, "PrimaRiesgo"] * 100,
        mode="markers", name=seg,
        marker=dict(color=color, size=4, opacity=0.55),
        hovertemplate="PD: %{x:.2f}%<br>Prima: %{y:.3f}%<extra>" + seg + "</extra>",
    ))
fig.add_hline(y=0.50, line_color=C["gray"], line_dash="dash", line_width=1.5,
    annotation_text="Floor 0.50%",
    annotation_font_color=C["subtext"], annotation_font_size=11)
fig.add_hline(y=2.50, line_color=C["brown"], line_dash="dash", line_width=1.5,
    annotation_text="Cap 2.50%",
    annotation_font_color=C["brown"], annotation_font_size=11)
fig.add_annotation(x=5, y=1.50,
    text="<b>Prima = clip(PD × LGD + overlays, 0.50%, 2.50%)</b><br>LGD = 28%",
    showarrow=False, bgcolor=C["beige"], bordercolor=C["blue"],
    borderwidth=1, font=dict(size=11, color=C["text"]),
    align="left", xanchor="left")
fig.update_layout(**base_layout(
    "PD → Prima de Riesgo Crediticio (EL = PD × LGD)",
    "PD Estimada (%)", "Prima de Riesgo (%)", w=860, h=570))
save(fig, "fig_pd_prima_riesgo")

# LoanTerm vs Prima de Liquidez 
term_liq = (df_p.groupby("LoanTerm")["PrimaLiquidez"]
              .mean().reset_index().sort_values("LoanTerm"))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=term_liq["LoanTerm"], y=term_liq["PrimaLiquidez"] * 100,
    mode="lines+markers+text",
    line=dict(color=C["blue"], width=2.5),
    marker=dict(color=C["blue"], size=12, line=dict(color=C["white"], width=2)),
    text=[f"{v*100:.2f}%" for v in term_liq["PrimaLiquidez"]],
    textposition="top center", textfont=dict(color=C["text"], size=12),
    hovertemplate="Plazo: %{x} meses<br>Prima Liquidez: %{y:.3f}%<extra></extra>",
))
fig.add_annotation(x=36, y=0.60,
    text="Interpolación lineal<br>entre plazo mín y máx",
    showarrow=False, bgcolor=C["beige"], bordercolor=C["blue"],
    borderwidth=1, font=dict(size=11, color=C["subtext"]))
fig.update_layout(**base_layout(
    "Prima de Liquidez por Plazo del Crédito",
    "Plazo (meses)", "Prima de Liquidez (%)", w=740, h=520))
fig.update_xaxes(tickvals=sorted(df_p["LoanTerm"].unique()),
                 tickfont=dict(color=C["text"]))
save(fig, "fig_plazo_liquidez")

#  LoanAmount vs Margen de Utilidad 
fig = go.Figure()
for seg, color in zip(SEG_ORDER, SEG_COLORS):
    mask = sample["SegmentoRiesgo"] == seg
    fig.add_trace(go.Scatter(
        x=sample.loc[mask, "LoanAmount"] / 1_000,
        y=sample.loc[mask, "MargenUtilidad"] * 100,
        mode="markers", name=seg,
        marker=dict(color=color, size=4, opacity=0.50),
        hovertemplate="Monto: $%{x:,.0f}k<br>Margen: %{y:.3f}%<extra>" + seg + "</extra>",
    ))
fig.add_annotation(
    x=sample["LoanAmount"].quantile(0.90) / 1_000, y=1.70,
    text="Escala inversa logarítmica:<br>monto mayor → menor margen",
    showarrow=False, bgcolor=C["beige"], bordercolor=C["gray"],
    borderwidth=1, font=dict(size=11, color=C["subtext"]), xanchor="right")
fig.update_layout(**base_layout(
    "Monto del Crédito vs Margen de Utilidad",
    "LoanAmount (miles)", "Margen de Utilidad (%)", w=860, h=560))
save(fig, "fig_monto_margen")

# Violin por segmento 
fig = go.Figure()
valid_segs = [(s, c) for s, c in zip(SEG_ORDER, SEG_COLORS)
              if (df_p["SegmentoRiesgo"] == s).sum() > 1]
for seg, color in valid_segs:
    vals = df_p.loc[df_p["SegmentoRiesgo"] == seg, "TasaConstruida"].values * 100
    fig.add_trace(go.Violin(
        y=vals, name=seg, box_visible=True, meanline_visible=True,
        fillcolor=color, opacity=0.65,
        line_color=C["subtext"], line_width=1, hoverinfo="y+name",
    ))
    fig.add_annotation(x=seg, y=vals.max() + 0.05,
        text=f"med {np.median(vals):.2f}%",
        showarrow=False, font=dict(size=10, color=C["text"]), yanchor="bottom")
fig.update_layout(**base_layout(
    "Distribución de Tasa Construida por Segmento de Riesgo",
    "Segmento de Riesgo", "Tasa Anual (%)", w=900, h=580),
    violinmode="overlay")
fig.update_xaxes(tickfont=dict(color=C["text"], size=12))
save(fig, "fig_violin_segmento")

#  Histograma overlay construida vs dataset 
bins_r = np.linspace(
    min(df_p["TasaConstruida"].min() * 100, df_p["InterestRate"].min()),
    max(df_p["TasaConstruida"].max() * 100, df_p["InterestRate"].max()), 55)
ctrs   = (bins_r[:-1] + bins_r[1:]) / 2
h_c, _ = np.histogram(df_p["TasaConstruida"] * 100, bins=bins_r, density=True)
h_d, _ = np.histogram(df_p["InterestRate"],          bins=bins_r, density=True)

fig = go.Figure()
fig.add_trace(go.Bar(x=ctrs, y=h_c, name="Tasa Construida",
    marker_color=C["blue"], opacity=0.72,
    hovertemplate="Tasa: %{x:.2f}%<br>Densidad: %{y:.3f}<extra>Construida</extra>"))
fig.add_trace(go.Bar(x=ctrs, y=h_d, name="InterestRate Dataset",
    marker_color=C["warm"], opacity=0.65,
    hovertemplate="Tasa: %{x:.2f}%<br>Densidad: %{y:.3f}<extra>Dataset</extra>"))
for val, color, label in [
    (df_p["TasaConstruida"].mean() * 100, C["blue"],
     f"  Media construida: {df_p['TasaConstruida'].mean()*100:.2f}%"),
    (df_p["InterestRate"].mean(), C["brown"],
     f"  Media dataset: {df_p['InterestRate'].mean():.2f}%"),
]:
    fig.add_vline(x=val, line_color=color, line_width=2, line_dash="dash",
        annotation_text=label, annotation_font_color=color,
        annotation_bgcolor=C["beige"], annotation_bordercolor=color,
        annotation_font_size=11)
fig.update_layout(**base_layout(
    "Distribución de Tasas: Construida vs Dataset",
    "Tasa Anual (%)", "Densidad", w=900, h=560),
    barmode="overlay")
save(fig, "fig_tasa_construida_vs_dataset")

# Scatter construida vs dataset 
fig = go.Figure()
for seg, color in zip(SEG_ORDER, SEG_COLORS):
    mask = sample["SegmentoRiesgo"] == seg
    fig.add_trace(go.Scatter(
        x=sample.loc[mask, "InterestRate"],
        y=sample.loc[mask, "TasaConstruida"] * 100,
        mode="markers", name=seg,
        marker=dict(color=color, size=4, opacity=0.50),
        hovertemplate="Dataset: %{x:.2f}%<br>Construida: %{y:.2f}%<extra>" + seg + "</extra>",
    ))
mn = min(df_p["InterestRate"].min(), df_p["TasaConstruida"].min() * 100)
mx = max(df_p["InterestRate"].max(), df_p["TasaConstruida"].max() * 100)
fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="y = x",
    line=dict(color=C["gray"], width=1.5, dash="dash")))
fig.add_annotation(x=0.97, y=0.05, xref="paper", yref="paper",
    text=(f"Pearson  ρ = {r_p:.4f}<br>Spearman ρ = {r_s:.4f}"),
    showarrow=False, bgcolor=C["beige"], bordercolor=C["blue"],
    borderwidth=1.5, font=dict(size=12, color=C["text"]),
    align="left", xanchor="right")
fig.update_layout(**base_layout(
    "Tasa Construida vs InterestRate del Dataset",
    "InterestRate Dataset (%)", "Tasa Construida (%)", w=860, h=570))
save(fig, "fig_scatter_construida_vs_dataset")

# Distribución de PD por clase real 
bins_pd = np.linspace(0, 1, 55)
ctr_pd  = (bins_pd[:-1] + bins_pd[1:]) / 2
h_nd, _ = np.histogram(df_p.loc[df_p["Default"] == 0, "PD"], bins=bins_pd, density=True)
h_df, _ = np.histogram(df_p.loc[df_p["Default"] == 1, "PD"], bins=bins_pd, density=True)

fig = go.Figure()
fig.add_trace(go.Bar(x=ctr_pd, y=h_nd, name="No Default (real)",
    marker_color=C["green"], opacity=0.75,
    hovertemplate="PD: %{x:.3f}<br>Densidad: %{y:.2f}<extra>No Default</extra>"))
fig.add_trace(go.Bar(x=ctr_pd, y=h_df, name="Default (real)",
    marker_color=C["brown"], opacity=0.75,
    hovertemplate="PD: %{x:.3f}<br>Densidad: %{y:.2f}<extra>Default</extra>"))
fig.add_vline(x=best_thr, line_color=C["blue"], line_width=2.5, line_dash="dash",
    annotation_text=f"  Umbral = {best_thr:.3f}",
    annotation_font_color=C["blue"], annotation_bgcolor=C["beige"],
    annotation_bordercolor=C["blue"], annotation_font_size=12)
fig.update_layout(**base_layout(
    "Distribución de PD Estimada por Clase Real de Default",
    "PD Estimada (CatBoost)", "Densidad", w=880, h=560),
    barmode="overlay")
save(fig, "fig_pd_distribucion")

print(f"\n {len(os.listdir('graficas_paleta'))} gráficas generadas en graficas_paleta/")