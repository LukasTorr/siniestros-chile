"""Regresión lineal: ¿qué características de una comuna explican su % de siniestros graves o fatales?

Usa el perfil de las 300 comunas (resultados/comunas_clusters.csv, generado en el proyecto).
Salidas:
    resultados/regresion_comunas.csv         real, predicho (validación cruzada) y residuo por comuna
    resultados/regresion_modelo.json         coeficientes y métricas (los lee la app)
    visualizaciones/reg_*.png                gráficos estáticos para el informe
    visualizaciones/reg_interactivo.html     versión interactiva (Plotly)

Ejecutar:  python pipeline/05_regresion.py  (después de 03_modelos.py y 04_visualizaciones.py)
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
import plotly.express as px
import plotly.graph_objects as go
from yellowbrick.regressor import PredictionError, ResidualsPlot

AQUI = Path(__file__).resolve().parents[1]  # raíz del repositorio
RES = AQUI / "resultados"
FIG = AQUI / "visualizaciones"
FIG.mkdir(exist_ok=True)
AZUL, NARANJO, ROJO, GRIS = "#2b6cb0", "#dd6b20", "#c53030", "#a0aec0"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

VARS = {"pct_rural": "Siniestros en zona rural", "pct_noche": "Siniestros de noche",
        "pct_atropello": "Atropellos", "pct_moto": "Siniestros con moto", "pct_curva": "Siniestros en curva"}


def pct(v, dec=0):
    """Formatea una proporción como porcentaje con coma decimal."""
    return f"{v * 100:.{dec}f}%".replace(".", ",")


def guardar(nombre):
    plt.tight_layout()
    plt.savefig(FIG / nombre, dpi=200)
    plt.close("all")


com = pd.read_csv(RES / "comunas_clusters.csv")
X, y = com[list(VARS)], com["pct_grave"]
kf = KFold(10, shuffle=True, random_state=42)

# Modelo simple (una variable) y múltiple (cinco variables)
simple = LinearRegression().fit(com[["pct_rural"]], y)
multiple = LinearRegression().fit(X, y)
r2_cv = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2").mean()
mae_cv = -cross_val_score(LinearRegression(), X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
r2_cv_simple = cross_val_score(LinearRegression(), com[["pct_rural"]], y, cv=kf, scoring="r2").mean()
pred_cv = cross_val_predict(LinearRegression(), X, y, cv=kf)

modelo = {
    "n_comunas": len(com),
    "variables": VARS,
    "intercepto": multiple.intercept_,
    "coeficientes": dict(zip(VARS, multiple.coef_)),
    "r2_entrenamiento": multiple.score(X, y),
    "r2_validacion_cruzada": r2_cv,
    "mae_validacion_cruzada": mae_cv,
    "promedios": X.mean().to_dict(),
    "desviaciones": X.std().to_dict(),
    "simple": {"variable": "pct_rural", "intercepto": simple.intercept_, "pendiente": simple.coef_[0],
               "r2_entrenamiento": simple.score(com[["pct_rural"]], y), "r2_validacion_cruzada": r2_cv_simple},
}
json.dump(modelo, open(RES / "regresion_modelo.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

salida = com[["region", "comuna", "siniestros", "nombre_grupo", *VARS, "pct_grave"]].assign(
    predicho=pred_cv, residuo=lambda d: d["pct_grave"] - d["predicho"])
salida.to_csv(RES / "regresion_comunas.csv", index=False, encoding="utf-8")

print(f"Simple  (rural): R2 = {modelo['simple']['r2_entrenamiento']:.3f}  R2 CV = {r2_cv_simple:.3f}  "
      f"pendiente = {simple.coef_[0]:.3f}")
print(f"Multiple       : R2 = {modelo['r2_entrenamiento']:.3f}  R2 CV = {r2_cv:.3f}  MAE CV = {mae_cv:.4f}")
for v, c in modelo["coeficientes"].items():
    print(f"  {VARS[v]:<28} +10 pts -> {c * 10:+.2f} | +1 desv. ({X[v].std() * 100:.1f} pts) -> "
          f"{c * X[v].std() * 100:+.2f} pts de % grave")

# --- Figura 1: regresión simple (la más fácil de leer para un no experto) ---
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.scatter(com["pct_rural"], y, s=np.sqrt(com["siniestros"]) * 0.8, color=AZUL, alpha=0.45, edgecolor="white")
xs = np.linspace(0, 1, 50)
ax.plot(xs, simple.intercept_ + simple.coef_[0] * xs, color=NARANJO, lw=2.5)
for _, r in com.nlargest(4, "pct_grave").iterrows():
    ax.annotate(r["comuna"], (r["pct_rural"], r["pct_grave"]), fontsize=8, xytext=(-6, 4),
                textcoords="offset points", ha="right", color="#4a5568")
ax.xaxis.set_major_formatter(lambda v, _: pct(v))
ax.yaxis.set_major_formatter(lambda v, _: pct(v))
ax.set_xlabel("% de los siniestros de la comuna que ocurren en zona rural")
ax.set_ylabel("% de siniestros graves o fatales")
ax.set_title(f"Más ruralidad, siniestros más graves: cada 10 puntos de ruralidad suman "
             f"{simple.coef_[0] * 10:.1f} pts de gravedad".replace(".", ","), fontsize=10)
ax.text(0.99, 0.02, "Cada punto es una comuna (tamaño = n° de siniestros)", transform=ax.transAxes,
        ha="right", fontsize=8, color="#718096")
guardar("reg_1_simple_ruralidad.png")

# --- Figura 2: los mismos coeficientes mostrados de dos formas ---
# (a) efecto de 10 puntos más: parece justo, pero 10 puntos no significan lo mismo en todas las variables
#     (la noche varía ±4 puntos entre comunas y la ruralidad ±29);
# (b) efecto de una diferencia típica entre comunas (una desviación estándar): es la comparación correcta.
coef = pd.Series(modelo["coeficientes"])
sd = pd.Series(modelo["desviaciones"])
ef10 = (coef * 10).rename(index=VARS)
efsd = (coef * sd * 100).rename(index={v: f"{VARS[v]} (±{sd[v] * 100:.0f} pts)" for v in VARS})
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, ef, titulo, xl in [
        (axs[0], ef10.sort_values(), "(a) Si la comuna tuviera 10 puntos más de...",
         "Cambio en el % grave o fatal (puntos)"),
        (axs[1], efsd.sort_values(), "(b) Si la comuna estuviera una diferencia típica\npor encima del promedio en...",
         "Cambio en el % grave o fatal (puntos)")]:
    ax.barh(ef.index, ef.values, color=[NARANJO if ax is axs[0] else ROJO] * len(ef))
    for i, v in enumerate(ef.values):
        ax.text(v + 0.04, i, f"+{v:.1f}".replace(".", ","), va="center", fontsize=9)
    ax.set_title(titulo, fontsize=10, loc="left")
    ax.set_xlabel(xl, fontsize=9)
    ax.set_xlim(0, ef.max() * 1.25)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}")
fig.suptitle("Mismos coeficientes, dos lecturas: en (a) parece que lo que más pesa es la moto; "
             "en (b), la ruralidad", fontsize=10.5, x=0.01, ha="left")
guardar("reg_2_coeficientes.png")

# --- Figuras 3 y 4: diagnóstico técnico con Yellowbrick (75% entrenamiento, 25% prueba) ---
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
fig, ax = plt.subplots(figsize=(5.5, 5))
viz = PredictionError(LinearRegression(), ax=ax, alpha=0.5, force_model=True)
viz.fit(X_tr, y_tr); viz.score(X_te, y_te)
viz.finalize()
ax.set_xlabel("% grave real"); ax.set_ylabel("% grave predicho")
ax.set_title("Real vs. predicho (Yellowbrick PredictionError)", fontsize=10)
guardar("reg_3_prediction_error.png")

fig, ax = plt.subplots(figsize=(7, 4))
viz = ResidualsPlot(LinearRegression(), ax=ax, hist=False, qqplot=True, force_model=True)
viz.fit(X_tr, y_tr); viz.score(X_te, y_te)
viz.finalize()
ax.set_title("Residuos del modelo (Yellowbrick ResidualsPlot)", fontsize=10)
ax.set_xlabel("% grave predicho"); ax.set_ylabel("Residuo (real − predicho)")
guardar("reg_4_residuos.png")
print("Figuras de regresión guardadas en", FIG)

# --- Versión interactiva: la misma dispersión, con el nombre de cada comuna al pasar el mouse ---
GRUPOS = ["Urbanas de alto volumen", "Intermedias con más atropellos",
          "Rurales con más siniestros de noche", "Rurales con curvas (las más letales)"]
COLORES = dict(zip(GRUPOS, ["#2b6cb0", "#38a169", "#dd6b20", "#c53030"]))
s = modelo["simple"]
fig = px.scatter(com, x="pct_rural", y="pct_grave", size="siniestros", size_max=25, color="nombre_grupo",
                 category_orders={"nombre_grupo": GRUPOS}, color_discrete_map=COLORES, hover_name="comuna",
                 hover_data={"region": True, "siniestros": ":,", "pct_grave": ":.1%", "pct_rural": ":.1%",
                             "nombre_grupo": False},
                 labels={"pct_rural": "% de siniestros en zona rural", "pct_grave": "% grave o fatal",
                         "region": "Región", "siniestros": "Siniestros"},
                 title=f"Cada 10 puntos más de ruralidad, la gravedad sube {s['pendiente'] * 10:.1f} puntos "
                       f"(R² = {s['r2_entrenamiento']:.2f})".replace(".", ","))
fig.add_trace(go.Scatter(x=xs, y=s["intercepto"] + s["pendiente"] * xs, mode="lines", name="Recta de regresión",
                         line=dict(color="#1a202c", width=3)))
fig.update_layout(template="plotly_white", xaxis_tickformat=".0%", yaxis_tickformat=".0%", legend_title="",
                  height=600)
fig.write_html(FIG / "reg_interactivo.html", include_plotlyjs=True)
print("HTML guardado en", FIG / "reg_interactivo.html")
