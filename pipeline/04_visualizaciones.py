"""Seminario de visualización: genera las visualizaciones estáticas (PNG) e
interactivas (HTML) de las tres técnicas a partir de resultados/.

Salida: visualizaciones/
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import shap
from pyvis.network import Network
from sklearn.metrics import precision_recall_curve

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "resultados"
VIS = BASE / "visualizaciones"
VIS.mkdir(exist_ok=True)
RAIZ = BASE
AZUL, NARANJO, GRIS, OSCURO = "#2b6cb0", "#dd6b20", "#a0aec0", "#1B2550"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def png(nombre):
    plt.tight_layout()
    plt.savefig(VIS / nombre, dpi=180)
    plt.close()


def html(fig, nombre):
    fig.update_layout(font_family="Arial", template="plotly_white")
    fig.write_html(VIS / nombre, include_plotlyjs="cdn", full_html=True)


# =============================================================================
# REGLAS DE ASOCIACIÓN
# =============================================================================
r = pd.read_csv(RES / "reglas.csv")
base = json.load(open(RES / "clasificacion_metricas.json", encoding="utf-8"))["tasa_base_grave"]
# Reglas "no obvias": sin la causa ligada al peatón, que por definición implica un atropello
no_obvias = r[~r["antecedente"].str.contains("peatón")]
top = no_obvias.sort_values("lift", ascending=False).drop_duplicates("antecedente").head(12)

# R1 · Estático: ranking de reglas por lift, en lenguaje simple
fig, ax = plt.subplots(figsize=(9, 5.2))
etiquetas = top["antecedente"].str.replace(" + ", "  y  ", regex=False)[::-1]
ax.barh(etiquetas, top["confianza"][::-1] * 100, color=AZUL)
ax.axvline(base * 100, color=NARANJO, ls="--", lw=1.5)
ax.text(base * 100 + 0.5, -0.9, f"promedio: {base:.0%}", color=NARANJO, fontsize=9)
for i, (c, l) in enumerate(zip(top["confianza"][::-1] * 100, top["lift"][::-1])):
    ax.text(c + 0.5, i, f"{c:.0f}%  (×{l:.1f})", va="center", fontsize=8)
ax.set_xlim(0, top["confianza"].max() * 118)
ax.set_xlabel("% de siniestros graves o fatales cuando se cumplen las condiciones  (×: veces el promedio)")
ax.set_title("Combinaciones de condiciones que más elevan la gravedad")
png("reglas_ranking.png")

# R2 · Estático: dispersión soporte–confianza (vista técnica clásica)
fig, ax = plt.subplots(figsize=(7.5, 4.8))
sc = ax.scatter(r["soporte"] * 100, r["confianza"] * 100, c=r["lift"], cmap="viridis",
                s=12 + 10 * r["n_items"], alpha=0.8)
fig.colorbar(sc, label="lift")
ax.set_xscale("log")
ax.set_xlabel("soporte (% de siniestros, escala log)")
ax.set_ylabel("confianza (%)")
ax.set_title(f"{len(r)} reglas con consecuente «grave o fatal»")
png("reglas_dispersion.png")

# R3 · Interactivo: la misma dispersión, con detalle al pasar el mouse
fig = px.scatter(r.assign(soporte_pct=r.soporte * 100, confianza_pct=r.confianza * 100),
                 x="soporte_pct", y="confianza_pct", color="lift", size="n_items",
                 log_x=True, color_continuous_scale="Viridis",
                 hover_data={"antecedente": True, "casos": True, "lift": ":.2f",
                             "soporte_pct": ":.2f", "confianza_pct": ":.1f", "n_items": False},
                 labels={"soporte_pct": "Soporte (% de siniestros)",
                         "confianza_pct": "Confianza (% graves o fatales)"},
                 title="Reglas de asociación: pase el mouse sobre un punto para ver la regla")
fig.add_hline(y=base * 100, line_dash="dash", line_color=NARANJO,
              annotation_text=f"promedio nacional {base:.0%}")
html(fig, "reglas_dispersion.html")

# R4 · Interactivo: red de reglas (condiciones → regla → grave o fatal)
# Diseño jerárquico de izquierda a derecha: condiciones (nivel 0) → regla (1) → resultado (2)
red = Network(height="640px", width="100%", directed=True, bgcolor="#ffffff", cdn_resources="remote")
red.add_node("Grave o fatal", label="GRAVE O FATAL", color=NARANJO, size=45, shape="dot", level=2,
             font={"size": 26})
for k, fila in enumerate(top.head(10).itertuples()):
    id_r = f"r{k}"
    red.add_node(id_r, label=f"{fila.confianza:.0%} grave · ×{fila.lift:.1f}", shape="box",
                 color="#bee3f8", level=1, font={"size": 18},
                 title=(f"Si: {fila.antecedente}\nEntonces: grave o fatal en el {fila.confianza:.0%} "
                        f"de los casos ({fila.lift:.1f} veces el promedio)\nCasos: {fila.casos:,}"))
    red.add_edge(id_r, "Grave o fatal", width=fila.lift, color="#cbd5e0")
    for item in fila.antecedente.split(" + "):
        red.add_node(item, label=item, color=OSCURO, shape="box", level=0,
                     font={"color": "white", "size": 20})
        red.add_edge(item, id_r, color="#a0aec0")
red.set_options("""{
  "layout": {"hierarchical": {"enabled": true, "direction": "LR", "sortMethod": "directed",
                              "levelSeparation": 330, "nodeSpacing": 52}},
  "physics": {"enabled": false},
  "nodes": {"font": {"size": 20}},
  "edges": {"smooth": {"type": "cubicBezier", "forceDirection": "horizontal"},
            "arrows": {"to": {"scaleFactor": 0.5}}},
  "interaction": {"hover": true, "navigationButtons": true}
}""")
red.write_html(str(VIS / "reglas_red.html"), notebook=False)

# =============================================================================
# CLASIFICACIÓN
# =============================================================================
t = pd.read_csv(RES / "clasificacion_test.csv")
met = json.load(open(RES / "clasificacion_metricas.json", encoding="utf-8"))
gb = met["Gradient boosting"]

# C1 · Estático: matriz de confusión con conteos y porcentajes por fila
m = np.array(gb["matriz"])
pct = m / m.sum(axis=1, keepdims=True)
etiq = np.array([[f"{m[i, j]:,}\n({pct[i, j]:.0%})".replace(",", ".") for j in range(2)]
                 for i in range(2)])
fig, ax = plt.subplots(figsize=(5.6, 4.3))
sns.heatmap(pct, annot=etiq, fmt="", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["No grave", "Grave o fatal"], yticklabels=["No grave", "Grave o fatal"])
ax.set_xlabel("Lo que predijo el modelo")
ax.set_ylabel("Lo que realmente pasó")
ax.set_title("Matriz de confusión (87.080 siniestros de prueba)")
png("clasif_matriz.png")

# C2 · Estático: curva precisión-recall de los dos modelos
fig, ax = plt.subplots(figsize=(6.2, 4.3))
y = (t["real"] == "Grave o fatal").astype(int)
prec, rec, _ = precision_recall_curve(y, t["probabilidad"])
ax.plot(rec, prec, color=AZUL, label=f"Gradient boosting (AUC-PR {gb['auc_pr']:.2f})")
ax.axhline(met["tasa_base_grave"], color=GRIS, ls="--",
           label=f"Azar (AUC-PR {met['tasa_base_grave']:.2f})")
ax.set_xlabel("Recall: % de graves que el modelo detecta")
ax.set_ylabel("Precisión: % de alertas que eran graves")
ax.legend(fontsize=8)
ax.set_title("Curva precisión-recall")
png("clasif_curva_pr.png")

# C3 · Estático: importancia global por permutación
imp = pd.read_csv(RES / "importancia.csv", index_col=0)["importancia"]
imp = imp[imp > 0.001].sort_values()
nombres = {"tipo": "Tipo de siniestro", "motocicleta": "Hay una moto", "tipo_via": "Ciudad o ruta",
           "bicicleta": "Hay una bicicleta", "hora": "Hora", "region": "Región", "zona": "Urbana o rural",
           "ubicacion_via": "Lugar de la vía", "franja_horaria": "Franja horaria"}
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh([nombres.get(i, i) for i in imp.index], imp.values / imp.sum() * 100, color=AZUL)
ax.set_xlabel("Peso relativo en la predicción (%)")
ax.set_title("¿Qué mira el modelo para predecir un siniestro grave?")
png("clasif_importancia.png")

# C4 y C5 · SHAP: global (beeswarm) y explicación de un caso (waterfall)
z = np.load(RES / "shap_valores.npz", allow_pickle=True)
sv, datos, cols, base_shap = z["valores"], z["datos"], list(z["columnas"]), float(z["base"][0])
legible = {c: (c.replace("tipo_via_", "Vía: ").replace("tipo_", "Tipo: ").replace("zona_", "Zona: ")
               .replace("ubicacion_via_", "Lugar: ").replace("franja_horaria_", "Franja: ")
               .replace("dia_semana_", "Día: ").replace("region_", "Región: ")
               .replace("motocicleta", "Hay moto").replace("bicicleta", "Hay bicicleta")
               .replace("fin_de_semana", "Fin de semana").replace("hora", "Hora").replace("mes_num", "Mes")) for c in cols}
exp = shap.Explanation(values=sv, base_values=np.full(len(sv), base_shap), data=datos,
                       feature_names=[legible[c] for c in cols])
plt.figure()
shap.plots.beeswarm(exp, max_display=12, show=False)
plt.xlabel("Valor SHAP: ← baja el riesgo estimado · lo sube →")
plt.title("SHAP: cómo empuja cada variable la predicción")
png("clasif_shap_global.png")

# caso: el siniestro de prueba con mayor probabilidad que sí fue grave
caso = int(np.argmax(sv.sum(axis=1)))
plt.figure()
shap.plots.waterfall(exp[caso], max_display=10, show=False)
plt.title("SHAP: por qué el modelo marcó este siniestro como grave")
png("clasif_shap_caso.png")

# C6 · Interactivo: distribución de probabilidades por clase real
fig = px.histogram(t, x="probabilidad", color="real", nbins=40, barmode="overlay", opacity=0.65,
                   color_discrete_map={"Grave o fatal": NARANJO, "No grave": AZUL},
                   labels={"probabilidad": "Probabilidad estimada de que sea grave", "real": "Resultado real"},
                   title="¿Separa el modelo los siniestros graves de los no graves?")
fig.add_vline(x=0.5, line_dash="dash", annotation_text="umbral 0,5")
html(fig, "clasif_probabilidades.html")

# =============================================================================
# CLUSTERING
# =============================================================================
com = pd.read_csv(RES / "comunas_clusters.csv")
res = com.groupby("grupo")[["pct_rural", "pct_curva", "pct_grave", "pct_atropello", "siniestros"]].mean()
nombre = {}
nombre[res["pct_curva"].idxmax()] = "Rurales con curvas (las más letales)"
nombre[res["pct_rural"].idxmin()] = "Urbanas de alto volumen"
resto = [g for g in res.index if g not in nombre]
nombre[res.loc[resto, "pct_rural"].idxmax()] = "Rurales con más siniestros de noche"
for g in res.index:
    nombre.setdefault(g, "Intermedias con más atropellos")
com["nombre_grupo"] = com["grupo"].map(nombre)
com.to_csv(RES / "comunas_clusters.csv", index=False, encoding="utf-8")
paleta = {"Urbanas de alto volumen": "#2b6cb0", "Intermedias con más atropellos": "#38a169",
          "Rurales con más siniestros de noche": "#dd6b20", "Rurales con curvas (las más letales)": "#c53030"}

# K1 · Estático: elección de k
k = pd.read_csv(RES / "clustering_k.csv")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(k["k"], k["inercia"], marker="o", color=AZUL); ax[0].set_title("Método del codo")
ax[0].set_xlabel("número de grupos (k)"); ax[0].set_ylabel("inercia")
ax[1].plot(k["k"], k["silueta"], marker="o", color=NARANJO); ax[1].set_title("Coeficiente de silueta")
ax[1].set_xlabel("número de grupos (k)")
png("cluster_eleccion_k.png")

# K2 · Estático: dispersión PCA
fig, ax = plt.subplots(figsize=(7.5, 5))
for n, g in com.groupby("nombre_grupo"):
    ax.scatter(g["pc1"], g["pc2"], s=18, color=paleta[n], label=f"{n} ({len(g)})", alpha=0.8)
ax.set_xlabel("Componente principal 1 (46% de la variación)")
ax.set_ylabel("Componente principal 2 (16%)")
ax.legend(fontsize=8, loc="best")
ax.set_title("Comunas agrupadas por K-Means (proyección PCA)")
png("cluster_pca.png")

# K3 · Estático: perfil de cada grupo (mapa de calor de promedios)
perfil = com.groupby("nombre_grupo")[["pct_grave", "pct_rural", "pct_noche", "pct_atropello",
                                       "pct_moto", "pct_curva"]].mean() * 100
perfil.columns = ["% grave o fatal", "% rural", "% de noche", "% atropellos", "% con moto", "% en curva"]
perfil = perfil.loc[list(paleta)]
fig, ax = plt.subplots(figsize=(8.5, 3.4))
sns.heatmap((perfil - perfil.mean()) / perfil.std(), annot=perfil.round(1), fmt="", cmap="RdBu_r",
            center=0, cbar_kws={"label": "sobre (rojo) o bajo (azul) el promedio"}, ax=ax)
ax.set_ylabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
ax.set_title("Perfil de cada grupo de comunas (valores en %)")
png("cluster_perfil.png")

# K4 · Interactivo: mapa de comunas por grupo
mapa = com.dropna(subset=["lat", "lon"]).query("lon > -80").assign(
    grave=lambda x: (x.pct_grave * 100).round(1), rural=lambda x: (x.pct_rural * 100).round(1),
    curva=lambda x: (x.pct_curva * 100).round(1))
fig = px.scatter_geo(mapa, lat="lat", lon="lon", color="nombre_grupo",
                     color_discrete_map=paleta, height=1000, width=620,
                     hover_name="comuna",
                     hover_data={"region": True, "siniestros": ":,", "grave": True, "rural": True,
                                 "curva": True, "lat": False, "lon": False, "nombre_grupo": False},
                     labels={"nombre_grupo": "Grupo", "grave": "% grave o fatal", "rural": "% rural",
                             "curva": "% en curva"},
                     title="Grupos de comunas según su siniestralidad")
fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.01, x=0, title=""),
                  margin=dict(l=10, r=10, t=60, b=90))
fig.update_geos(fitbounds="locations", projection_type="mercator", showcountries=True, countrycolor="#b0b0b0", showland=True, landcolor="#f2f0eb", showocean=True, oceancolor="#e8f1f8", showlakes=False, resolution=50)
html(fig, "cluster_mapa.html")

# K5 · Interactivo: perfil en coordenadas paralelas
cp = com.assign(g=com["nombre_grupo"].map({n: i for i, n in enumerate(paleta)}))
fig = go.Figure(go.Parcoords(
    line=dict(color=cp["g"], colorscale=[[i / 3, c] for i, c in enumerate(paleta.values())]),
    dimensions=[dict(label=l, values=cp[v] * 100) for l, v in
                [("% grave o fatal", "pct_grave"), ("% rural", "pct_rural"), ("% de noche", "pct_noche"),
                 ("% atropellos", "pct_atropello"), ("% con moto", "pct_moto"), ("% en curva", "pct_curva")]]))
fig.update_layout(title="Coordenadas paralelas: arrastre sobre un eje para filtrar comunas")
html(fig, "cluster_paralelas.html")

print("Listo:", sorted(p.name for p in VIS.iterdir()))
print(com.groupby("nombre_grupo").size())
