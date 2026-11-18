"""Seminario de visualización: aplica las tres técnicas sobre la base del proyecto
y guarda sus resultados para visualizarlos después.

Entrada : data/processed/siniestros_limpio.csv
Salidas : resultados/
    reglas.csv                 reglas de asociación con consecuente "Grave o fatal"
    clasificacion_test.csv     casos de prueba con clase real, predicha y probabilidad
    clasificacion_metricas.json
    importancia.csv            importancia por permutación (global)
    shap_valores.npz           valores SHAP de una muestra de prueba
    comunas_clusters.csv       perfil de cada comuna y su grupo
    clusters_resumen.csv / clustering_k.csv
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, recall_score, roc_auc_score)
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "resultados"
OUT.mkdir(parents=True, exist_ok=True)
SEMILLA = 42

d = pd.read_csv(RAIZ / "data" / "processed" / "siniestros_limpio.csv", low_memory=False)
d = d[d["ubicacion_via"] != "Sin dato"].copy()
d["dia"] = np.where(d["fin_de_semana"] == 1, "Fin de semana", "Día hábil")
print("siniestros:", len(d))

# =============================================================================
# 1. Reglas de asociación (FP-Growth)
#    Cada siniestro es una "transacción" con sus características como ítems.
# =============================================================================
items = pd.DataFrame({
    "Zona": "Zona " + d["zona"].str.lower(),
    "Franja": d["franja_horaria"].astype(str),
    "Día": d["dia"],
    "Tipo": "Tipo: " + d["tipo"].str.lower(),
    "Lugar": "Lugar: " + d["ubicacion_via"].str.lower(),
    "Causa": "Causa: " + d["causa"].str.lower(),
})
trans = pd.get_dummies(items, prefix="", prefix_sep="").astype(bool)
trans["Grave o fatal"] = d["grave_o_fatal"].astype(bool).values
trans["Motocicleta"] = d["motocicleta"].astype(bool).values
trans["Bicicleta"] = d["bicicleta"].astype(bool).values

# soporte mínimo 0,2%: unos 870 siniestros, suficiente para que una regla no sea casualidad
frecuentes = fpgrowth(trans, min_support=0.002, use_colnames=True, max_len=4)
reglas = association_rules(frecuentes, metric="lift", min_threshold=1.0)
reglas = reglas[reglas["consequents"] == frozenset({"Grave o fatal"})].copy()
reglas["antecedente"] = reglas["antecedents"].apply(lambda s: " + ".join(sorted(s)))
reglas["n_items"] = reglas["antecedents"].apply(len)
reglas["casos"] = (reglas["support"] * len(trans)).round().astype(int)
base_grave = trans["Grave o fatal"].mean()
reglas = reglas[["antecedente", "n_items", "support", "confidence", "lift", "casos"]]
reglas = reglas.rename(columns={"support": "soporte", "confidence": "confianza"})
reglas = reglas.sort_values("lift", ascending=False)
reglas.to_csv(OUT / "reglas.csv", index=False, encoding="utf-8")
print(f"reglas con consecuente grave: {len(reglas)}, con lift>=2: {(reglas.lift >= 2).sum()}")
print(reglas.head(10).to_string(index=False))

# =============================================================================
# 2. Clasificación: ¿el siniestro será grave o fatal?
#    Solo variables conocidas al momento del siniestro. Se excluyen los conteos de
#    víctimas (definen la clase) y la causa (la asigna Carabineros después).
# =============================================================================
cat = ["zona", "tipo_via", "franja_horaria", "dia_semana", "tipo", "ubicacion_via", "region"]
num = ["hora", "mes_num", "fin_de_semana", "atropello", "bicicleta", "motocicleta"]
X = pd.get_dummies(d[cat], dtype=np.uint8).join(d[num].astype(float))
y = d["grave_o_fatal"].values
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, y, d.index, test_size=0.2, stratify=y, random_state=SEMILLA)

modelos = {
    "Regresión logística": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Gradient boosting": HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=SEMILLA),
}
metricas = {"tasa_base_grave": float(y.mean()), "n_train": int(len(y_tr)), "n_test": int(len(y_te))}
probas = {}
for nombre, m in modelos.items():
    Xa = X_tr if nombre != "Regresión logística" else (X_tr - X_tr.mean()) / (X_tr.std() + 1e-9)
    Xb = X_te if nombre != "Regresión logística" else (X_te - X_tr.mean()) / (X_tr.std() + 1e-9)
    m.fit(Xa, y_tr)
    p = m.predict_proba(Xb)[:, 1]
    pred = (p >= 0.5).astype(int)
    probas[nombre] = p
    metricas[nombre] = {
        "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred),
        "auc_roc": roc_auc_score(y_te, p), "auc_pr": average_precision_score(y_te, p),
        "matriz": confusion_matrix(y_te, pred).tolist(),
    }
    print(nombre, {k: round(v, 3) for k, v in metricas[nombre].items() if k != "matriz"})

mejor = modelos["Gradient boosting"]
p = probas["Gradient boosting"]
test = d.loc[idx_te, cat + num + ["comuna", "fecha"]].copy()
test.insert(0, "id", d.loc[idx_te, "id"].values)
test["real"] = np.where(y_te == 1, "Grave o fatal", "No grave")
test["predicho"] = np.where(p >= 0.5, "Grave o fatal", "No grave")
test["probabilidad"] = p.round(3)
test["acierto"] = test["real"] == test["predicho"]
test.to_csv(OUT / "clasificacion_test.csv", index=False, encoding="utf-8")
print(classification_report(y_te, (p >= 0.5).astype(int), target_names=["No grave", "Grave/fatal"]))

# importancia global por permutación, agrupando las columnas one-hot por variable original
rng = np.random.RandomState(SEMILLA)
muestra = rng.choice(len(X_te), 20000, replace=False)
pi = permutation_importance(mejor, X_te.iloc[muestra], y_te[muestra], scoring="average_precision",
                            n_repeats=5, random_state=SEMILLA, n_jobs=-1)
imp = pd.DataFrame({"columna": X.columns, "importancia": pi.importances_mean})
origen = {c: next((v for v in cat if c.startswith(v + "_")), c) for c in X.columns}
imp["variable"] = imp["columna"].map(origen)
imp_var = imp.groupby("variable")["importancia"].sum().sort_values(ascending=False)
imp_var.to_csv(OUT / "importancia.csv", header=["importancia"], encoding="utf-8")
print(imp_var)

# SHAP: explicación de cada predicción para 3.000 casos de prueba
m_shap = rng.choice(len(X_te), 3000, replace=False)
expl = shap.TreeExplainer(mejor)
sv = expl.shap_values(X_te.iloc[m_shap])
np.savez_compressed(OUT / "shap_valores.npz", valores=sv, datos=X_te.iloc[m_shap].values,
                    columnas=np.array(X.columns), base=np.array(expl.expected_value).ravel(),
                    indices=np.array(idx_te[m_shap]))

import joblib
joblib.dump({"modelo": mejor, "columnas": list(X.columns)}, OUT / "modelo_gb.joblib")

with open(OUT / "clasificacion_metricas.json", "w", encoding="utf-8") as f:
    json.dump(metricas, f, ensure_ascii=False, indent=2)

# =============================================================================
# 3. Clustering de comunas (K-Means)
# =============================================================================
c = d.assign(noche=d["hora"].isin(list(range(0, 6)) + list(range(18, 24))).astype(int),
             rural=(d["zona"] == "Rural").astype(int),
             curva=(d["ubicacion_via"] == "Tramo en curva").astype(int))
com = c.groupby(["region", "comuna"]).agg(
    siniestros=("id", "size"), pct_grave=("grave_o_fatal", "mean"),
    pct_fatal=("fallecidos", lambda s: (s > 0).mean()), pct_rural=("rural", "mean"),
    pct_noche=("noche", "mean"), pct_atropello=("atropello", "mean"),
    pct_moto=("motocicleta", "mean"), pct_curva=("curva", "mean"),
    lat=("lat", "median"), lon=("lon", "median")).reset_index()
com = com[com["siniestros"] >= 100].copy()  # tasas estables
vars_k = ["pct_grave", "pct_rural", "pct_noche", "pct_atropello", "pct_moto", "pct_curva"]
Z = StandardScaler().fit_transform(com[vars_k].assign(log_n=np.log10(com["siniestros"])))

pruebas = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEMILLA).fit(Z)
    pruebas.append({"k": k, "inercia": km.inertia_, "silueta": silhouette_score(Z, km.labels_)})
pruebas = pd.DataFrame(pruebas)
pruebas.to_csv(OUT / "clustering_k.csv", index=False)
print(pruebas.round(3))
K = int(pruebas.loc[pruebas["k"].between(3, 6), :].sort_values("silueta", ascending=False).iloc[0]["k"])
km = KMeans(n_clusters=K, n_init=20, random_state=SEMILLA).fit(Z)
com["grupo"] = km.labels_
pca = PCA(n_components=2, random_state=SEMILLA).fit(Z)
com[["pc1", "pc2"]] = pca.transform(Z)
com.to_csv(OUT / "comunas_clusters.csv", index=False, encoding="utf-8")
resumen = com.groupby("grupo").agg(comunas=("comuna", "size"), siniestros_mediana=("siniestros", "median"),
                                   **{v: (v, "mean") for v in vars_k})
resumen.to_csv(OUT / "clusters_resumen.csv", encoding="utf-8")
print("k elegido:", K, " varianza PCA:", pca.explained_variance_ratio_.round(3))
print(resumen.round(3).to_string())
for g in range(K):
    print(g, com[com.grupo == g].sort_values("siniestros", ascending=False).comuna.head(8).tolist())
