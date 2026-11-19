"""App de visualización interactiva de los resultados de minería de datos.

Ejecutar desde la carpeta del repositorio:
    streamlit run app.py
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

RES = Path(__file__).resolve().parent / "resultados"
NARANJO, AZUL = "#dd6b20", "#2b6cb0"
st.set_page_config(page_title="Siniestros de tránsito en Chile", page_icon="🚦", layout="wide")


@st.cache_data
def cargar():
    reglas = pd.read_csv(RES / "reglas.csv")
    test = pd.read_csv(RES / "clasificacion_test.csv")
    com = pd.read_csv(RES / "comunas_clusters.csv")
    met = json.load(open(RES / "clasificacion_metricas.json", encoding="utf-8"))
    return reglas, test, com, met


@st.cache_resource
def modelo():
    m = joblib.load(RES / "modelo_gb.joblib")
    return m["modelo"], m["columnas"], shap.TreeExplainer(m["modelo"])


reglas, test, com, met = cargar()
BASE = met["tasa_base_grave"]

st.sidebar.title("🚦 Siniestros de tránsito")
PAGINAS = ["Inicio", "Condiciones de riesgo (reglas)", "Predicción de gravedad (clasificación)",
           "Tipos de comunas (clustering)", "Gravedad por comuna (regresión)"]
# ?p=reglas | clasificacion | clustering | regresion abre directamente esa página
atajo = {"reglas": 1, "clasificacion": 2, "clustering": 3, "regresion": 4}.get(st.query_params.get("p", ""), 0)
pagina = st.sidebar.radio("Ir a", PAGINAS, index=atajo)
st.sidebar.caption("Datos: CONASET, siniestros 2020–2025 (partes de Carabineros). "
                   "Los resultados muestran asociaciones, no causas.")

# =============================================================================
if pagina == "Inicio":
    st.title("¿Qué hace que un siniestro de tránsito termine en muerte?")
    st.write("Esta aplicación muestra, en lenguaje simple, lo que encontraron cuatro técnicas de "
             "minería de datos al analizar 435 mil siniestros ocurridos en Chile entre 2020 y 2025.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Siniestros analizados", "435.400")
    c2.metric("Terminan graves o fatales", f"{BASE:.0%}")
    c3.metric("Comunas agrupadas", f"{len(com)}")
    st.markdown("""
**Qué puede hacer aquí**
- **Condiciones de riesgo:** ver qué combinaciones de lugar, hora y tipo de siniestro hacen más probable una muerte o una lesión grave.
- **Predicción de gravedad:** ver qué tan bien un modelo anticipa los siniestros graves, en qué se equivoca y por qué marcó un caso como grave.
- **Tipos de comunas:** elegir un grupo de comunas y conocer qué lo caracteriza.
- **Gravedad por comuna:** ver qué características de una comuna van de la mano con siniestros más graves y simular cuánto cambiaría su gravedad.

**Cómo leer los porcentajes:** cuando decimos "grave o fatal" nos referimos a siniestros con al menos una persona fallecida o con lesiones graves. En promedio, uno de cada diez siniestros lo es.
""")

# =============================================================================
elif pagina == "Condiciones de riesgo (reglas)":
    st.title("¿Qué combinaciones de condiciones hacen más grave un siniestro?")
    st.write(f"Cada fila dice: **si se dan estas condiciones**, el siniestro termina grave o fatal en "
             f"cierto porcentaje de los casos. Como referencia, el promedio nacional es **{BASE:.0%}**.")
    c1, c2, c3 = st.columns(3)
    min_veces = c1.slider("Mostrar solo si el riesgo es al menos ... veces el promedio", 1.0, 7.0, 2.0, 0.5)
    min_casos = c2.slider("Casos mínimos que respaldan la regla", 800, 5000, 1000, 100)
    ocultar = c3.checkbox("Ocultar reglas obvias (imprudencia del peatón)", value=True,
                          help="«Imprudencia del peatón» implica un atropello, así que esas reglas no aportan información nueva.")
    items = sorted({i for a in reglas["antecedente"] for i in a.split(" + ")})
    contiene = st.multiselect("Que incluya estas condiciones (opcional)", items)

    f = reglas[(reglas["lift"] >= min_veces) & (reglas["casos"] >= min_casos)]
    if ocultar:
        f = f[~f["antecedente"].str.contains("peatón")]
    for it in contiene:
        f = f[f["antecedente"].str.split(" + ", regex=False).apply(lambda s: it in s)]
    f = f.sort_values("lift", ascending=False)
    st.caption(f"{len(f)} reglas cumplen los filtros.")

    tabla = pd.DataFrame({
        "Si ocurre...": f["antecedente"].str.replace(" + ", " · ", regex=False),
        "...es grave o fatal en": (f["confianza"] * 100).round(0).astype(int).astype(str) + "%",
        "Veces el promedio": f["lift"].round(1),
        "Siniestros que la respaldan": f["casos"],
    })
    st.dataframe(tabla, width="stretch", hide_index=True, height=360)

    if len(f):
        fig = px.bar(f.head(15).iloc[::-1], x="confianza", y="antecedente", orientation="h",
                     text=f.head(15).iloc[::-1]["lift"].map(lambda v: f"×{v:.1f}"),
                     labels={"confianza": "% grave o fatal", "antecedente": ""})
        fig.update_traces(marker_color=AZUL)
        fig.add_vline(x=BASE, line_dash="dash", line_color=NARANJO, annotation_text="promedio")
        fig.update_layout(xaxis_tickformat=".0%", height=520)
        st.plotly_chart(fig, width="stretch")

# =============================================================================
elif pagina == "Predicción de gravedad (clasificación)":
    st.title("¿Se puede anticipar si un siniestro será grave?")
    st.write("Un modelo (gradient boosting) estima, para cada siniestro, la probabilidad de que termine "
             "grave o fatal, usando solo lo que se sabe al momento: lugar, hora, tipo y vehículos involucrados. "
             "Se evaluó con 87 mil siniestros que no vio durante el entrenamiento.")

    umbral = st.slider("Marcar como grave si la probabilidad supera...", 0.1, 0.9, 0.5, 0.05,
                       help="Un umbral bajo detecta más siniestros graves, pero da más falsas alarmas.")
    pred = np.where(test["probabilidad"] >= umbral, "Grave o fatal", "No grave")
    real = test["real"]
    vp = int(((pred == "Grave o fatal") & (real == "Grave o fatal")).sum())
    fn = int(((pred == "No grave") & (real == "Grave o fatal")).sum())
    fp = int(((pred == "Grave o fatal") & (real == "No grave")).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Graves detectados", f"{vp / (vp + fn):.0%}", help="Recall")
    c2.metric("Alertas que eran graves", f"{vp / max(vp + fp, 1):.0%}", help="Precisión")
    c3.metric("Graves que se le escaparon", f"{fn:,}".replace(",", "."))

    m = pd.crosstab(pd.Series(real, name="Real"), pd.Series(pred, name="Predicho"))
    m = m.reindex(index=["No grave", "Grave o fatal"], columns=["No grave", "Grave o fatal"], fill_value=0)
    fig = px.imshow(m, color_continuous_scale="Blues",
                    labels={"x": "Lo que predijo el modelo", "y": "Lo que realmente pasó", "color": "casos"})
    fig.update_traces(text=m.map(lambda v: f"{v:,}".replace(",", ".")).values, texttemplate="%{text}",
                      textfont_size=18)
    fig.update_layout(height=360, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Explorar casos")
    tipo_caso = st.radio("Ver", ["Errores: graves no detectados", "Errores: falsas alarmas",
                                 "Aciertos: graves detectados", "Todos los graves reales"], horizontal=True)
    filtro = {"Errores: graves no detectados": (real == "Grave o fatal") & (pred == "No grave"),
              "Errores: falsas alarmas": (real == "No grave") & (pred == "Grave o fatal"),
              "Aciertos: graves detectados": (real == "Grave o fatal") & (pred == "Grave o fatal"),
              "Todos los graves reales": real == "Grave o fatal"}[tipo_caso]
    casos = test[filtro].copy()
    reg = st.selectbox("Región", ["Todas"] + sorted(casos["region"].unique()))
    if reg != "Todas":
        casos = casos[casos["region"] == reg]
    vista = casos[["id", "fecha", "hora", "comuna", "zona", "tipo", "ubicacion_via", "motocicleta",
                   "probabilidad", "real"]].sort_values("probabilidad", ascending=False)
    st.dataframe(vista.head(300), width="stretch", hide_index=True, height=280)

    st.subheader("¿Por qué el modelo decidió eso?")
    if len(vista):
        elegido = st.selectbox("Elija un siniestro por su ID", vista["id"].head(300))
        fila = test[test["id"] == elegido].iloc[0]
        gb, columnas, expl = modelo()
        cat = ["zona", "tipo_via", "franja_horaria", "dia_semana", "tipo", "ubicacion_via", "region"]
        num = ["hora", "mes_num", "fin_de_semana", "atropello", "bicicleta", "motocicleta"]
        x = pd.get_dummies(fila[cat].to_frame().T, dtype=np.uint8).join(
            fila[num].to_frame().T.astype(float)).reindex(columns=columnas, fill_value=0)
        sv = expl.shap_values(x)[0]
        aporte = pd.Series(sv, index=columnas)
        aporte = aporte[x.iloc[0].values != 0].sort_values(key=abs, ascending=False).head(8)[::-1]
        prefijos = {"tipo_via_": "Vía: ", "tipo_": "Tipo: ", "zona_": "Zona: ", "ubicacion_via_": "Lugar: ",
                    "franja_horaria_": "Franja: ", "dia_semana_": "Día: ", "region_": "Región: "}
        simples = {"hora": f"Hora: {int(fila['hora'])}", "mes_num": f"Mes: {int(fila['mes_num'])}",
                   "motocicleta": "Hay una moto", "bicicleta": "Hay una bicicleta",
                   "atropello": "Es atropello", "fin_de_semana": "Fin de semana"}

        def legible(c):
            if c in simples:
                return simples[c]
            for p, t in prefijos.items():
                if c.startswith(p):
                    return t + c[len(p):]
            return c
        aporte.index = [legible(c) for c in aporte.index]
        fig = go.Figure(go.Bar(x=aporte.values, y=aporte.index, orientation="h",
                               marker_color=[NARANJO if v > 0 else AZUL for v in aporte.values]))
        fig.update_layout(height=380, xaxis_title="← baja el riesgo · sube el riesgo →",
                          title=f"Probabilidad estimada: {fila['probabilidad']:.0%} · resultado real: {fila['real']}")
        st.plotly_chart(fig, width="stretch")
        st.caption("Barras naranjas: características que hicieron que el modelo viera este siniestro como más "
                   "grave. Azules: las que bajaron su estimación. Calculado con SHAP.")

elif pagina == "Gravedad por comuna (regresión)":
    reg = json.load(open(RES / "regresion_modelo.json", encoding="utf-8"))
    rc = pd.read_csv(RES / "regresion_comunas.csv")
    VARS = reg["variables"]
    st.title("¿Qué hace que en algunas comunas los siniestros sean más graves?")
    st.write("Una **regresión lineal** estima el % de siniestros graves o fatales de cada comuna a partir de "
             "cinco características de sus siniestros. Sirve para ver **cuánto pesa cada una** y qué comunas "
             "tienen más gravedad de la que se esperaría por su perfil.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Comunas analizadas", reg["n_comunas"])
    c2.metric("Diferencias que explica el modelo", f"{reg['r2_validacion_cruzada']:.0%}",
              help="R² en validación cruzada: qué parte de las diferencias entre comunas explica el modelo. "
                   "El resto depende de factores que no están en los datos.")
    c3.metric("Error típico", f"± {reg['mae_validacion_cruzada'] * 100:.1f} pts".replace(".", ","),
              help="Error absoluto medio en validación cruzada, en puntos porcentuales.")

    st.subheader("1. Una característica a la vez")
    var = st.selectbox("Comparar el % grave o fatal con...", list(VARS), format_func=lambda v: VARS[v])
    a, b = np.polyfit(rc[var], rc["pct_grave"], 1)[::-1]
    resaltar = st.multiselect("Resaltar comunas", sorted(rc["comuna"]), max_selections=5)
    fig = px.scatter(rc, x=var, y="pct_grave", size="siniestros", size_max=22, hover_name="comuna",
                     color=np.where(rc["comuna"].isin(resaltar), "Seleccionada", "Comuna"),
                     color_discrete_map={"Comuna": "#90cdf4", "Seleccionada": NARANJO},
                     hover_data={"region": True, "siniestros": True},
                     labels={var: f"% {VARS[var].lower()}", "pct_grave": "% grave o fatal", "color": ""})
    xs = np.linspace(rc[var].min(), rc[var].max(), 50)
    fig.add_scatter(x=xs, y=a + b * xs, mode="lines", line=dict(color="#1a202c", width=3), name="Tendencia")
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=440, legend_title="")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Tendencia: cada 10 puntos más de «{VARS[var].lower()}», el % grave o fatal cambia "
               f"{b * 10:+.1f} puntos. Cada punto es una comuna; el tamaño indica cuántos siniestros tiene.".replace(".", ",", 1))

    st.subheader("2. Todas juntas: ¿cuánto pesa cada una?")
    sd = pd.Series(reg["desviaciones"])
    ef = (pd.Series(reg["coeficientes"]) * sd * 100).rename(
        index={v: f"{VARS[v]} (±{sd[v] * 100:.0f} pts)" for v in VARS}).sort_values()
    fig = go.Figure(go.Bar(x=ef.values, y=ef.index, orientation="h", marker_color=NARANJO,
                           text=[f"+{v:.1f} pts".replace(".", ",") for v in ef.values], textposition="outside"))
    fig.update_layout(height=320, xaxis_title="Cambio en el % grave o fatal si la comuna estuviera una diferencia "
                                              "típica por encima del promedio en...",
                      xaxis_range=[0, ef.max() * 1.3])
    st.plotly_chart(fig, width="stretch")
    st.caption("Cada barra muestra el efecto de esa característica manteniendo las demás iguales. Entre paréntesis, "
               "cuánto suele variar entre comunas (por ejemplo, la ruralidad varía ±29 puntos y la noche solo ±4). "
               "Comparar «10 puntos más» de cada una sería engañoso, porque 10 puntos más de noche es un cambio "
               "que casi ninguna comuna tiene.")

    st.subheader("3. Simulador")
    st.write("Elija una comuna y mueva las características para ver cuánto cambiaría su gravedad estimada.")
    base_com = st.selectbox("Comuna de partida", sorted(rc["comuna"]),
                            index=sorted(rc["comuna"]).index("Arica") if "Arica" in set(rc["comuna"]) else 0)
    fila = rc[rc["comuna"] == base_com].iloc[0]
    cols = st.columns(len(VARS))
    valores = {v: cols[i].slider(VARS[v], 0, 100, int(round(fila[v] * 100)), key=f"sim_{v}") / 100
               for i, v in enumerate(VARS)}
    # el perfil actual se calcula con los mismos valores redondeados que muestran los deslizadores
    est_orig = reg["intercepto"] + sum(reg["coeficientes"][v] * round(fila[v] * 100) / 100 for v in VARS)
    est_nuevo = reg["intercepto"] + sum(reg["coeficientes"][v] * valores[v] for v in VARS)
    d1, d2, d3 = st.columns(3)
    d1.metric("% grave real de la comuna", f"{fila['pct_grave']:.1%}".replace(".", ","))
    d2.metric("Estimado con su perfil actual", f"{est_orig:.1%}".replace(".", ","))
    d3.metric("Estimado con los cambios", f"{est_nuevo:.1%}".replace(".", ","),
              (f"{(est_nuevo - est_orig) * 100:+.1f} pts".replace(".", ",") if abs(est_nuevo - est_orig) >= 0.0005 else None), delta_color="inverse")
    st.caption("Es una estimación de un modelo simple: indica asociaciones, no garantiza que cambiar una "
               "característica cambie la gravedad.")

    st.subheader("4. Comunas más graves de lo esperado")
    st.write("Diferencia entre la gravedad real y la que el modelo esperaba por su perfil. Las de arriba "
             "merecen una revisión: algo que el modelo no ve las hace más graves.")
    minimo = st.slider("Solo comunas con al menos ... siniestros", 100, 3000, 500, 100,
                       help="En comunas con pocos siniestros el porcentaje varía mucho por azar.")
    tabla = rc[rc["siniestros"] >= minimo].sort_values("residuo", ascending=False).head(15)
    st.dataframe(pd.DataFrame({
        "Comuna": tabla["comuna"], "Región": tabla["region"], "Siniestros": tabla["siniestros"],
        "% grave real": (tabla["pct_grave"] * 100).round(1), "% esperado": (tabla["predicho"] * 100).round(1),
        "Diferencia (pts)": (tabla["residuo"] * 100).round(1)}),
        width="stretch", hide_index=True, height=300)

# =============================================================================
else:
    st.title("¿Qué tipos de comunas hay según sus siniestros?")
    st.write("Agrupamos 300 comunas con al menos 100 siniestros según seis características de sus "
             "siniestros y la cantidad total que registran. "
             "Elija un grupo para ver qué lo distingue.")
    grupos = ["Urbanas de alto volumen", "Intermedias con más atropellos",
              "Rurales con más siniestros de noche", "Rurales con curvas (las más letales)"]
    colores = dict(zip(grupos, ["#2b6cb0", "#38a169", "#dd6b20", "#c53030"]))
    g = st.selectbox("Grupo", grupos, index=3)
    sel = com[com["nombre_grupo"] == g]
    prom = com[["pct_grave", "pct_rural", "pct_noche", "pct_atropello", "pct_moto", "pct_curva"]].mean()
    c = st.columns(4)
    c[0].metric("Comunas", len(sel))
    c[1].metric("% grave o fatal", f"{sel.pct_grave.mean():.1%}",
                f"{(sel.pct_grave.mean() - prom.pct_grave) * 100:+.1f} pts vs promedio", delta_color="inverse")
    c[2].metric("% en zona rural", f"{sel.pct_rural.mean():.0%}")
    c[3].metric("% en curva", f"{sel.pct_curva.mean():.0%}")

    etiquetas = {"pct_grave": "Grave o fatal", "pct_rural": "Rural", "pct_noche": "De noche",
                 "pct_atropello": "Atropellos", "pct_moto": "Con moto", "pct_curva": "En curva"}
    comp = pd.DataFrame({"Este grupo": sel[list(etiquetas)].mean(), "Promedio de las comunas": prom}).rename(index=etiquetas)
    fig = px.bar(comp.reset_index(), x="index", y=["Este grupo", "Promedio de las comunas"], barmode="group",
                 color_discrete_sequence=[colores[g], "#a0aec0"], labels={"index": "", "value": "%", "variable": ""})
    fig.update_layout(yaxis_tickformat=".0%", height=380)
    st.plotly_chart(fig, width="stretch")

    mapa = com.dropna(subset=["lat", "lon"]).query("lon > -80").assign(
        seleccion=lambda x: np.where(x.nombre_grupo == g, g, "Otros grupos"))
    fig = px.scatter_geo(mapa, lat="lat", lon="lon", color="seleccion",
                         color_discrete_map={g: colores[g], "Otros grupos": "#cbd5e0"}, hover_name="comuna",
                         height=800)
    fig.update_geos(fitbounds="locations", projection_type="mercator", showcountries=True, countrycolor="#b0b0b0", showland=True, landcolor="#f2f0eb", showocean=True, oceancolor="#e8f1f8", showlakes=False, resolution=50)
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
    fig.update_layout(legend_title="", legend=dict(orientation="h", y=-0.02),
                      margin=dict(l=0, r=0, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.write("Comunas del grupo, de más a menos siniestros:")
    st.dataframe(sel.sort_values("siniestros", ascending=False)[["comuna", "region", "siniestros"]]
                 .assign(**{"% grave o fatal": (sel["pct_grave"] * 100).round(1)}),
                 width="stretch", hide_index=True, height=260)
