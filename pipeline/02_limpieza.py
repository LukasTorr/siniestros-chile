"""Limpieza y preprocesamiento de la base de siniestros CONASET 2020-2025.

Entrada : data/raw/siniestros_conaset_2020_2025.csv
Salida  : data/processed/siniestros_limpio.csv
          data/processed/bitacora_limpieza.json  (conteos de cada paso)
"""
from pathlib import Path
import json
import unicodedata

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "data" / "raw" / "siniestros_conaset_2020_2025.csv"
SALIDA = RAIZ / "data" / "processed" / "siniestros_limpio.csv"
BITACORA = RAIZ / "data" / "processed" / "bitacora_limpieza.json"

bitacora: dict = {}


def sin_tildes(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))


# ----------------------------------------------------------------------------
# 1. Carga
# ----------------------------------------------------------------------------
df = pd.read_csv(ENTRADA, low_memory=False)
bitacora["registros_originales"] = len(df)
bitacora["columnas_originales"] = df.shape[1]

# ----------------------------------------------------------------------------
# 2. Columnas que se eliminan
#    - FID: índice interno de ArcGIS.
#    - Calle_Uno, Calle_Dos, Ruta, Número, Ubicación, Dirección: texto libre de la
#      dirección. Muy alta cardinalidad, entre 21% y 81% vacíos, y la posición ya
#      está en Latitud/longitud.
#    - Siniestro: solo vale "Si" cuando FALLECIDOS > 0 (verificado 100%), es redundante.
# ----------------------------------------------------------------------------
eliminar = ["FID", "Calle_Uno", "Calle_Dos", "Ruta", "Número", "Ubicación",
            "Dirección", "Siniestro"]
assert ((df["Siniestro"] == "Si") == (df["FALLECIDOS"] > 0)).all()
df = df.drop(columns=eliminar)
bitacora["columnas_eliminadas"] = eliminar

# ----------------------------------------------------------------------------
# 3. Duplicados
#    No hay duplicados exactos ni IDs repetidos. Se buscan duplicados "lógicos":
#    mismo día, hora, comuna, coordenadas, tipo y mismas víctimas.
# ----------------------------------------------------------------------------
bitacora["duplicados_identicos_salvo_id"] =int(df.drop(columns=["ID"]).duplicated().sum())
clave = ["FECHA", "Hora", "COMUNA", "Latitud", "longitud", "TIPO_SINIE",
         "CAUSA_SINI", "FALLECIDOS", "GRAVES", "MENOS_GRAV", "LEVES"]
dup = df.duplicated(subset=clave, keep="first")
bitacora["duplicados_logicos_eliminados"] = int(dup.sum())
df = df[~dup].copy()

# ----------------------------------------------------------------------------
# 4. Fecha y hora
# ----------------------------------------------------------------------------
df["fecha"] = pd.to_datetime(df["FECHA"], unit="ms").dt.date
fecha = pd.to_datetime(df["fecha"])
assert (fecha.dt.year == df["AÑO"]).all()
df["anio"] = df["AÑO"]
df["mes_num"] = fecha.dt.month
df["mes"] = df["Mes"].str.strip()
df["dia_semana"] = df["Dia_semana"].str.strip()
df["fin_de_semana"] = df["dia_semana"].isin(["Sábado", "Domingo"]).astype(int)

partes = df["Hora"].astype(str).str.split(":", expand=True)
df["hora"] = partes[0].astype(int)
df["minuto"] = partes[1].astype(int)
assert (df["hora"] == df["Hora_aprox"]).all()
# Una de cada cinco horas termina en :00, señal de que el registro se redondea.
df["hora_redondeada"] = (df["minuto"] == 0).astype(int)
bitacora["pct_horas_en_punto"] = round(df["hora_redondeada"].mean() * 100, 1)

df["franja_horaria"] = pd.cut(
    df["hora"], bins=[-1, 5, 11, 17, 23],
    labels=["Madrugada (0-5)", "Mañana (6-11)", "Tarde (12-17)", "Noche (18-23)"])

# ----------------------------------------------------------------------------
# 5. Variables de gravedad (variable objetivo)
#    Se clasifica cada siniestro por la peor consecuencia registrada, igual que
#    la escala KABCO usada en seguridad vial.
# ----------------------------------------------------------------------------
assert ((df["GRAVES"] + df["MENOS_GRAV"] + df["LEVES"]) == df["TOTAL_LESI"]).all()
condiciones = [df["FALLECIDOS"] > 0, df["GRAVES"] > 0, df["MENOS_GRAV"] > 0, df["LEVES"] > 0]
df["gravedad"] = np.select(condiciones, ["Fatal", "Grave", "Menos grave", "Leve"],
                           default="Sin lesionados")
# Fallecido o lesionado grave (indicador KSI, "killed or seriously injured").
df["grave_o_fatal"] = ((df["FALLECIDOS"] > 0) | (df["GRAVES"] > 0)).astype(int)
df["victimas"] = df["FALLECIDOS"] + df["TOTAL_LESI"]
bitacora["distribucion_gravedad"] = df["gravedad"].value_counts().to_dict()

# ----------------------------------------------------------------------------
# 6. Tipo de siniestro y causa
# ----------------------------------------------------------------------------
tildes = {"Colision": "Colisión", "Caida": "Caída", "Peaton": "Peatón",
          "peaton": "peatón", "señalizacion": "señalización", "Distraccion": "Distracción",
          "mecanicas": "mecánicas", "vehiculo": "vehículo", "frontal": "frontal"}


def acentuar(s: pd.Series) -> pd.Series:
    s = s.str.strip().str.capitalize()
    for sin, con in tildes.items():
        s = s.str.replace(rf"\b{sin}\b", con, regex=True)
    return s


df["tipo"] = acentuar(df["TIPO__CONA"])
df["subtipo"] = acentuar(df["TIPO_SINIE"])
df["causa"] = acentuar(df["CAUSA_NUEV"])
df["causa_detalle"] = df["CAUSA_SINI"].str.strip().str.capitalize()
df["causa_indeterminada"] = df["causa"].isin(["Causas no determinadas", "Otras causas"]).astype(int)

# ----------------------------------------------------------------------------
# 7. Ubicación en la vía
#    El campo trae 207 valores: tildes inconsistentes (SEÑAL/SENAL), valores
#    repetidos separados por ";" (cuando participan varios vehículos) y combinaciones.
#    Se toma el primer valor y se agrupa en 9 categorías.
# ----------------------------------------------------------------------------
ubic = (df["UBICACION"].fillna("").astype(str).map(sin_tildes).str.upper()
        .str.split(";").str[0].str.strip())


def grupo_ubicacion(u: str) -> str:
    if u == "":
        return "Sin dato"
    if "SEMAFORO" in u:
        return "Cruce con semáforo"
    if "PARE" in u:
        return "Cruce con señal Pare"
    if "CEDA" in u:
        return "Cruce con señal Ceda el paso"
    if "CRUCE" in u:  # 'CRUCE SIN SENALIZACION' y 'CRUCE CON SENAL' sin detalle
        return "Cruce sin señal / señal no especificada"
    if "CURVA" in u:
        return "Tramo en curva"
    if "RECTA" in u:
        return "Tramo recto"
    if "ROTONDA" in u or "ENLACE" in u or "PUENTE" in u or "TUNEL" in u:
        return "Rotonda, enlace o puente"
    return "Otro lugar"


df["ubicacion_via"] = ubic.map(grupo_ubicacion)
bitacora["ubicacion_valores_originales"] = int(df["UBICACION"].nunique())
bitacora["ubicacion_categorias_finales"] = int(df["ubicacion_via"].nunique())
bitacora["ubicacion_sin_dato"] = int((df["ubicacion_via"] == "Sin dato").sum())

# ----------------------------------------------------------------------------
# 8. Territorio
# ----------------------------------------------------------------------------
df["region"] = df["REGION"].str.strip()
df["provincia"] = df["PROVINCIA"].str.strip()
df["comuna"] = df["COMUNA"].str.strip()
df["zona"] = df["ZONA"].str.strip().str.capitalize()            # criterio Carabineros
df["tipo_via"] = df["Zonas_urb_"].str.strip().replace({"Rutas": "Ruta"})  # criterio geográfico
bitacora["zona_vs_tipo_via_discrepantes"] = int(
    ((df["zona"] == "Urbana") & (df["tipo_via"] == "Ruta")).sum()
    + ((df["zona"] == "Rural") & (df["tipo_via"] == "Urbano")).sum())

nombres_am = {"Gran Concepcion": "Gran Concepción", "Gran Valparaiso": "Gran Valparaíso",
              "temuco": "Gran Temuco", "OHiggins": "Gran Rancagua"}
df["area_metropolitana"] = (df["AM"].fillna("").str.strip().replace(nombres_am)
                            .replace("", "Fuera de área metropolitana"))

# ----------------------------------------------------------------------------
# 9. Coordenadas
#    Todas están dentro de Chile (Isla de Pascua, lon -109, es válida). Se marcan
#    como dudosas las que quedan a más de 100 km de la mediana de su comuna:
#    son errores de geocodificación (p. ej. "Ruta 5 Sur km 687" asignado a Talca
#    pero ubicado en Padre Las Casas). No se borra el registro, solo se anula la
#    coordenada para los análisis espaciales.
# ----------------------------------------------------------------------------
med = df.groupby("comuna")[["Latitud", "longitud"]].transform("median")
dist_km = np.hypot(df["Latitud"] - med["Latitud"], df["longitud"] - med["longitud"]) * 111
df["coord_dudosa"] = (dist_km > 100).astype(int)
df["lat"] = df["Latitud"].where(df["coord_dudosa"] == 0)
df["lon"] = df["longitud"].where(df["coord_dudosa"] == 0)
bitacora["coordenadas_dudosas"] = int(df["coord_dudosa"].sum())

# ----------------------------------------------------------------------------
# 10. Participantes vulnerables
# ----------------------------------------------------------------------------
df["atropello"] = df["Atropello"].astype(int)
df["bicicleta"] = df["Bicicleta"].astype(int)
df["motocicleta"] = df["Motociclet"].astype(int)

# ----------------------------------------------------------------------------
# 11. Valores extremos de víctimas: se revisaron los 15 siniestros con más de 30
#     lesionados. Son volcaduras y colisiones de buses, casos reales. Se conservan.
# ----------------------------------------------------------------------------
bitacora["siniestros_mas_30_lesionados"] = int((df["TOTAL_LESI"] > 30).sum())

# ----------------------------------------------------------------------------
# 12. Selección y orden final
# ----------------------------------------------------------------------------
final = df[[
    "ID", "fecha", "anio", "mes_num", "mes", "dia_semana", "fin_de_semana", "hora",
    "minuto", "hora_redondeada", "franja_horaria",
    "region", "provincia", "comuna", "area_metropolitana", "zona", "tipo_via",
    "lat", "lon", "coord_dudosa",
    "tipo", "subtipo", "causa", "causa_detalle", "causa_indeterminada", "ubicacion_via",
    "atropello", "bicicleta", "motocicleta",
    "FALLECIDOS", "GRAVES", "MENOS_GRAV", "LEVES", "TOTAL_LESI", "victimas",
    "gravedad", "grave_o_fatal",
]].rename(columns={"ID": "id", "FALLECIDOS": "fallecidos", "GRAVES": "les_graves",
                   "MENOS_GRAV": "les_menos_graves", "LEVES": "les_leves",
                   "TOTAL_LESI": "total_lesionados"})

assert final.drop(columns=["lat", "lon"]).isna().sum().sum() == 0
SALIDA.parent.mkdir(parents=True, exist_ok=True)
final.to_csv(SALIDA, index=False, encoding="utf-8")

bitacora["registros_finales"] = len(final)
bitacora["columnas_finales"] = final.shape[1]
BITACORA.write_text(json.dumps(bitacora, ensure_ascii=False, indent=2, default=int),
                    encoding="utf-8")
print(json.dumps(bitacora, ensure_ascii=False, indent=2, default=int))
