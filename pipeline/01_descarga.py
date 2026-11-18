"""Descarga las bases de CONASET (ArcGIS Open Data) usadas en el proyecto.

Fuente: https://mapas-conaset.opendata.arcgis.com/
Cada servicio entrega un máximo de registros por consulta, así que se pagina.

Uso:
    python pipeline/01_descarga.py            # descarga las tres bases
    python pipeline/01_descarga.py personas   # solo una
"""
from pathlib import Path
import sys
import time

import pandas as pd
import requests

BASE = "https://services3.arcgis.com/vaJl1B5HEzZj7154/arcgis/rest/services"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

BASES = {
    # siniestros georreferenciados 2020-2025 (base principal)
    "siniestros": ("Base_SINIESTROS_2020_2025", "FID", "siniestros_conaset_2020_2025.csv"),
    # una fila por persona involucrada, 2020-2024
    "personas": ("Base_de_persona_vehiculo_4", "ObjectId", "personas_conaset_2020_2024.csv"),
    # siniestros 2020-2024 con el mismo IdAccidente que la base de personas (tabla puente)
    "puente": ("Base_de_siniestros_2", "ObjectId", "siniestros_idaccidente_2020_2024.csv"),
}


def bajar_pagina(url: str, orden: str, offset: int, tam: int) -> list[dict]:
    params = {"where": "1=1", "outFields": "*", "returnGeometry": "false",
              "orderByFields": orden, "resultOffset": offset,
              "resultRecordCount": tam, "f": "json"}
    for intento in range(5):
        try:
            r = requests.get(url, params=params, timeout=90)
            r.raise_for_status()
            datos = r.json()
            if "error" in datos:
                raise RuntimeError(datos["error"])
            return [f["attributes"] for f in datos["features"]]
        except Exception as e:  # reintenta ante cortes de red
            print(f"  offset {offset}, intento {intento + 1}: {e}")
            time.sleep(3 * (intento + 1))
    raise RuntimeError(f"No se pudo bajar el offset {offset}")


def descargar(nombre: str) -> None:
    servicio, orden, archivo = BASES[nombre]
    capa = f"{BASE}/{servicio}/FeatureServer/0"
    tam = requests.get(capa, params={"f": "json"}, timeout=60).json().get("maxRecordCount", 2000)
    filas, offset = [], 0
    while True:
        pagina = bajar_pagina(f"{capa}/query", orden, offset, tam)
        if not pagina:
            break
        filas.extend(pagina)
        offset += len(pagina)
        if offset % 50000 < tam:
            print(f"[{nombre}] {offset} registros")
    df = pd.DataFrame(filas)
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW / archivo, index=False, encoding="utf-8")
    print(f"[{nombre}] listo: {len(df)} filas, {df.shape[1]} columnas -> {RAW / archivo}")


if __name__ == "__main__":
    for n in (sys.argv[1:] or BASES):
        descargar(n)
