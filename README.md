# ¿Qué hace que un siniestro de tránsito termine en muerte?

Aplicación web que muestra, en lenguaje simple, los resultados de cuatro técnicas de minería de datos aplicadas a 435 mil siniestros de tránsito registrados en Chile entre 2020 y 2025.

- **Condiciones de riesgo (reglas de asociación):** qué combinaciones de lugar, hora y tipo de siniestro hacen más probable una muerte o una lesión grave.
- **Predicción de gravedad (clasificación):** qué tan bien un modelo anticipa los siniestros graves, en qué se equivoca y por qué marcó un caso como grave.
- **Tipos de comunas (clustering):** grupos de comunas con un perfil de siniestralidad parecido.
- **Gravedad por comuna (regresión):** qué características de una comuna van de la mano con siniestros más graves, con un simulador.

Proyecto del curso Minería de Datos, Universidad de Tarapacá. Lukas Torres y Cristian Gutiérrez.

## Datos

CONASET, *Siniestros individuales 2020–2025*, a partir de los partes de Carabineros de Chile (https://mapas-conaset.opendata.arcgis.com/). La información es referencial. Los resultados muestran asociaciones, no causas.

## Estructura

| Ruta | Contenido |
|---|---|
| `app.py` | Aplicación Streamlit |
| `resultados/` | Resultados de los algoritmos que usa la app: reglas, predicciones, métricas, grupos de comunas, modelo entrenado, importancia de variables y valores SHAP |
| `pipeline/01_descarga.py` | Descarga las bases desde el servicio ArcGIS de CONASET |
| `pipeline/02_limpieza.py` | Limpia la base de siniestros → `data/processed/siniestros_limpio.csv` |
| `pipeline/03_modelos.py` | Reglas de asociación (FP-Growth), clasificación (gradient boosting) y clustering (K-Means) → `resultados/` |
| `pipeline/04_visualizaciones.py` | Figuras PNG y vistas HTML → `visualizaciones/`. También agrega a `comunas_clusters.csv` el nombre de cada grupo, que usa la app |
| `pipeline/05_regresion.py` | Regresión lineal por comuna (scikit-learn y Yellowbrick) → `resultados/regresion_*` y figuras |

Las carpetas `data/` y `visualizaciones/` no se suben al repositorio: la base limpia pesa 128 MB, y ambas se regeneran con los scripts.

## Ver la aplicación

Requiere Python 3.13. La carpeta `resultados/` ya trae todo lo necesario.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reproducir los resultados

Desde la raíz del repositorio y en este orden (`04` necesita lo que genera `03`, y `05` lo que genera `04`):

```bash
pip install -r requirements-pipeline.txt
python pipeline/01_descarga.py siniestros
python pipeline/02_limpieza.py
python pipeline/03_modelos.py
python pipeline/04_visualizaciones.py
python pipeline/05_regresion.py
```

La descarga tarda unos 10 minutos y `03_modelos.py` unos 3. Los algoritmos usan una semilla fija (42), así que los resultados coinciden con los de `resultados/`.
