# ¿Qué hace que un siniestro de tránsito termine en muerte?

Aplicación web que muestra, en lenguaje simple, los resultados de tres técnicas de minería de datos aplicadas a 435 mil siniestros de tránsito registrados en Chile entre 2020 y 2025.

- **Condiciones de riesgo (reglas de asociación):** qué combinaciones de lugar, hora y tipo de siniestro hacen más probable una muerte o una lesión grave.
- **Predicción de gravedad (clasificación):** qué tan bien un modelo anticipa los siniestros graves, en qué se equivoca y por qué marcó un caso como grave.
- **Tipos de comunas (clustering):** grupos de comunas con un perfil de siniestralidad parecido.

Proyecto del curso Minería de Datos, Universidad de Tarapacá. Lukas Torres y Cristian Gutiérrez.

## Datos

CONASET, *Siniestros individuales 2020–2025*, a partir de los partes de Carabineros de Chile (https://mapas-conaset.opendata.arcgis.com/). La información es referencial. Los resultados muestran asociaciones, no causas.

La carpeta `resultados/` contiene lo que la aplicación necesita: las reglas encontradas, las predicciones sobre el conjunto de prueba, los grupos de comunas y el modelo entrenado.

## Ejecutar en local

Requiere Python 3.13.

```bash
pip install -r requirements.txt
streamlit run app.py
```
