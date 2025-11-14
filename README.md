
# 📘 Proyecto Procesamiento de Reseñas de Trustpilot mediante Scraping Web

Este proyecto implementa un flujo completo de **extracción, limpieza, transformación y visualización** de reseñas obtenidas desde la plataforma **Trustpilot**. La solución combina scraping automatizado, procesamiento estructurado y un dashboard interactivo orientado al análisis de la experiencia del cliente.

Incluye:  
- 🕸️ **Web Scraping** automatizado con Selenium  
- 🧹 **ETL** (limpieza, validación y normalización de datos)  
- 🗂️ **Datalake estructurado** con tres zonas  
- 📊 **Dashboard interactivo en Streamlit**  
- 🤖 **Análisis de sentimiento** usando modelos BERT en español  
- 🔍 **Visualización de tendencias, nubes de palabras y distribución de calificaciones**

---

## 📁 Estructura del proyecto (Datalake)

````
datalake/
│
├── 1_LANDING_ZONE/         # Datos crudos obtenidos por scraping
├── 2_REFINED_ZONE/         # Datos limpios y transformados (JSON final)
└── 3_CONSUMPTION_ZONE/     # Dashboard Streamlit listo para ejecución
````


---

## 🧩 Requisitos del sistema

- **Linux Debian** o similar  
- **Python 3.13.0**  
- **Google Chrome**  
- **Jupyter Notebook**  
- **VS Code** (opcional)

---

## ⚙️ Instalación de librerías

Ejecutar en terminal:

```bash
pip install pandas selenium plotly streamlit sqlalchemy psycopg2-binary webdriver-manager jupyter nltk pytz transformers wordcloud

```
Instalar stopwords (solo primera vez):

```bash
python -c "import nltk; nltk.download('stopwords')"
```

---

## 🚀 Ejecución del proyecto

### 1️⃣ Ejecutar el cuaderno ETL

Desde Jupyter o VS Code:

```bash
jupyter notebook scrapp.ipynb
```

Este cuaderno realiza:

* Extracción de reseñas mediante scraping
* Limpieza, validación y estandarización de datos
* Conversión de fechas y normalización de textos
* Generación del archivo final:

```
datalake/2_REFINED_ZONE/dataset_reviews_limpio.json
```

---

### 2️⃣ Ejecutar el dashboard

Desde la raíz del proyecto:

```bash
streamlit run datalake/3_CONSUMPTION_ZONE/app.py
```

### Funcionalidades del dashboard:

* Filtro por categoría, empresa, año y mes
* Nube de palabras dinámica
* Análisis de sentimiento automático
* Distribución de calificaciones por empresa
* Actividad diaria de reseñas
* Tabla detallada de reseñas
* KPI de volumen, promedio de calificación y sentimiento



