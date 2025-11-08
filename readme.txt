README - Proyecto Trustpilot ETL + Dashboard
============================================

Requisitos del sistema
----------------------
- Sistema operativo: Linux Debian o similar
- Python 3.13.0
- VS Code (opcional)
- Jupyter Notebook
- Google Chrome (instalado)

Instalación de librerías
------------------------
Ejecutar en terminal:

pip install pandas selenium plotly streamlit sqlalchemy psycopg2-binary webdriver-manager jupyter nltk pytz

Descargar stopwords de NLTK en Python:

import nltk
nltk.download("stopwords")

Ejecución del ETL
-----------------
Dar permisos y ejecutar:

chmod +x run_notebook.sh
./run_notebook.sh

Ejecución del dashboard streamlit
-----------------------
Desde la raíz del proyecto:

streamlit run datalake/3_CONSUMPTION_ZONE/app.py

Estructura del proyecto
-----------------------
- datalake/
  - 1_LANDING_ZONE/         → Datos crudos extraídos de Trustpilot
  - 2_REFINED_ZONE/         → Datos limpios y transformados
  - 3_CONSUMPTION_ZONE/     → Código del dashboard (app.py)

Notas
-----
- El dashboard permite filtrar por categoría, empresa, año y mes.
- La estructura de carpetas sigue el esquema de Data Lake por requerimiento de la tarea.
