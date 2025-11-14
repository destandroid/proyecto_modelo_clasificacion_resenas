import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
import numpy as np
from collections import Counter
from pathlib import Path
import re
from transformers import pipeline  

# ------------------------------------------------
# Configuración de la página
# ------------------------------------------------
st.set_page_config(page_title="Dashboard Trustpilot", layout="wide")
st.title("📊 Dashboard de Reseñas - Trustpilot")

# CSS estilo Material-like para cards
st.markdown("""
<style>
.md-card {
  background: #ffffff;
  border-radius: 14px;
  padding: 18px 18px 6px 18px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.08);
  border: 1px solid rgba(0,0,0,0.04);
}
.md-card h3 {
  margin-top: 0; margin-bottom: 10px;
  font-weight: 600;
  font-size: 1.05rem;
  color: #374151;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Stopwords ES
# ------------------------------------------------
try:
    stop_words = set(stopwords.words("spanish"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("spanish"))

punctuation = set(string.punctuation)

# ------------------------------------------------
# Utilidades
# ------------------------------------------------
def _safe_get(d, *names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def _aplanar_json_empresas(data):
    rows = []
    for emp in data:
        nombre = _safe_get(emp, "nombre", "Nombre")
        categoria = _safe_get(emp, "categoria", "Categoría", "Categoria")
        ubicacion = _safe_get(emp, "ubicacion", "ubicación", "Ubicacion", "Ubicación")
        puntuacion_empresa = _safe_get(emp, "puntuacion", "Puntuación", "rating", "score")
        pagina_web = _safe_get(emp, "pagina_web", "web", "sitio", "url")

        resenas = _safe_get(emp, "reseñas", "resenas", "reviews", default=[])
        for r in resenas:
            rows.append({
                "nombre": nombre,
                "nombre_categoria": categoria,
                "ubicacion": ubicacion,
                "puntuacion_empresa": puntuacion_empresa,
                "pagina_web": pagina_web,
                "id_resena": _safe_get(r, "id_reseña", "id_resena", "id"),
                "titulo": _safe_get(r, "Título", "titulo", "title", default=""),
                "contenido": _safe_get(r, "Contenido", "contenido", "text", default=""),
                "calificacion": _safe_get(r, "Calificación", "calificacion", "rating", default=None),
                "fecha_local": _safe_get(r, "fecha_local", "Fecha_local", "fecha", "date"),
                "hora_local": _safe_get(r, "hora_local", "Hora_local", "hora", "time")
            })
    df = pd.DataFrame(rows)

    if not df.empty:
        df["fecha_local"] = pd.to_datetime(df["fecha_local"], errors="coerce")
        df["hora_local"] = pd.to_datetime(df["hora_local"], errors="coerce").dt.time

        df["momento"] = pd.to_datetime(
            df["fecha_local"].dt.strftime("%Y-%m-%d") + " " +
            df["hora_local"].fillna(pd.to_datetime("00:00:00").time()).astype(str),
            errors="coerce"
        )

        df["anio"] = df["fecha_local"].dt.year
        df["mes"] = df["fecha_local"].dt.month
        df["anio_mes"] = df["fecha_local"].dt.to_period("M").astype(str)
        df["calificacion"] = pd.to_numeric(df["calificacion"], errors="coerce")
    return df

# ==========================================================
# ANÁLISIS DE SENTIMIENTO - Modelo HF (FelipeV)
# ==========================================================
@st.cache_resource(show_spinner=True)
def load_sentiment_model():
    """
    Carga el modelo de sentimiento solo una vez.
    Modelo: FelipeV/bert-base-spanish-uncased-sentiment
    """
    return pipeline(
        "sentiment-analysis",
        model="FelipeV/bert-base-spanish-uncased-sentiment",
        tokenizer="FelipeV/bert-base-spanish-uncased-sentiment"
    )


def analizar_sentimiento_model(titulo: str, contenido: str):
    """
    Usa el modelo de HuggingFace para obtener:
      - etiqueta: 'positivo' / 'negativo' / 'neutral'
      - score: 0 a 100 (para el gauge)
    """
    texto = f"{titulo or ''} {contenido or ''}".strip()
    if not texto:
        return "neutral", 50.0

    clf = load_sentiment_model()
    result = clf(texto)[0]          # {'label': 'POS'/'NEG'/'NEU'?, 'score': ...}
    raw = result["label"]
    prob = float(result["score"])

    # Mapear a nuestras 3 clases y al rango 0–100
    raw_upper = raw.upper()
    if raw_upper in ("POS", "POSITIVE", "LABEL_2"):
        label = "positivo"
        score = 50 + prob * 50       # 50–100
    elif raw_upper in ("NEG", "NEGATIVE", "LABEL_0"):
        label = "negativo"
        score = 50 - prob * 50       # 0–50
    else:
        # NEU, neutral o label intermedio
        label = "neutral"
        score = 50.0

    return label, float(round(score, 2))


def aplicar_sentimiento(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el análisis de sentimiento a todo el DataFrame.
    Crea:
      - 'sentimiento'
      - 'sentiment_score'
    """
    if df_in.empty:
        return df_in.copy()

    df_out = df_in.copy()
    etiquetas = []
    puntajes = []

    titulos = df_out["titulo"].fillna("")
    contenidos = df_out["contenido"].fillna("")

    for t, c in zip(titulos, contenidos):
        lab, sc = analizar_sentimiento_model(t, c)
        etiquetas.append(lab)
        puntajes.append(sc)

    df_out["sentimiento"] = etiquetas
    df_out["sentiment_score"] = puntajes
    return df_out

# ==========================================================
# LIMPIEZA DE TEXTO Y NUBE DE PALABRAS
# ==========================================================
def _texto_limpio_desde_df(df_in: pd.DataFrame) -> list[str]:
    textos = df_in["titulo"].fillna('') + " " + df_in["contenido"].fillna('')
    texto = " ".join(textos).lower()

    texto = (
        texto.replace("á", "a").replace("é", "e")
             .replace("í", "i").replace("ó", "o")
             .replace("ú", "u").replace("ñ", "n")
    )

    palabras = [
        p.strip("".join(punctuation))
        for p in texto.split()
        if p not in stop_words and p.isalpha() and len(p) > 2
    ]
    return palabras


def generar_wordcloud_desde_df(df_in: pd.DataFrame) -> WordCloud | None:
    palabras = _texto_limpio_desde_df(df_in)
    if not palabras:
        return None

    texto_wc = " ".join(palabras)

    wc = WordCloud(
        width=900,
        height=350,
        background_color="white",
        collocations=False,
        colormap="GnBu"
    ).generate(texto_wc)

    return wc


def top_palabras_df(df_in: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    palabras = _texto_limpio_desde_df(df_in)
    if not palabras:
        return pd.DataFrame(columns=["Palabra", "Frecuencia"])

    conteo = Counter(palabras).most_common(top_n)
    return pd.DataFrame(conteo, columns=["Palabra", "Frecuencia"])

# ==========================================================
# CARGA DE DATOS SIEMPRE DEL JSON FIJO
# ==========================================================
RUTA_JSON_FIJO = Path("datalake/2_REFINED_ZONE/dataset_reviews_limpio.json")

@st.cache_data(show_spinner=True)
def cargar_datos_json_fijo():
    if not RUTA_JSON_FIJO.exists():
        return pd.DataFrame()
    with open(RUTA_JSON_FIJO, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for _, v in data.items():
            if isinstance(v, list):
                data = v
                break
    df0 = _aplanar_json_empresas(data)
    return aplicar_sentimiento(df0)


df = cargar_datos_json_fijo()
if df.empty:
    st.error(f"No se encontraron datos en: {RUTA_JSON_FIJO}")
    st.stop()

# ==========================================================
# Última reseña
# ==========================================================
orden_col = "momento" if "momento" in df.columns else "fecha_local"
ultima_fila = df.sort_values(by=[orden_col], ascending=False).iloc[0]
fecha_formateada = pd.to_datetime(ultima_fila["fecha_local"]).strftime("%d/%m/%Y") \
    if pd.notnull(ultima_fila["fecha_local"]) else "--/--/----"

if pd.notnull(ultima_fila.get("hora_local")):
    try:
        hora_formateada = ultima_fila["hora_local"].strftime("%H:%M")
    except Exception:
        hora_formateada = str(ultima_fila["hora_local"])[:5]
else:
    hora_formateada = "--:--"

st.markdown(
    f"<div style='text-align: right; font-size: 16px;'>📅 Última reseña: "
    f"<strong>{fecha_formateada}</strong> a las <strong>{hora_formateada}</strong></div>",
    unsafe_allow_html=True
)

# ==========================================================
# Filtros
# ==========================================================
categoria_sel = st.sidebar.selectbox(
    "Categoría",
    ["Todas"] + sorted(x for x in df["nombre_categoria"].dropna().astype(str).unique())
)
df_f = df.copy()
if categoria_sel != "Todas":
    df_f = df_f[df_f["nombre_categoria"].astype(str) == categoria_sel]

empresas_disponibles = sorted(df_f["nombre"].dropna().astype(str).unique())
empresa_sel = st.sidebar.selectbox("Empresa", ["Todas"] + empresas_disponibles)
if empresa_sel != "Todas":
    df_f = df_f[df_f["nombre"].astype(str) == empresa_sel]

# Año y mes
anios_disponibles = sorted([int(a) for a in df_f["anio"].dropna().unique()], reverse=True)
lista_anios = ["Todos"] + [str(a) for a in anios_disponibles]
index_por_defecto = lista_anios.index("2025") if "2025" in lista_anios else 0
anio_sel = st.sidebar.selectbox("Año", lista_anios, index=index_por_defecto)

meses_dict = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

if anio_sel != "Todos":
    df_f = df_f[df_f["anio"] == int(anio_sel)]
    meses_disponibles = sorted(df_f["mes"].dropna().unique())
    opciones_meses = ["Todos"] + [meses_dict[m] for m in meses_disponibles]
    mes_nombre_sel = st.sidebar.selectbox("Mes", opciones_meses)
    if mes_nombre_sel != "Todos":
        mes_sel = [k for k, v in meses_dict.items() if v == mes_nombre_sel][0]
        df_f = df_f[df_f["mes"] == mes_sel]

# ==========================================================
# KPIs
# ==========================================================
col1, col2, col3 = st.columns(3)
col1.metric("Empresas", df_f["nombre"].nunique())
col2.metric("Total Reseñas", df_f.shape[0])
col3.metric("Prom. Calificación", f"{df_f['calificacion'].mean():.2f}" if df_f.shape[0] else "0.00")

# ==========================================================
# FILA 1: Nube de palabras + Indicador de Sentimiento
# ==========================================================
col_words, col_gauge = st.columns(2)

with col_words:
    st.markdown("### 🧠 Nube de palabras")
    st.markdown('<div class="md-card">', unsafe_allow_html=True)

    if df_f.empty:
        st.info("No hay texto disponible para generar la nube de palabras con los filtros actuales.")
    else:
        df_wc = df_f.copy()
        wc = generar_wordcloud_desde_df(df_wc)

        if wc is None:
            st.info("No hay suficientes palabras para generar la nube.")
        else:
            st.image(wc.to_array(), use_container_width=True)

            df_top = top_palabras_df(df_wc, top_n=10)
            if not df_top.empty:
                min_f = int(df_top["Frecuencia"].min())
                max_f = int(df_top["Frecuencia"].max())

                st.markdown(
                    "<div style='text-align:center; font-size:13px; margin-top:8px;'>"
                    "Número de menciones"
                    "</div>",
                    unsafe_allow_html=True
                )

                niveles = 7
                valores = np.linspace(min_f, max_f, niveles)
                matriz = np.array([valores])

                fig_scale = px.imshow(
                    matriz,
                    aspect="auto",
                    color_continuous_scale="GnBu"
                )
                fig_scale.update_xaxes(visible=False)
                fig_scale.update_yaxes(visible=False)
                fig_scale.update_layout(
                    height=40,
                    margin=dict(l=40, r=40, t=0, b=0),
                    coloraxis_showscale=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(fig_scale, use_container_width=True)

                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:space-between; 
                                align-items:flex-end; font-size:11px; margin:4px 40px 0 40px;'>
                        <div style='text-align:left;'>
                            Menos<br><b>{min_f}</b>
                        </div>
                        <div style='text-align:right;'>
                            Más<br><b>{max_f}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown('</div>', unsafe_allow_html=True)

with col_gauge:
    if df_f.shape[0]:
        avg_sent = float(df_f["sentiment_score"].mean()) if "sentiment_score" in df_f.columns else 50.0
        label_txt = (
            "Muy negativo" if avg_sent < 20 else
            "Negativo"     if avg_sent < 40 else
            "Neutral"      if avg_sent < 60 else
            "Positivo"     if avg_sent < 80 else
            "Muy positivo"
        )
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_sent,
            number={'suffix': " / 100"},
            title={'text': f"Sentimiento promedio: {label_txt}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'thickness': 0.3},
                'shape': 'angular',
                'steps': [
                    {'range': [0, 20],  'color': '#ff4d4f'},
                    {'range': [20, 40], 'color': '#ff9f7f'},
                    {'range': [40, 60], 'color': '#ffd666'},
                    {'range': [60, 80], 'color': '#95de64'},
                    {'range': [80,100], 'color': '#52c41a'},
                ]
            }
        ))
        fig_gauge.update_layout(height=320, margin=dict(t=60, b=0), paper_bgcolor="white", plot_bgcolor="white")
        st.markdown('<div class="md-card"><h3>🧭 Indicador de Sentimiento</h3>', unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# FILA 2: Tabla de Reseñas + Distribución de calificaciones
# ==========================================================
col_table, col_dist = st.columns(2)

with col_table:
    st.markdown("### 🗂️ Reseñas")
    df_tabla = df_f.copy()
    if df_tabla.shape[0]:
        df_tabla["fecha_local"] = pd.to_datetime(df_tabla["fecha_local"], errors="coerce").dt.strftime("%d/%m/%Y")
    cols = ["nombre", "sentimiento", "titulo", "contenido", "calificacion", "fecha_local", "hora_local"]
    cols_presentes = [c for c in cols if c in df_tabla.columns]
    st.dataframe(df_tabla[cols_presentes])

with col_dist:
    st.markdown("### 📊 Distribución de calificaciones")
    colores = {"1": "#d73027", "2": "#fc8d59", "3": "#fee08b", "4": "#91cf60", "5": "#1a9850"}

    if df_f.shape[0]:
        if empresa_sel != "Todas":
            serie = df_f["calificacion"].dropna().astype(int)
            df_dona = (
                serie.value_counts()
                     .rename_axis("calificacion")
                     .reset_index(name="cantidad")
            )
            df_dona["calificacion"] = df_dona["calificacion"].astype(str)
            df_dona = df_dona.sort_values("calificacion", ascending=False)

            fig_dona = px.pie(
                df_dona,
                names="calificacion",
                values="cantidad",
                title=f"Calificaciones - {empresa_sel}",
                hole=0.6,
                color="calificacion",
                color_discrete_map=colores
            )
            fig_dona.update_traces(textposition="inside", textinfo="percent+label")
            fig_dona.update_layout(showlegend=False, paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig_dona, use_container_width=True)

        else:
            df_dist = df_f.groupby(["nombre", "calificacion"]).size().reset_index(name="cantidad")
            df_total = df_dist.groupby("nombre")["cantidad"].transform("sum")
            df_dist["porcentaje"] = df_dist["cantidad"] / df_total * 100
            df_dist["calificacion"] = df_dist["calificacion"].astype("Int64").astype(str)
            df_dist["calificacion"] = pd.Categorical(
                df_dist["calificacion"],
                categories=["5", "4", "3", "2", "1"],
                ordered=True
            )
            fig = px.bar(
                df_dist,
                x="nombre",
                y="porcentaje",
                color="calificacion",
                title="Distribución porcentual de calificaciones por empresa",
                color_discrete_map=colores,
                barmode="stack",
                labels={"nombre": "Empresa", "porcentaje": "% Reseñas", "calificacion": "Calificación"}
            )
            fig.update_layout(yaxis_tickformat=".0f", yaxis_range=[0, 100], paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# FILA INFERIOR: Número de reseñas por día
# ==========================================================
if df_f["fecha_local"].notna().any():
    df_dias = (
        df_f.assign(anio=lambda x: x["fecha_local"].dt.year)
            .groupby(["anio", "fecha_local"])
            .size()
            .reset_index(name="cantidad")
    )
    fig_dias = px.bar(
        df_dias,
        x="fecha_local",
        y="cantidad",
        facet_col="anio",
        facet_col_wrap=3,
        title="📅 Número de reseñas por día (faceteado por año)",
        labels={"fecha_local": "Fecha", "cantidad": "Cantidad de reseñas", "anio": "Año"}
    )
    fig_dias.for_each_xaxis(lambda a: a.update(tickformat="%d/%m"))
    fig_dias.update_layout(height=520, margin=dict(t=60), paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig_dias, use_container_width=True)

    # ----------------------------------------------------------
    # BARRA APILADA: número de calificaciones por día (1–5)
    # ----------------------------------------------------------
    if df_f["calificacion"].notna().any():
        df_cal_dia = (
            df_f.dropna(subset=["fecha_local", "calificacion"])
                .assign(
                    calificacion=lambda x: x["calificacion"].astype("Int64"),
                    fecha_local=lambda x: x["fecha_local"].dt.normalize()  # solo fecha
                )
                .groupby(["fecha_local", "calificacion"])
                .size()
                .reset_index(name="cantidad")
        )

        if not df_cal_dia.empty:
            # Colores consistentes con el resto del dashboard
            colores_dias = {"1": "#d73027", "2": "#fc8d59", "3": "#fee08b", "4": "#91cf60", "5": "#1a9850"}

            # Convertir calificación a categoría ordenada
            df_cal_dia["calificacion"] = df_cal_dia["calificacion"].astype(int).astype(str)
            df_cal_dia["calificacion"] = pd.Categorical(
                df_cal_dia["calificacion"],
                categories=["5", "4", "3", "2", "1"],
                ordered=True
            )

            # Calcular total de reseñas por día
            df_totales = (
                df_cal_dia.groupby("fecha_local")["cantidad"]
                .sum()
                .reset_index(name="total_reseñas")
            )

            # Unir para mostrar en tooltip
            df_cal_dia = df_cal_dia.merge(df_totales, on="fecha_local", how="left")

            fig_cal_dia = px.bar(
                df_cal_dia,
                x="fecha_local",
                y="cantidad",
                color="calificacion",
                title="📊 Calificaciones por día (barra apilada por nota)",
                labels={
                    "fecha_local": "Fecha",
                    "cantidad": "Cantidad de calificaciones",
                    "calificacion": "Nota",
                    "total_reseñas": "Cantidad de reseñas"
                },
                color_discrete_map=colores_dias,
                barmode="stack",
                hover_data=["total_reseñas"]
            )

            fig_cal_dia.for_each_xaxis(lambda a: a.update(tickformat="%d/%m"))
            fig_cal_dia.update_layout(
                height=520,
                margin=dict(t=60),
                paper_bgcolor="white",
                plot_bgcolor="white"
            )

            st.plotly_chart(fig_cal_dia, use_container_width=True)
