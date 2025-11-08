import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import nltk
from nltk.corpus import stopwords
import string
from collections import Counter
from pathlib import Path
import re

# -----------------------------
# Configuración de la página
# -----------------------------
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

# -----------------------------
# Stopwords ES
# -----------------------------
try:
    stop_words = set(stopwords.words("spanish"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("spanish"))

punctuation = set(string.punctuation)

# -----------------------------
# Utilidades
# -----------------------------
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

# -----------------------------
# Sentiment (léxico simple ES)
# -----------------------------
_POS_WORDS = {
    "bueno","excelente","recomendado","recomiendo","rapido","rápido","claro","facil","fácil",
    "genial","perfecto","encantado","satisfecho","amable","eficiente","cumplido","seriedad",
    "transparente","ayuda","solucion","solución","agradecido","contento","mejor","bien"
}
_NEG_WORDS = {
    "malo","pesimo","pésimo","lento","tarde","caro","engano","engaño","problema","terrible",
    "horrible","decepcion","decepción","mala","pobre","nefasta","reclamo","queja","no recomiendo",
    "nunca","peor","fallo","error","falla","duda"
}

def _normalize_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = s.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("ñ","n")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def analizar_sentimiento_row(titulo: str, contenido: str):
    txt = (_normalize_text(titulo) + " " + _normalize_text(contenido)).strip()
    if not txt:
        return "neutral", 50.0
    tokens = txt.split()
    pos = sum(1 for t in tokens if t in _POS_WORDS)
    neg = sum(1 for t in tokens if t in _NEG_WORDS)
    if pos == 0 and neg == 0:
        return "neutral", 50.0
    score = (pos - neg) / max(1, (pos + neg))  # [-1,1]
    if score > 0.2:
        label = "positivo"
    elif score < -0.2:
        label = "negativo"
    else:
        label = "neutral"
    score_0_100 = (score + 1) / 2 * 100.0
    return label, score_0_100

def aplicar_sentimiento(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in.empty:
        return df_in.copy()
    df_out = df_in.copy()
    sent_labels, sent_scores = [], []
    for t, c in zip(df_out["titulo"].fillna(""), df_out["contenido"].fillna("")):
        lab, sc = analizar_sentimiento_row(t, c)
        sent_labels.append(lab)
        sent_scores.append(sc)
    df_out["sentimiento"] = sent_labels
    df_out["sentiment_score"] = sent_scores
    return df_out

# -----------------------------
# Carga de datos SIEMPRE del JSON fijo
# -----------------------------
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

# -----------------------------
# Última reseña
# -----------------------------
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

# -----------------------------
# Filtros
# -----------------------------
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

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Empresas", df_f["nombre"].nunique())
col2.metric("Total Reseñas", df_f.shape[0])
col3.metric("Prom. Calificación", f"{df_f['calificacion'].mean():.2f}" if df_f.shape[0] else "0.00")

# ==========================================================
# FILA 1: Palabras más frecuentes (izq) + Indicador (der)
# ==========================================================
col_words, col_gauge = st.columns(2)

with col_words:
    def obtener_top_palabras_por_empresa(df_in, top_n=3):
        resultados = []
        for empresa in df_in["nombre"].dropna().astype(str).unique():
            df_empresa = df_in[df_in["nombre"].astype(str) == empresa]
            textos = df_empresa["titulo"].fillna('') + " " + df_empresa["contenido"].fillna('')
            texto_completo = " ".join(textos).lower()
            palabras = [
                palabra.strip("".join(punctuation))
                for palabra in texto_completo.split()
                if palabra not in stop_words and palabra.isalpha()
            ]
            if not palabras:
                continue
            top_palabras = Counter(palabras).most_common(top_n)
            for palabra, frecuencia in top_palabras:
                resultados.append({"Empresa": empresa, "Palabra": palabra, "Frecuencia": frecuencia})
        return pd.DataFrame(resultados)

    st.markdown("### 🧠 Palabras más frecuentes")
    df_palabras = obtener_top_palabras_por_empresa(df_f)
    st.dataframe(df_palabras)

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
        st.plotly_chart(fig_gauge, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# FILA 2: Tabla de Reseñas (izq) + Distribución (der)
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
        # Si hay una sola empresa seleccionada -> DONA
        if empresa_sel != "Todas":
            serie = df_f["calificacion"].dropna().astype(int)
            df_dona = (serie.value_counts()
                             .rename_axis("calificacion")
                             .reset_index(name="cantidad"))
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
            st.plotly_chart(fig_dona, width="stretch")

        else:
            # Varias empresas -> barra apilada 100%
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
            st.plotly_chart(fig, width="stretch")

# ==========================================================
# FILA INFERIOR (FULL WIDTH): Número de reseñas por día
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

    st.plotly_chart(fig_dias, width="stretch")
