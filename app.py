
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Predicción de Éxito de Películas",
    page_icon="🎬",
    layout="wide"
)

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "modelo_blockbuster.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features_modelo.pkl")
DATA_PATH = os.path.join(BASE_DIR, "global_movies_dataset_1950_2026_app.csv")
MODELS_PATH = os.path.join(BASE_DIR, "comparacion_modelos.csv")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "feature_importance.csv")
METRICS_PATH = os.path.join(BASE_DIR, "metricas_app_modelo.csv")
CONFUSION_PATH = os.path.join(BASE_DIR, "matriz_confusion_app.csv")

# ============================================================
# CARGA DE RECURSOS
# ============================================================

@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return modelo, features

@st.cache_data
def cargar_datos():
    df = pd.read_csv(DATA_PATH)
    comparacion = pd.read_csv(MODELS_PATH)
    importance = pd.read_csv(IMPORTANCE_PATH)
    metricas = pd.read_csv(METRICS_PATH)
    matriz = pd.read_csv(CONFUSION_PATH)
    return df, comparacion, importance, metricas, matriz

modelo, features = cargar_modelo()
df, comparacion_modelos, feature_importance, metricas_modelo, matriz_confusion = cargar_datos()

# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #FAFAFA;
    }
    .stMetric {
        background-color: #161B22;
        padding: 15px;
        border-radius: 15px;
        border: 1px solid #30363D;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎬 Movie Success App")
st.sidebar.write("Modelo predictivo del éxito comercial de películas.")

pagina = st.sidebar.radio(
    "Menú",
    [
        "Inicio",
        "Dashboard EDA",
        "Predicción",
        "Machine Learning",
        "Caso de prueba",
        "Conclusiones"
    ]
)

# ============================================================
# PÁGINA 1: INICIO
# ============================================================

if pagina == "Inicio":
    st.title("🎬 Predicción de Éxito Comercial de Películas")
    st.write(
        """
        Esta aplicación presenta una solución de analítica predictiva desarrollada para estimar
        si una película podría alcanzar la categoría de **blockbuster** utilizando técnicas de
        Machine Learning.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de películas", f"{len(df):,}")
    col2.metric("Blockbusters", f"{int(df['blockbuster_flag'].sum()):,}")
    col3.metric("Modelo ganador", "Gradient Boosting")

    col4, col5, col6 = st.columns(3)
    col4.metric("Accuracy", "82.7%")
    col5.metric("ROC-AUC", "0.866")
    col6.metric("F1-score", "0.560")

    st.subheader("Objetivo del proyecto")
    st.write(
        """
        El objetivo es apoyar la toma de decisiones en la industria cinematográfica mediante
        una herramienta capaz de estimar la probabilidad de éxito comercial de una película
        antes de su lanzamiento o distribución.
        """
    )

    st.subheader("Flujo de la solución")
    st.write(
        """
        **Dataset → Limpieza → EDA → Transformación → Modelado → Evaluación → Streamlit**
        """
    )

# ============================================================
# PÁGINA 2: DASHBOARD EDA
# ============================================================

elif pagina == "Dashboard EDA":
    st.title("📊 Dashboard EDA")

    st.write(
        """
        En esta sección se presentan visualizaciones exploratorias para comprender el comportamiento
        de las películas según género, presupuesto, ingresos, plataforma y éxito comercial.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Películas", f"{len(df):,}")
    col2.metric("Blockbusters", f"{int(df['blockbuster_flag'].sum()):,}")
    col3.metric("Rating IMDb promedio", f"{df['imdb_rating'].mean():.2f}")
    col4.metric("Budget promedio", f"${df['budget_million'].mean():.1f}M")

    st.subheader("Insight 1: Géneros con mayor presencia")
    genre_counts = df["genre"].astype(str).str.split("|").explode().str.strip().value_counts().head(10).reset_index()
    genre_counts.columns = ["genre", "cantidad"]

    fig_genre = px.bar(
        genre_counts,
        x="cantidad",
        y="genre",
        orientation="h",
        title="Top 10 géneros más frecuentes",
        labels={"cantidad": "Cantidad de películas", "genre": "Género"}
    )
    st.plotly_chart(fig_genre, use_container_width=True)

    st.info(
        "Los géneros con mayor frecuencia permiten identificar las categorías cinematográficas más representativas del dataset."
    )

    st.subheader("Insight 2: Relación entre presupuesto e ingresos")
    sample_df = df.sample(min(5000, len(df)), random_state=42)

    fig_scatter = px.scatter(
        sample_df,
        x="budget_million",
        y="revenue_million",
        color=sample_df["blockbuster_flag"].map({0: "No blockbuster", 1: "Blockbuster"}),
        size="popularity_score",
        hover_data=["title", "genre"],
        title="Presupuesto vs ingresos",
        labels={
            "budget_million": "Presupuesto (millones)",
            "revenue_million": "Ingresos (millones)",
            "color": "Tipo"
        }
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.info(
        "Se observa que las películas con mayor presupuesto suelen presentar mayores ingresos, aunque el presupuesto por sí solo no garantiza el éxito comercial."
    )

    st.subheader("Insight 3: Evolución de blockbusters por año")
    yearly = df.groupby("release_year")["blockbuster_flag"].sum().reset_index()

    fig_year = px.line(
        yearly,
        x="release_year",
        y="blockbuster_flag",
        title="Cantidad de blockbusters por año",
        labels={"release_year": "Año", "blockbuster_flag": "Cantidad de blockbusters"}
    )
    st.plotly_chart(fig_year, use_container_width=True)

    st.info(
        "La evolución temporal permite identificar periodos donde se incrementó la producción de películas con alto éxito comercial."
    )

    st.subheader("Distribución por plataforma de streaming")
    platform_counts = df["streaming_platform"].value_counts().head(10).reset_index()
    platform_counts.columns = ["platform", "cantidad"]

    fig_platform = px.bar(
        platform_counts,
        x="platform",
        y="cantidad",
        title="Distribución de películas por plataforma",
        labels={"platform": "Plataforma", "cantidad": "Cantidad"}
    )
    st.plotly_chart(fig_platform, use_container_width=True)

# ============================================================
# PÁGINA 3: PREDICCIÓN
# ============================================================

elif pagina == "Predicción":
    st.title("🤖 Predicción interactiva")

    st.write(
        """
        Ingrese las características de una película para estimar la probabilidad de que sea
        clasificada como blockbuster.
        """
    )

    st.sidebar.header("Datos de entrada")

    inputs = {}

    for col in features:
        if col == "release_year":
            inputs[col] = st.sidebar.number_input("Año de estreno", min_value=1950, max_value=2026, value=2024)
        elif col == "decade":
            inputs[col] = st.sidebar.selectbox("Década", [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020], index=7)
        elif col == "runtime_min":
            inputs[col] = st.sidebar.number_input("Duración (min)", min_value=60, max_value=240, value=120)
        elif col == "genre_count":
            inputs[col] = st.sidebar.number_input("Cantidad de géneros", min_value=1, max_value=5, value=2)
        elif col == "budget_million":
            inputs[col] = st.sidebar.number_input("Presupuesto (millones)", min_value=0.0, max_value=300.0, value=100.0)
        elif col == "marketing_budget_million":
            inputs[col] = st.sidebar.number_input("Marketing (millones)", min_value=0.0, max_value=200.0, value=60.0)
        elif col == "award_nominations":
            inputs[col] = st.sidebar.number_input("Nominaciones", min_value=0, max_value=100, value=4)
        elif col == "award_wins":
            inputs[col] = st.sidebar.number_input("Premios ganados", min_value=0, max_value=50, value=1)
        elif col == "franchise_flag":
            inputs[col] = st.sidebar.selectbox("¿Pertenece a franquicia?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        elif col == "genre":
            inputs[col] = st.sidebar.selectbox("Género", ["Action", "Adventure", "Comedy", "Drama", "Thriller", "Horror", "Animation", "Other"])
        elif col == "subgenre":
            inputs[col] = st.sidebar.selectbox("Subgénero", ["Superhero", "Sci-Fi", "Comedy Animation", "Crime", "Romance", "Other"])
        elif col == "country":
            inputs[col] = st.sidebar.selectbox("País", ["United States", "United Kingdom", "India", "France", "Japan", "Other"])
        elif col == "language":
            inputs[col] = st.sidebar.selectbox("Idioma", ["English", "Spanish", "French", "Hindi", "Japanese", "Other"])
        elif col == "streaming_platform":
            inputs[col] = st.sidebar.selectbox("Plataforma", ["Cinema", "Netflix", "Disney+", "Prime Video", "Apple TV+", "Other"])
        elif col in ["imdb_rating", "metascore", "audience_score", "popularity_score", "votes"]:
            if col == "imdb_rating":
                inputs[col] = st.sidebar.slider("IMDb Rating", 0.0, 10.0, 7.0)
            elif col == "metascore":
                inputs[col] = st.sidebar.slider("Metascore", 0, 100, 65)
            elif col == "audience_score":
                inputs[col] = st.sidebar.slider("Audience Score", 0, 100, 70)
            elif col == "popularity_score":
                inputs[col] = st.sidebar.slider("Popularity Score", 0.0, 100.0, 60.0)
            elif col == "votes":
                inputs[col] = st.sidebar.number_input("Votes", min_value=0, max_value=2000000, value=50000)
        else:
            inputs[col] = st.sidebar.text_input(col, "Other")

    caso = pd.DataFrame([inputs])
    caso = caso[features]

    st.subheader("Datos ingresados")
    st.dataframe(caso)

    if st.button("Predecir éxito comercial"):
        prob = modelo.predict_proba(caso)[:, 1][0]
        pred = int(prob >= 0.50)

        col1, col2 = st.columns(2)
        col1.metric("Probabilidad de blockbuster", f"{prob:.2%}")
        col2.metric("Clasificación", "Blockbuster" if pred == 1 else "No blockbuster")

        st.progress(float(prob))

        if pred == 1:
            st.success("Resultado: La película presenta alta probabilidad de ser blockbuster.")
        else:
            st.warning("Resultado: La película no presenta alta probabilidad de ser blockbuster.")

        st.write(
            """
            Esta predicción puede apoyar decisiones relacionadas con inversión, marketing,
            selección de proyectos y planificación de lanzamiento.
            """
        )

# ============================================================
# PÁGINA 4: MACHINE LEARNING
# ============================================================

elif pagina == "Machine Learning":
    st.title("📈 Resultados de Machine Learning")

    st.subheader("Comparación de modelos")
    st.dataframe(comparacion_modelos)

    fig_models = px.bar(
        comparacion_modelos,
        x="modelo",
        y=["accuracy", "precision", "recall", "f1_score", "roc_auc"],
        barmode="group",
        title="Comparación de modelos por métricas"
    )
    st.plotly_chart(fig_models, use_container_width=True)

    st.info(
        "Gradient Boosting fue seleccionado como modelo final debido a que obtuvo el mejor ROC-AUC y un adecuado equilibrio entre las métricas evaluadas."
    )

    st.subheader("Métricas del modelo ganador")
    fig_metrics = px.bar(
        metricas_modelo,
        x="metrica",
        y="valor",
        title="Métricas del modelo Gradient Boosting",
        text="valor"
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    st.subheader("Importancia de variables")
    top_importance = feature_importance.sort_values(by=feature_importance.columns[-1], ascending=False).head(15)

    fig_importance = px.bar(
        top_importance,
        x=top_importance.columns[-1],
        y=top_importance.columns[0],
        orientation="h",
        title="Top 15 variables más importantes"
    )
    st.plotly_chart(fig_importance, use_container_width=True)

    st.info(
        "La importancia de variables permite identificar qué factores influyen más en la predicción del éxito comercial."
    )

    st.subheader("Matriz de confusión")
    st.dataframe(matriz_confusion)

# ============================================================
# PÁGINA 5: CASO DE PRUEBA
# ============================================================

elif pagina == "Caso de prueba":
    st.title("🎯 Caso de prueba")

    st.write(
        """
        Caso simulado: película de acción con alto presupuesto, inversión relevante en marketing
        y pertenencia a franquicia.
        """
    )

    caso_demo = {}
    for col in features:
        if col == "release_year":
            caso_demo[col] = 2025
        elif col == "decade":
            caso_demo[col] = 2020
        elif col == "runtime_min":
            caso_demo[col] = 125
        elif col == "genre_count":
            caso_demo[col] = 2
        elif col == "budget_million":
            caso_demo[col] = 130.0
        elif col == "marketing_budget_million":
            caso_demo[col] = 80.0
        elif col == "award_nominations":
            caso_demo[col] = 5
        elif col == "award_wins":
            caso_demo[col] = 2
        elif col == "franchise_flag":
            caso_demo[col] = 1
        elif col == "genre":
            caso_demo[col] = "Action"
        elif col == "subgenre":
            caso_demo[col] = "Superhero"
        elif col == "country":
            caso_demo[col] = "United States"
        elif col == "language":
            caso_demo[col] = "English"
        elif col == "streaming_platform":
            caso_demo[col] = "Cinema"
        elif col == "imdb_rating":
            caso_demo[col] = 7.5
        elif col == "metascore":
            caso_demo[col] = 70
        elif col == "audience_score":
            caso_demo[col] = 78
        elif col == "popularity_score":
            caso_demo[col] = 75
        elif col == "votes":
            caso_demo[col] = 100000
        else:
            caso_demo[col] = "Other"

    df_demo = pd.DataFrame([caso_demo])[features]
    st.dataframe(df_demo)

    prob = modelo.predict_proba(df_demo)[:, 1][0]
    pred = int(prob >= 0.50)

    st.metric("Probabilidad estimada", f"{prob:.2%}")

    if pred == 1:
        st.success("El modelo predice que la película tiene alta probabilidad de ser blockbuster.")
    else:
        st.warning("El modelo predice que la película no tendría alta probabilidad de ser blockbuster.")

    st.write(
        """
        Este caso permite demostrar cómo la aplicación puede utilizarse para simular escenarios
        antes de invertir en una producción cinematográfica.
        """
    )

# ============================================================
# PÁGINA 6: CONCLUSIONES
# ============================================================

elif pagina == "Conclusiones":
    st.title("📌 Conclusiones")

    st.write(
        """
        El proyecto permitió construir una solución integral de analítica predictiva para estimar
        el éxito comercial de películas mediante Machine Learning.
        """
    )

    st.markdown(
        """
        - Gradient Boosting fue seleccionado como modelo final por su mejor desempeño en ROC-AUC.
        - El análisis exploratorio permitió identificar relaciones entre presupuesto, ingresos, género y éxito comercial.
        - Streamlit permitió desplegar el modelo en una aplicación web interactiva.
        - La solución puede apoyar decisiones relacionadas con inversión, marketing y planificación cinematográfica.
        """
    )
