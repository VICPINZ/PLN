import subprocess
import sys
import os
import streamlit as st
import pandas as pd
import joblib
from docx import Document
import google.generativeai as genai
from sklearn.base import BaseEstimator, TransformerMixin
import string
from collections import Counter
from fpdf import FPDF
import spacy


# Cargar el modelo de spaCy
@st.cache_resource
def load_model():
    return spacy.load("es_core_news_md")

nlp = load_model()

# Configurar la clave de API de Gemini
genai.configure(api_key=st.secrets["GENAI_API_KEY"])

# Seleccionar el modelo de Gemini
model = genai.GenerativeModel("gemini-1.5-flash")

# Cargar modelos de clasificación
pipeline = joblib.load('modelo_rf.pkl')
le = joblib.load('label_encoder.pkl')

# Clase SpacyVectorizer (si la necesitas en el pipeline)
class SpacyVectorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [nlp(text).vector for text in X]

# Función para predecir nuevas clasificaciones
def predecir_clasificacion(textos):
    predicciones = pipeline.predict(textos)
    return le.inverse_transform(predicciones)

# Función para generar recomendaciones con Gemini
def generar_recomendaciones(comentarios):
    entrada = " ".join(comentarios)
    prompt = f"Basado en los siguientes comentarios de recomendación, sugiere mejoras concretas:\n{entrada}\n\nRecomendaciones:"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Ocurrió un error al generar las recomendaciones: {e}"

# Función para generar informe PDF
def generar_informe_pdf(resumen, recomendaciones, nombre_archivo):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Título
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Informe de Resultados de la Encuesta", ln=True, align='C')
    pdf.ln(10)

    # Resumen por clase
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Resumen por clase", ln=True)
    pdf.set_font("Arial", size=12)
    for fila in resumen:
        pdf.cell(0, 10, f"Clase: {fila['Clase']}", ln=True)
        pdf.cell(0, 10, f"Conteo: {fila['Conteo']}", ln=True)
        pdf.multi_cell(0, 10, f"Top 10 palabras: {fila['Top 10 palabras']}")
        pdf.ln(5)

    # Propuesta de Mejora
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Propuesta de Mejora", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, recomendaciones)

    # Guardar el PDF
    pdf.output(nombre_archivo)

# --- Interfaz en Streamlit ---
st.title("Análisis de Encuesta - Ministerio de Defensa")

archivo = st.file_uploader("Cargar archivo Excel", type=["xlsx"])
if archivo:
    df = pd.read_excel(archivo)
    st.write("Vista previa de datos:", df.head())

    variable = st.text_input("Ingrese el nombre de la variable de texto:")
    if st.button("Generar Informe"):
        if variable in df.columns:
            df = df.dropna(subset=[variable])
            df[variable] = df[variable].astype(str)
            df['clasificacion'] = predecir_clasificacion(df[variable])

            # Tabla resumen por clase
            stopwords = nlp.Defaults.stop_words

            resumen = []
            for clase, grupo in df.groupby('clasificacion'):
                textos = " ".join(grupo[variable].tolist()).lower()
                # Quitar puntuación
                textos = textos.translate(str.maketrans('', '', string.punctuation))
                palabras = [w for w in textos.split() if w not in stopwords and len(w) > 2]
                mas_comunes = [w for w, _ in Counter(palabras).most_common(10)]
                resumen.append({
                    'Clase': clase,
                    'Conteo': len(grupo),
                    'Top 10 palabras': ", ".join(mas_comunes)
                })
            st.subheader("Resumen por clase")
            st.dataframe(pd.DataFrame(resumen))

            # Filtrar comentarios tipo Recomendación
            recomendaciones_comentarios = df[df['clasificacion'].str.lower() == 'comentario'][variable].tolist()

            if recomendaciones_comentarios:
                recomendaciones_generadas = generar_recomendaciones(recomendaciones_comentarios)
            else:
                recomendaciones_generadas = "No se encontraron recomendaciones para analizar."

            # Mostrar recomendaciones en pantalla
            st.subheader("Recomendaciones Generadas")
            st.write(recomendaciones_generadas)

            # Generar informe PDF
            nombre_archivo = "Informe_Encuesta.pdf"
            generar_informe_pdf(resumen, recomendaciones_generadas, nombre_archivo)

            with open(nombre_archivo, "rb") as f:
                st.download_button("Descargar Informe", f, file_name=nombre_archivo)
        else:
            st.error("Variable no válida. La columna no existe en el archivo.")
