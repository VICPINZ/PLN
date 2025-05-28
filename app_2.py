# app.py
import os
import joblib
import pandas as pd
import streamlit as st
from docx import Document
import string
from collections import Counter
import nltk
import google.generativeai as genai
from train_model import entrenar_modelo

# Configurar clave API de Gemini
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# Descargar stopwords si no están
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_es = set(stopwords.words('spanish'))

# Funciones
def predecir_clasificacion(textos):
    try:
        pipeline = joblib.load('modelos/modelo_mlp.pkl')
        le = joblib.load('modelos/label_encoder.pkl')
    except FileNotFoundError:
        st.error("Modelo no entrenado. Por favor, entrena el modelo primero.")
        return []
    predicciones = pipeline.predict(textos)
    return le.inverse_transform(predicciones)

def generar_recomendaciones(comentarios):
    entrada = " ".join(comentarios)
    prompt = f"Basado en los siguientes comentarios de recomendación, sugiere mejoras concretas:\n{entrada}\n\nRecomendaciones:"
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response and hasattr(response, "text") else "No se pudo generar recomendaciones."
    except Exception as e:
        return f"Ocurrió un error: {e}"

def generar_informe(comentarios, recomendaciones, nombre_archivo, resumen):
    doc = Document()
    doc.add_heading("Informe de Resultados de la Encuesta", 0)
    doc.add_heading("Resumen por clase", level=1)
    table = doc.add_table(rows=1, cols=3)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Clase'
    hdr_cells[1].text = 'Conteo'
    hdr_cells[2].text = 'Top 10 palabras'
    for fila in resumen:
        row_cells = table.add_row().cells
        row_cells[0].text = str(fila['Clase'])
        row_cells[1].text = str(fila['Conteo'])
        row_cells[2].text = fila['Top 10 palabras']
    doc.add_heading("Propuesta de Mejora", level=1)
    doc.add_paragraph(recomendaciones)
    doc.save(nombre_archivo)

# --- Interfaz Streamlit ---
st.set_page_config(page_title="Análisis de Encuesta", layout="wide")
st.title("Análisis de Encuesta - Ministerio de Defensa")

tab1, tab2 = st.tabs(["📊 Entrenamiento del Modelo", "📄 Generación de Informe"])

# --- Pestaña 1 ---
with tab1:
    st.header("Entrenar modelo de clasificación de texto")
    archivo_entrenamiento = st.file_uploader("Cargar archivo Excel para entrenamiento", type=["xlsx"])
    if archivo_entrenamiento:
        df_entrenamiento = pd.read_excel(archivo_entrenamiento)
        st.write("Vista previa:", df_entrenamiento.head())
        columnas = df_entrenamiento.columns.tolist()
        variable_texto = st.selectbox("Selecciona la columna de texto", columnas)
        columna_clasificacion = st.selectbox("Selecciona la columna de clasificación", columnas)
        if st.button("Entrenar modelo"):
            os.makedirs("modelos", exist_ok=True)
            accuracy = entrenar_modelo(df_entrenamiento, variable_texto, columna_clasificacion)
            st.success(f"Modelo entrenado con éxito. Accuracy: {accuracy:.4f}")
            st.info("Modelo y encoder guardados en la carpeta 'modelos'.")

# --- Pestaña 2 ---
with tab2:
    st.header("Generar informe a partir de datos clasificados")
    archivo_informe = st.file_uploader("Cargar archivo Excel para análisis", type=["xlsx"])
    if archivo_informe:
        df_informe = pd.read_excel(archivo_informe)
        st.write("Vista previa:", df_informe.head())
        variable = st.text_input("Nombre de la columna de texto:")
        if st.button("Generar Informe"):
            if variable in df_informe.columns:
                df_informe[variable] = df_informe[variable].astype(str)
                df_informe['clasificacion'] = predecir_clasificacion(df_informe[variable])
                resumen = []
                for clase, grupo in df_informe.groupby('clasificacion'):
                    textos = " ".join(grupo[variable].tolist()).lower()
                    textos = textos.translate(str.maketrans('', '', string.punctuation))
                    palabras = [w for w in textos.split() if w not in stopwords_es and len(w) > 2]
                    mas_comunes = [w for w, _ in Counter(palabras).most_common(10)]
                    resumen.append({'Clase': clase, 'Conteo': len(grupo), 'Top 10 palabras': ", ".join(mas_comunes)})
                st.subheader("Resumen por clase")
                st.dataframe(pd.DataFrame(resumen))
                recomendaciones_comentarios = df_informe[df_informe['clasificacion'] == 'COMENTARIO'][variable].tolist()
                recomendaciones_generadas = generar_recomendaciones(recomendaciones_comentarios) if recomendaciones_comentarios else "No se encontraron recomendaciones."
                nombre_archivo = "Informe_Encuesta.docx"
                generar_informe(df_informe[[variable, 'clasificacion']].rename(columns={variable: 'comentario'}), recomendaciones_generadas, nombre_archivo, resumen)
                with open(nombre_archivo, "rb") as f:
                    st.download_button("📥 Descargar Informe", f, file_name=nombre_archivo)
            else:
                st.error("La columna especificada no existe en el archivo.")
