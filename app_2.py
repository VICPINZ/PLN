import os
import joblib
import pandas as pd
import streamlit as st
from docx import Document
import string
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import nltk
import google.generativeai as genai

# Descargar stopwords de NLTK
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_es = set(stopwords.words('spanish'))

# Configurar clave de API de Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Función para predecir clasificación ---
def predecir_clasificacion(textos):
    modelo_path = 'modelo_entrenado.pkl'
    encoder_path = 'label_encoder.pkl'

    if not os.path.isfile(modelo_path) or not os.path.isfile(encoder_path):
        st.error(f"❌ No se encontró el archivo del modelo o del codificador.\n\n"
                 f"Asegúrate de subir 'modelo_rf.pkl' y 'label_encoder.pkl' a tu repositorio y que estén en la misma carpeta que este archivo.")
        st.stop()

    pipeline = joblib.load(modelo_path)
    le = joblib.load(encoder_path)
    predicciones = pipeline.predict(textos)
    return le.inverse_transform(predicciones)

# --- Función para generar recomendaciones ---
def generar_recomendaciones(comentarios):
    entrada = " ".join(comentarios)
    prompt = f"Basado en los siguientes comentarios de recomendación, sugiere mejoras concretas:\n{entrada}\n\nRecomendaciones:"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Ocurrió un error al generar las recomendaciones: {e}"

# --- Función para generar informe Word ---
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


# --- Pestaña 1: Entrenamiento ---
with tab1:
    st.header("Entrenar modelo de clasificación de texto")
    archivo_entrenamiento = st.file_uploader("Cargar archivo Excel para entrenamiento", type=["xlsx"], key="entrenamiento")
    if archivo_entrenamiento:
        df_entrenamiento = pd.read_excel(archivo_entrenamiento)
        st.write("Vista previa:", df_entrenamiento.head())

        columnas = df_entrenamiento.columns.tolist()
        variable_texto = st.selectbox("Selecciona la columna de texto", columnas, key="texto_entrenamiento")
        columna_clasificacion = st.selectbox("Selecciona la columna de clasificación", columnas, key="clasificacion_entrenamiento")

        if st.button("Entrenar modelo"):
            accuracy = entrenar_modelo(df_entrenamiento, variable_texto, columna_clasificacion)
            st.success(f"Modelo entrenado con éxito. Accuracy: {accuracy:.4f}")
            st.info("Los archivos 'modelo_rf.pkl' y 'label_encoder.pkl' han sido guardados.")


# --- Pestaña 2: Generación de Informe ---
with tab2:
    st.header("Generar informe a partir de datos clasificados")
    archivo_informe = st.file_uploader("Cargar archivo Excel para análisis", type=["xlsx"], key="informe")
    if archivo_informe:
        df_informe = pd.read_excel(archivo_informe)
        st.write("Vista previa:", df_informe.head())

        variable = st.text_input("Nombre de la columna de texto:", key="texto_informe")
        if st.button("Generar Informe"):
            if variable in df_informe.columns:
                df_informe[variable] = df_informe[variable].astype(str)

                predicciones = predecir_clasificacion(df_informe[variable])
                if predicciones is None:
                    st.warning("No se pudo generar el informe debido a la falta del modelo.")
                else:
                    df_informe['clasificacion'] = predicciones

                    resumen = []
                    for clase, grupo in df_informe.groupby('clasificacion'):
                        textos = " ".join(grupo[variable].tolist()).lower()
                        textos = textos.translate(str.maketrans('', '', string.punctuation))
                        palabras = [w for w in textos.split() if w not in stopwords_es and len(w) > 2]
                        mas_comunes = [w for w, _ in Counter(palabras).most_common(10)]
                        resumen.append({
                            'Clase': clase,
                            'Conteo': len(grupo),
                            'Top 10 palabras': ", ".join(mas_comunes)
                        })

                    st.subheader("Resumen por clase")
                    st.dataframe(pd.DataFrame(resumen))

                    recomendaciones_comentarios = df_informe[df_informe['clasificacion'] == 'COMENTARIO'][variable].tolist()
                    recomendaciones_generadas = generar_recomendaciones(recomendaciones_comentarios) if recomendaciones_comentarios else "No se encontraron recomendaciones para analizar."

                    nombre_archivo = "Informe_Encuesta.docx"
                    generar_informe(df_informe[[variable, 'clasificacion']].rename(columns={variable: 'comentario'}), recomendaciones_generadas, nombre_archivo, resumen)

                    with open(nombre_archivo, "rb") as f:
                        st.download_button("📥 Descargar Informe", f, file_name=nombre_archivo)
            else:
                st.error("La columna especificada no existe en el archivo.")

