import os
import streamlit as st
import pandas as pd
import joblib
from collections import Counter
import string
from fpdf import FPDF
import google.generativeai as genai

# Configurar la clave de API de Gemini
genai.configure(api_key=st.secrets["GENAI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# Cargar modelos entrenados (pipeline y label encoder)
@st.cache_resource
def cargar_modelos():
    pipeline = joblib.load("modelo_rf.pkl")
    le = joblib.load("label_encoder.pkl")
    return pipeline, le

# Función para predecir clasificaciones
def predecir_clasificacion(textos, pipeline, le):
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
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Informe de Resultados de la Encuesta", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Resumen por clase", ln=True)
    pdf.set_font("Arial", size=12)
    for fila in resumen:
        pdf.cell(0, 10, f"Clase: {fila['Clase']}", ln=True)
        pdf.cell(0, 10, f"Conteo: {fila['Conteo']}", ln=True)
        pdf.multi_cell(0, 10, f"Top 10 palabras: {fila['Top 10 palabras']}")
        pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Propuesta de Mejora", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, recomendaciones)

    pdf.output(nombre_archivo)

# Interfaz de usuario con Streamlit
def main():
    st.title("Análisis de Encuesta - Ministerio de Defensa")
    archivo = st.file_uploader("Cargar archivo Excel", type=["xlsx"])
    
    if archivo:
        df = pd.read_excel(archivo)
        st.write("Vista previa de datos:", df.head())

        variable = st.text_input("Ingrese el nombre de la columna de texto:")
        if st.button("Generar Informe"):
            if variable in df.columns:
                df = df.dropna(subset=[variable])
                df[variable] = df[variable].astype(str)

                # Cargar modelos una vez
                pipeline, le = cargar_modelos()
                df['clasificacion'] = predecir_clasificacion(df[variable], pipeline, le)

                # Tabla resumen por clase
                resumen = []
                for clase, grupo in df.groupby('clasificacion'):
                    textos = " ".join(grupo[variable].tolist()).lower()
                    textos = textos.translate(str.maketrans('', '', string.punctuation))
                    palabras = [w for w in textos.split() if len(w) > 2]
                    mas_comunes = [w for w, _ in Counter(palabras).most_common(10)]
                    resumen.append({
                        'Clase': clase,
                        'Conteo': len(grupo),
                        'Top 10 palabras': ", ".join(mas_comunes)
                    })
                st.subheader("Resumen por clase")
                st.dataframe(pd.DataFrame(resumen))

                # Filtrar comentarios tipo Recomendación (opcional)
                recomendaciones_comentarios = df[df['clasificacion'].str.lower() == 'comentario'][variable].tolist()
                recomendaciones_generadas = generar_recomendaciones(recomendaciones_comentarios) if recomendaciones_comentarios else "No se encontraron recomendaciones para analizar."

                st.subheader("Recomendaciones Generadas")
                st.write(recomendaciones_generadas)

                # Generar PDF
                nombre_archivo = "Informe_Encuesta.pdf"
                generar_informe_pdf(resumen, recomendaciones_generadas, nombre_archivo)
                with open(nombre_archivo, "rb") as f:
                    st.download_button("Descargar Informe PDF", f, file_name=nombre_archivo)
            else:
                st.error("Variable no válida. La columna no existe en el archivo.")

if __name__ == "__main__":
    main()
