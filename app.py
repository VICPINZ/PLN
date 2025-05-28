import os
import streamlit as st
import pandas as pd
import joblib
from train_model import entrenar_modelo
import google.generativeai as genai

# Leer API Key desde secrets de Streamlit
API_KEY = st.secrets["gemini"]["api_key"]
genai.configure(api_key=API_KEY)

def cargar_modelos():
    pipeline = joblib.load("modelo_rf.pkl")
    le = joblib.load("label_encoder.pkl")
    return pipeline, le

def obtener_recomendacion_gemini(texto, categoria):
    prompt = f"Comentario: {texto}\nClasificación: {categoria}\nGenera recomendaciones específicas para este comentario."
    try:
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error al generar recomendación: {e}"

def main():
    st.title("Clasificación de Comentarios y Recomendaciones - Gemini")

    st.info("Primero, sube el archivo Excel para entrenar el modelo:")

    archivo_excel = st.file_uploader("Carga el archivo Excel (.xlsx)", type="xlsx")
    if archivo_excel:
        variable_texto = st.text_input("Nombre de la columna de texto")
        columna_clasificacion = st.text_input("Nombre de la columna de clasificación")

        if st.button("Entrenar Modelo"):
            with open("temp_data.xlsx", "wb") as f:
                f.write(archivo_excel.read())
            entrenar_modelo("temp_data.xlsx", variable_texto, columna_clasificacion)
            st.success("Modelo entrenado correctamente.")

    st.divider()
    st.subheader("Clasificar nuevos comentarios")

    texto_usuario = st.text_area("Escribe el comentario aquí")
    if st.button("Clasificar y generar recomendación"):
        if os.path.exists("modelo_rf.pkl") and os.path.exists("label_encoder.pkl"):
            pipeline, le = cargar_modelos()
            categoria_pred = pipeline.predict([texto_usuario])[0]
            categoria_nombre = le.inverse_transform([categoria_pred])[0]
            st.write(f"**Clasificación del comentario:** {categoria_nombre}")

            recomendacion = obtener_recomendacion_gemini(texto_usuario, categoria_nombre)
            st.write("**Recomendación personalizada:**")
            st.write(recomendacion)
        else:
            st.warning("Primero debes entrenar el modelo subiendo un archivo Excel.")

if __name__ == "__main__":
    main()
