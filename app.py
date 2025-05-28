# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.exceptions import NotFittedError

# Cargar modelos entrenados
try:
    pipeline = joblib.load('modelo_rf.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    pipeline = None
    le = None

def main():
    st.set_page_config(page_title="Clasificación de Comentarios", layout="centered")
    st.title("📊 Clasificador de Comentarios (PLN)")

    st.sidebar.header("Opciones")

    menu = st.sidebar.selectbox("Selecciona una opción:", ["Predicción de comentarios", "Entrenar modelo"])

    if menu == "Predicción de comentarios":
        if pipeline is None or le is None:
            st.warning("⚠️ No se ha cargado un modelo entrenado. Por favor, entrena uno primero en la pestaña 'Entrenar modelo'.")
            return
        
        st.subheader("Predicción de Comentarios")
        comentario = st.text_area("Escribe tu comentario aquí:")

        if st.button("Predecir"):
            if comentario.strip() == "":
                st.warning("Por favor, ingresa un comentario.")
            else:
                try:
                    pred = pipeline.predict([comentario])[0]
                    etiqueta = le.inverse_transform([pred])[0]
                    st.success(f"✅ Clasificación del comentario: **{etiqueta}**")
                except NotFittedError:
                    st.error("El modelo no está entrenado. Entrena el modelo antes de usar esta funcionalidad.")
    
    elif menu == "Entrenar modelo":
        st.subheader("Entrenar un nuevo modelo")
        archivo = st.file_uploader("Carga tu archivo Excel con datos", type=["xlsx"])
        if archivo is not None:
            df = pd.read_excel(archivo)
            st.write("📋 Vista previa de los datos:", df.head())

            columnas = df.columns.tolist()
            columna_texto = st.selectbox("Selecciona la columna de texto:", columnas)
            columna_clasificacion = st.selectbox("Selecciona la columna de clasificación:", columnas)

            if st.button("Entrenar modelo"):
                with st.spinner("Entrenando el modelo... Esto puede tardar un momento."):
                    from train_model import entrenar_modelo
                    archivo_path = f"datos_temporales.xlsx"
                    df.to_excel(archivo_path, index=False)
                    entrenar_modelo(archivo_path, columna_texto, columna_clasificacion)

                    # Recargar modelos
                    global pipeline, le
                    pipeline = joblib.load('modelo_rf.pkl')
                    le = joblib.load('label_encoder.pkl')

                st.success("✅ Modelo entrenado y cargado exitosamente.")

if __name__ == "__main__":
    main()
