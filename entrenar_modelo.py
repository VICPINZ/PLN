
import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

def entrenar_modelo(df, variable_texto, columna_clasificacion):
    # Eliminar modelos existentes si ya hay
    for f in ["modelo_rf.pkl", "label_encoder.pkl"]:
        if os.path.exists(f):
            os.remove(f)

    # Reemplazar valores nulos en la columna de texto por cadenas vacías
    df[variable_texto] = df[variable_texto].fillna('')

    # Preparar datos
    X = df[variable_texto]
    y = df[columna_clasificacion]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Pipeline: TF-IDF vectorizer + MLPClassifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='spanish')),
        ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42))
    ])

    # Entrenar modelo
    pipeline.fit(X, y_encoded)

    # Calcular accuracy
    y_pred = pipeline.predict(X)
    acc = accuracy_score(y_encoded, y_pred)

    # Guardar modelo y LabelEncoder
    joblib.dump(pipeline, 'modelo_rf.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    return acc

# --- Interfaz Streamlit ---
st.title("Entrenamiento de Modelo de Clasificación de Texto")

archivo = st.file_uploader("Cargar archivo Excel", type=["xlsx"])
if archivo:
    df = pd.read_excel(archivo)
    st.write("Vista previa de los datos:", df.head())

    columnas = df.columns.tolist()
    variable_texto = st.selectbox("Selecciona la columna de texto", columnas)
    columna_clasificacion = st.selectbox("Selecciona la columna de clasificación", columnas)

    if st.button("Entrenar modelo"):
        if variable_texto and columna_clasificacion:
            accuracy = entrenar_modelo(df, variable_texto, columna_clasificacion)
            st.success(f"Modelo entrenado con éxito. Accuracy en entrenamiento: {accuracy:.4f}")
            st.info("Los archivos 'modelo_rf.pkl' y 'label_encoder.pkl' han sido guardados.")
        else:
            st.error("Por favor selecciona ambas columnas.")
