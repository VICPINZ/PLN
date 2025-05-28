import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

def entrenar_modelo(archivo_excel, variable_texto, columna_clasificacion):
    # Eliminar modelos existentes si ya hay
    for f in ["modelo_rf.pkl", "label_encoder.pkl"]:
        if os.path.exists(f):
            os.remove(f)

    # Cargar datos
    df = pd.read_excel(archivo_excel)

    # Reemplazar valores nulos en la columna de texto por cadenas vacías
    df[variable_texto] = df[variable_texto].fillna('')

    # Preparar datos
    X = df[variable_texto]
    y = df[columna_clasificacion]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Pipeline: TF-IDF Vectorizer + MLPClassifier (red neuronal)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42))
    ])

    # Entrenar modelo
    pipeline.fit(X, y_encoded)

    # Calcular accuracy
    y_pred = pipeline.predict(X)
    acc = accuracy_score(y_encoded, y_pred)
    print(f"✅ Accuracy del modelo en los datos de entrenamiento: {acc:.4f}")

    # Guardar modelo y LabelEncoder
    joblib.dump(pipeline, 'modelo_rf.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    print("✅ Modelo entrenado y guardado exitosamente.")

# Ejecución directa por terminal (opcional)
if __name__ == "__main__":
    archivo = input("Ingrese el nombre del archivo Excel (ej. datos.xlsx): ")
    variable = input("Ingrese el nombre de la columna de texto: ")
    clasificacion = input("Ingrese el nombre de la columna de clasificación (etiquetas): ")
    entrenar_modelo(archivo, variable, clasificacion)
