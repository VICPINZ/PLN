# entrenar_modelo.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

def entrenar_modelo(df, columna_texto, columna_clasificacion):
    # Paso 1: Preprocesamiento
    X = df[columna_texto].astype(str)
    y = df[columna_clasificacion].astype(str)

    # Codificar etiquetas (LabelEncoder)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Definir stopwords en español usando NLTK
    stopwords_es = list(stopwords.words('spanish'))

    # Pipeline: TF-IDF + MLPClassifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words=stopwords_es)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42))
    ])

    # División de datos (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Entrenar el modelo
    pipeline.fit(X_train, y_train)

    # Calcular accuracy en el conjunto de prueba
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Guardar el modelo entrenado y el label encoder
    import joblib
    joblib.dump(pipeline, 'modelo_entrenado.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    return accuracy
