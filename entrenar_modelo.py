# train_model.py
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

def entrenar_modelo(df, variable_texto, columna_clasificacion):
    df[variable_texto] = df[variable_texto].astype(str).fillna('')
    X = df[variable_texto]
    y = df[columna_clasificacion]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='spanish')),
        ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42))
    ])

    pipeline.fit(X, y_encoded)
    y_pred = pipeline.predict(X)
    acc = accuracy_score(y_encoded, y_pred)

    # Guardar modelo y encoder en carpeta modelos/
    joblib.dump(pipeline, 'modelos/modelo_mlp.pkl')
    joblib.dump(le, 'modelos/label_encoder.pkl')

    return acc
