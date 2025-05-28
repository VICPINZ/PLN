
import streamlit as st

def limpiar_caracteres_invisibles(nombre_archivo):
    with open(nombre_archivo, 'r', encoding='utf-8') as file:
        contenido = file.read()

    # Reemplazar caracteres invisibles
    contenido_limpio = ''.join(c for c in contenido if c.isprintable())

    nuevo_nombre_archivo = nombre_archivo.replace('.py', '_cleaned.py')
    with open(nuevo_nombre_archivo, 'w', encoding='utf-8') as file:
        file.write(contenido_limpio)

    return nuevo_nombre_archivo

# Interfaz de Streamlit
st.title("Limpiar caracteres invisibles de archivos Python")

archivo = st.file_uploader("Cargar archivo Python", type=["py"])
if archivo:
    nombre_archivo = archivo.name
    with open(nombre_archivo, 'wb') as f:
        f.write(archivo.getbuffer())

    nuevo_nombre_archivo = limpiar_caracteres_invisibles(nombre_archivo)
    st.success(f"Archivo limpio guardado como: {nuevo_nombre_archivo}")

    with open(nuevo_nombre_archivo, 'rb') as f:
        st.download_button("Descargar archivo limpio", f, file_name=nuevo_nombre_archivo)
