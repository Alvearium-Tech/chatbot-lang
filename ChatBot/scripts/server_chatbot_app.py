import streamlit as st
import requests
import tempfile
import soundfile as sf
import speech_recognition as sr

# Título de la aplicación
st.title("Chatbot Interactivo")

# URL del servidor
server_url = "http://localhost:8000"

# Función para realizar una solicitud al servidor y obtener la respuesta
def send_request(endpoint, payload):
    try:
        response = requests.post(f"{server_url}/{endpoint}", json=payload)
        response.raise_for_status()  # Lanzar una excepción si la solicitud falla
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Error en la solicitud: {e}")

# Definir las opciones de la aplicación
option = st.selectbox(
    "Seleccione una opción:",
    ("Realizar pregunta", "Ver historial del chat")
)

# Lógica para cada opción
if option == "Realizar pregunta":
    st.subheader("Realizar pregunta")

    # Opción para ingresar texto o cargar un archivo de audio
    method = st.radio("Seleccione un método:", ("Texto", "Audio"))

    if method == "Texto":
        user_input = st.text_input("Ingrese su pregunta:")
    else:
        st.info("Cargue un archivo de audio (mp3 o wav) para realizar la pregunta.")
        audio_file = st.file_uploader("Cargar archivo de audio")

    if st.button("Enviar pregunta"):
        if method == "Texto" and user_input:
            payload = {"text": user_input}
            response = send_request("answer", payload)
            st.audio(response, format="audio/mp3")
        elif method == "Audio" and audio_file:
            with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
                temp_audio.write(audio_file.read())
                temp_audio_path = temp_audio.name

            try:
                recognizer = sr.Recognizer()
                with sr.AudioFile(temp_audio_path) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data, language="es-ES")

                payload = {"text": text}
                response = send_request("answer", payload)
                st.audio(response, format="audio/mp3")
            except Exception as e:
                st.error(f"Error al procesar el archivo de audio: {e}")

elif option == "Ver historial del chat":
    st.subheader("Ver historial del chat")
    response = send_request("chat_history", {})
    chat_history = response.json().get("chat_history", [])
    for turn in chat_history:
        st.write(f"{turn[0]}: {turn[1]}")
