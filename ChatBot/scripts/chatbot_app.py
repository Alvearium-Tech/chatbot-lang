import os
import time
import requests
import streamlit as st

# Definir la URL del servidor
url_servidor = "http://localhost:8000"

# Definir la ruta base donde se encuentran los archivos de audio
ruta_base_audio = "./audio_files/"

# Definir la ruta del archivo de audio MP3 original
ruta_archivo_audio_mp3_original = os.path.join(ruta_base_audio, "recorded_audio.mp3")

# Definir la ruta del archivo de audio WAV
ruta_archivo_audio_wav = os.path.join(ruta_base_audio, "temp_audio_resampled.wav")

# Definir la ruta del archivo de audio MP3 de respuesta
ruta_archivo_audio_mp3_respuesta = os.path.join(ruta_base_audio, "respuesta.mp3")

# CSS personalizado para aplicar la paleta de colores
def aplicar_estilo_personalizado():
    st.markdown(f"""
    <style>
        /* Cambiar el fondo de toda la aplicación */
        body {{
            background-color: #390075;
            color: #f5fdff;
        }}
        /* Estilos para botones */
        .stButton>button {{
            border: 2px solid #b13237;
            color: #f5fdff;
            background-color: #b13237;
        }}
        .stButton>button:hover {{
            background-color: #00c5b3;
            color: #390075;
            border-color: #00c5b3;
        }}
        /* Estilos para inputs y otros elementos */
        .stTextInput>div>div>input, .stSelectbox>select {{
            background-color: #bc87fb;
            color: #390075;
            border-color: #bc87fb;
        }}
    </style>
    """, unsafe_allow_html=True)

# Función para grabar audio y enviarlo al servidor
def grabar_audio_y_enviar():
    try:
        # Realizar la solicitud al servidor para grabar audio
        response_record_audio = requests.post(f"{url_servidor}/record_audio")
        response_record_audio.raise_for_status()  # Lanzar una excepción en caso de error de solicitud

        st.success("Audio grabado exitosamente")

        # Esperar un tiempo para que el servidor tenga tiempo de procesar el audio
        time.sleep(2)  # Ajusta el tiempo de espera según sea necesario

        # Realizar la solicitud al servidor para convertir el audio grabado a texto (STT)
        response_stt = requests.post(f"{url_servidor}/speech_to_text", files={"file": open(ruta_archivo_audio_mp3_original, "rb")})
        response_stt.raise_for_status()  # Lanzar una excepción en caso de error de solicitud

        # Obtener la transcripción de audio a texto
        transcription = response_stt.json().get("text")

        st.write(f"Transcripción de audio a texto: {transcription}")

        # Realizar la solicitud al servidor para obtener la respuesta del chatbot
        response_answer = requests.post(f"{url_servidor}/answer", json={"text": transcription})
        response_answer.raise_for_status()  # Lanzar una excepción en caso de error de solicitud

        # Obtener la respuesta del chatbot
        answer = response_answer.content

        # Guardar la respuesta del chatbot en un archivo de audio MP3
        with open(ruta_archivo_audio_mp3_respuesta, "wb") as audio_file:
            audio_file.write(answer)

        st.success("Respuesta del chatbot generada correctamente")

        # Reproducir la respuesta del chatbot
        st.audio(ruta_archivo_audio_mp3_respuesta, format='audio/mp3')

    except requests.exceptions.RequestException as e:
        st.error(f"Error en la solicitud: {e}")
    except Exception as e:
        st.error(f"Error inesperado: {e}")

# Función para obtener el historial del chat desde la API
def get_chat_history():
    try:
        response = requests.get("http://localhost:8000/chat_history")
        response.raise_for_status()  # Lanzar una excepción si la solicitud falla
        return response.json().get("chat_history", [])
    except Exception as e:
        st.error(f"Error al obtener el historial del chat: {e}")
        return []

# Función principal de la aplicación Streamlit
def main():
    aplicar_estilo_personalizado()
    st.title("Alvearium - Chatbot")

    # Agregar pestañas para las diferentes funcionalidades
    tabs = st.sidebar.radio("Navegación", ["Grabar Audio", "Ver Historial de Conversación"])

    if tabs == "Grabar Audio":
        st.sidebar.image("cropped-cropped-favicon-01-32x32.png", width=50)  # Agregar icono a la pestaña
        # Mostrar botón para iniciar la grabación de audio
        if st.button("Iniciar grabación de audio"):
            grabar_audio_y_enviar()

    elif tabs == "Ver Historial de Conversación":
        st.sidebar.image("cropped-cropped-favicon-01-32x32.png", width=50)  # Agregar icono a la pestaña
        # Obtener el historial de la conversación desde la API
        chat_history = get_chat_history()
        
        # Mostrar el historial de la conversación en Streamlit
        for speaker, message in chat_history:
            st.write(f"{speaker}: {message}")

# Llamar a la función principal para iniciar la aplicación Streamlit
if __name__ == "__main__":
    main()