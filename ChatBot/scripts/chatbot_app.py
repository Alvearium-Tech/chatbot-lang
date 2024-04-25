import os
import requests
import streamlit as st
import base64
import tempfile
from streamlit.components.v1 import html

# Definir la URL del servidor
url_servidor = "http://18.192.57.108:8000"
#url_servidor = "http://127.0.0.1:8000"

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
def enviar_pregunta_escrita_al_modelo(pregunta):
    try:
        # Realizar la solicitud al servidor para obtener la respuesta del chatbot
        response = requests.post(f"{url_servidor}/answer", json={"text": pregunta})
        response.raise_for_status()  # Lanzar una excepción en caso de error de solicitud

        # Obtener la respuesta del chatbot
        respuesta = response.json()
        answer = respuesta["text_response"]

        # Mostrar la respuesta del chatbot
        st.write("Respuesta del chatbot:")
        st.write(answer)  # Decodificar la respuesta a UTF-8 antes de mostrarla

    except requests.exceptions.RequestException as e:
        st.error(f"Error en la solicitud: {e}")
    except Exception as e:
        st.error(f"Error inesperado: {e}")

# Función para obtener el historial del chat desde la API
def get_chat_history():
    try:
        response = requests.get(f"{url_servidor}/chat_history")
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
    tabs = st.sidebar.radio("Navegación", ["Escribir Pregunta", "Grabar Pregunta", "Ver Historial de Conversación"])

    if tabs == "Escribir Pregunta":
        #st.sidebar.image("cropped-cropped-favicon-01-32x32.png", width=50)

        st.header("Escribir Pregunta")
        pregunta_usuario = st.text_area("Escribe tu pregunta aqui")

        if st.button("Enviar Pregunta"):
            if pregunta_usuario:
                enviar_pregunta_escrita_al_modelo(pregunta_usuario)
            else:
                st.warning("Por favor ingresa una pregunta antes de enviarla")

    elif tabs == "Grabar Pregunta":
        #st.sidebar.image("cropped-cropped-favicon-01-32x32.png", width=50)
        if st.button("Iniciar grabación de audio"):
            st.write("Haz clic en el botón de grabar para iniciar la grabación.")
            html_code = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Grabador de Audio</title>
                <!-- Corregir el enlace a la librería RecordRTC.js -->
                <script src="https://cdn.webrtc-experiment.com/RecordRTC.js"></script>
            </head>
            <body>
                <button id="btn-start-recording">Iniciar Grabación</button>
                <button id="btn-stop-recording" disabled>Detener Grabación</button>

                <script>
                    let recorder;
                    let stream;
                    const startButton = document.getElementById('btn-start-recording');
                    const stopButton = document.getElementById('btn-stop-recording');

                    function stopStream() {
                        if (stream) {
                            stream.getTracks().forEach(track => track.stop());
                        }
                    }

                    startButton.addEventListener('click', function() {
                        this.disabled = true;
                        stopButton.disabled = false;
                        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                            navigator.mediaDevices.getUserMedia({ audio: true })
                            .then(function(mediaStream) {
                                stream = mediaStream;
                                recorder = RecordRTC(stream, {
                                    type: 'audio',
                                    // Cambiar el tipo MIME a audio/webm para una mejor compatibilidad
                                    mimeType: 'audio/webm',
                                    recorderType: RecordRTC.StereoAudioRecorder,
                                    desiredSampRate: 16000
                                });

                                recorder.startRecording();
                            })
                            .catch(function(error) {
                                console.error('Error al acceder a los dispositivos de medios o el usuario denegó el acceso:', error);
                                startButton.disabled = false;
                                stopButton.disabled = true;
                            });
                        } else {
                            console.error('getUserMedia no es compatible');
                            startButton.disabled = false;
                            stopButton.disabled = true;
                        }
                    });

                    stopButton.addEventListener('click', function() {
                        this.disabled = true;
                        if (recorder && typeof recorder.stopRecording === 'function') {
                            recorder.stopRecording(function() {
                                let blob = recorder.getBlob();

                                // Enviar el archivo de audio al servidor
                                let formData = new FormData();
                                formData.append('file', blob, 'grabacion_audio.webm');
                                fetch('http://18.192.57.108:8000/speech_to_text', {
                                    method: 'POST',
                                    body: formData
                                })
                                .then(response => response.json())
                                .then(data => {
                                    console.log('Transcripción recibida:', data.text);

                                    // Realizar la solicitud fetch adicional para obtener la respuesta del servidor
                                    fetch('http://18.192.57.108:8000/answer', {
                                        method: 'POST',
                                        headers: {
                                            'Content-Type': 'application/json'
                                        },
                                        body: JSON.stringify({ text: data.text })
                                    })
                                    .then(response => response.json())
                                    .then(data => {
                                        console.log('Respuesta del servidor:', data);

                                        // Obtener la URL del archivo de audio
                                        const audioUrl = data.audio_url;

                                        // Reproducir el archivo de audio
                                        const audioElement = new Audio(audioUrl);
                                        audioElement.play();
                                    })
                                    .catch(error => {
                                        console.error('Error al enviar la solicitud de audio:', error);
                                    })
                                    .finally(() => {
                                        startButton.disabled = false;
                                        stopStream(); // Detener el flujo de medios
                                    });
                                })
                                .catch(error => {
                                    console.error('Error al enviar el audio:', error);
                                })
                                .finally(() => {
                                    startButton.disabled = false;
                                    stopStream(); // Detener el flujo de medios
                                });
                            });
                        } else {
                            console.error('El grabador no está inicializado o no tiene un método stopRecording');
                        }
                    });
                </script>
            </body>
            </html>
            """
            st.components.v1.html(html_code, height=200)

    elif tabs == "Ver Historial de Conversación":
        #st.sidebar.image("ChatBot\scripts\cropped-cropped-favicon-01-32x32.png", width=50)  # Agregar icono a la pestaña
        # Obtener el historial de la conversación desde la API
        chat_history = get_chat_history()
        
        # Mostrar el historial de la conversación en Streamlit
        for speaker, message in chat_history:
            st.write(f"{speaker}: {message}")

# Llamar a la función principal para iniciar la aplicación Streamlit
if __name__ == "__main__":
    main()
