import os
import time
import requests

# Definir la URL del servidor
url_servidor = "http://localhost:8000"

# Definir la ruta del archivo de audio MP3 original
ruta_archivo_audio_mp3_original = r"C:\Users\ferna\OneDrive\Escritorio\GitHub\Alvearium-VR\ChatBot\audio_files\Record (online-voice-recorder.com).mp3"

# Definir la ruta del archivo de audio WAV
ruta_archivo_audio_wav = r"C:\Users\ferna\OneDrive\Escritorio\GitHub\Alvearium-VR\ChatBot\scripts\temp_audio_resampled.wav"

# Definir la ruta del archivo de audio MP3 de respuesta
ruta_archivo_audio_mp3_respuesta = r"C:\Users\ferna\OneDrive\Escritorio\GitHub\Alvearium-VR\ChatBot\audio_files\respuesta.mp3"

try:
    # Eliminar el archivo WAV existente, si lo hay
    if os.path.exists(ruta_archivo_audio_wav):
        os.remove(ruta_archivo_audio_wav)

    # Realizar la solicitud al servidor para convertir el MP3 a WAV
    response_mp3_to_wav = requests.post(f"{url_servidor}/ask_audio", files={"file": open(ruta_archivo_audio_mp3_original, "rb")})
    response_mp3_to_wav.raise_for_status()  # Lanzar una excepción en caso de error de solicitud

    # Esperar un tiempo para que el servidor tenga tiempo de crear el archivo WAV
    time.sleep(2)  # Ajusta el tiempo de espera según sea necesario

    # Realizar la solicitud al servidor para convertir el audio WAV a texto (STT)
    with open(ruta_archivo_audio_wav, "rb") as file:
        audio_data = {"file": file}
        response_stt = requests.post(f"{url_servidor}/speech_to_text", files=audio_data)
    response_stt.raise_for_status()  # Lanzar una excepción en caso de error de solicitud

    # Obtener la transcripción de audio a texto
    transcription = response_stt.json().get("text")

    print(f"Transcripción de audio a texto exitosa: {transcription}")

    # Realizar la solicitud al servidor para obtener la respuesta del chatbot
    response_answer = requests.post(f"{url_servidor}/answer", json={"text": transcription})
    response_answer.raise_for_status()  # Lanzar una excepción en caso de error de solicitud

    # Obtener la respuesta del chatbot
    answer = response_answer.content

    # Guardar la respuesta del chatbot en un archivo de audio MP3
    with open(ruta_archivo_audio_mp3_respuesta, "wb") as audio_file:
        audio_file.write(answer)

    print("Respuesta del chatbot generada correctamente")

except requests.exceptions.RequestException as e:
    print(f"Error en la solicitud: {e}")
except Exception as e:
    print(f"Error inesperado: {e}")

