import os
import time
import requests

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

try:
    # Realizar la solicitud al servidor para grabar audio
    response_record_audio = requests.post(f"{url_servidor}/record_audio")
    response_record_audio.raise_for_status()  # Lanzar una excepción en caso de error de solicitud

    print("Audio grabado exitosamente")

    # Esperar un tiempo para que el servidor tenga tiempo de procesar el audio
    time.sleep(2)  # Ajusta el tiempo de espera según sea necesario

    # Realizar la solicitud al servidor para convertir el audio grabado a texto (STT)
    response_stt = requests.post(f"{url_servidor}/speech_to_text", files={"file": open(ruta_archivo_audio_mp3_original, "rb")})
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