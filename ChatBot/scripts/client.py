import os
import time
import requests
from google.cloud import storage

# Definir la URL del servidor
url_servidor = "http://localhost:8000"

# Nombre del bucket de Google Cloud Storage
nombre_bucket = "alvea_example"

# Inicializar el cliente de Google Cloud Storage
cliente_storage = storage.Client()

# Definir las rutas de los archivos de audio
ruta_archivo_audio_mp3_original = "./audio_files/original.mp3"
ruta_archivo_audio_wav = "./audio_files/temp_audio_resampled.wav"
ruta_archivo_audio_mp3_respuesta = "./audio_files/respuesta.mp3"

# Función para subir un archivo a Google Cloud Storage
def upload_to_gcs(file_path, gcs_file_name):
    bucket = cliente_storage.bucket(nombre_bucket)
    blob = bucket.blob(gcs_file_name)
    blob.upload_from_filename(file_path)

# Función para descargar un archivo desde Google Cloud Storage
def download_from_gcs(gcs_file_name, local_file_path):
    bucket = cliente_storage.bucket(nombre_bucket)
    blob = bucket.blob(gcs_file_name)
    blob.download_to_filename(local_file_path)

try:
    # Eliminar el archivo WAV existente, si lo hay
    if os.path.exists(ruta_archivo_audio_wav):
        os.remove(ruta_archivo_audio_wav)

    # Subir el archivo MP3 original a Google Cloud Storage
    upload_to_gcs(ruta_archivo_audio_mp3_original, "audio_files/original.mp3")

    # Realizar la solicitud al servidor para convertir el MP3 a WAV
    response_mp3_to_wav = requests.post(f"{url_servidor}/ask_audio", json={"bucket": nombre_bucket, "file_name": "audio_files/original.mp3"})
    response_mp3_to_wav.raise_for_status()  # Lanzar una excepción en caso de error de solicitud

    # Esperar un tiempo para que el servidor tenga tiempo de crear el archivo WAV
    time.sleep(2)  # Ajusta el tiempo de espera según sea necesario

    # Realizar la solicitud al servidor para convertir el audio WAV a texto (STT)
    response_stt = requests.post(f"{url_servidor}/speech_to_text", json={"bucket": nombre_bucket, "file_name": "audio_files/temp_audio_resampled.wav"})
    response_stt.raise_for_status()  # Lanzar una excepción en caso de error de solicitud

    # Obtener la transcripción de audio a texto
    transcription = response_stt.json().get("text")

    print(f"Transcripción de audio a texto exitosa: {transcription}")

    # Realizar la solicitud al servidor para obtener la respuesta del chatbot
    response_answer = requests.post(f"{url_servidor}/answer", json={"text": transcription})
    response_answer.raise_for_status()  # Lanzar una excepción en caso de error de solicitud

    # Obtener la respuesta del chatbot
    answer = response_answer.content

    # Guardar la respuesta del chatbot en un archivo de audio MP3 en Google Cloud Storage
    with open(ruta_archivo_audio_mp3_respuesta, "wb") as audio_file:
        audio_file.write(answer)

    upload_to_gcs(ruta_archivo_audio_mp3_respuesta, "audio_files/respuesta.mp3")

    print("Respuesta del chatbot generada correctamente")

except requests.exceptions.RequestException as e:
    print(f"Error en la solicitud: {e}")
except Exception as e:
    print(f"Error inesperado: {e}")
