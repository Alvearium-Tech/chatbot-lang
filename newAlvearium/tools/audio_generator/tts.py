from openai import OpenAI
from extract_apis_keys import load
import subprocess
import os

# Carga de la clave de la API OpenAI
OPENAI_API_KEY = load()[0]

UPLOAD_DIRECTORY = "newAlvearium\\tools\\audio_generator\\audio"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


def text_to_speech(text: str, save_path: str) -> bytes:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )

        # Guardar el audio en formato MP3
        mp3_file_path = save_path
        mp3_subprocess = subprocess.Popen(['ffmpeg', "-y", "-i", "pipe:0", "-codec:a", "libmp3lame", mp3_file_path], stdin=subprocess.PIPE)
        mp3_subprocess.communicate(input=response.read())
        mp3_subprocess.wait()

        # Convertir el archivo MP3 a WAV
        wav_file_path = save_path.replace('.mp3', '.wav')
        wav_subprocess = subprocess.run(['ffmpeg', '-y', '-i', mp3_file_path, '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', wav_file_path], check=True)

        # Leer el contenido del archivo WAV como bytes
        with open(wav_file_path, 'rb') as audio_file:
            audio_content = audio_file.read()

        return audio_content
    
    except Exception as e:
        print(e)

text = '''Bienvenido a Grecia! Lugar especial en el musical Mamma Mia, disfruta del trailer del musical en la pantalla gigante!
Si te animas a comprar una entrada, puedes hacerlo aquí o pasar por nuestro Marketplace. 
Pulsa el botón y disfruta de nuestra oferta de productos y servicios!
'''

file_path = os.path.join(UPLOAD_DIRECTORY, "Preview_Mamma_Mia_1.mp3")
text_to_speech(text, file_path)