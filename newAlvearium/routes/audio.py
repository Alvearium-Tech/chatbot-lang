import os
from fastapi import File, UploadFile, HTTPException
from extract_apis_keys import load
from fastapi.responses import JSONResponse
from openai import OpenAI
import subprocess
from fastapi import APIRouter
from pydantic import BaseModel

app_audio =  APIRouter()

class TextToSpeechRequest(BaseModel):
    text: str
    save_path: str

UPLOAD_DIRECTORY = "audio_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

OPENAI_API_KEY = load()[0]

client = OpenAI(api_key=OPENAI_API_KEY)

def text_to_speech(text: str, save_path: str) -> bytes:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )

        audio_data = response.read()

        wav_file_path = save_path
        with subprocess.Popen(
            ['ffmpeg', '-y', '-i', 'pipe:0', wav_file_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        ) as wav_subprocess:
            stdout, stderr = wav_subprocess.communicate(input=audio_data)
            if wav_subprocess.returncode != 0:
                raise Exception(f"ffmpeg error: {stderr.decode()}")

        print(f"Received audio data length: {len(audio_data)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def speech_to_text_internal(file_path: str) -> str:
    try:
        print(file_path)
        with open(file_path, "rb") as file:
        # Detecta el texto del audio
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=file,
                response_format="text",
                prompt="Alvearium, alvea"
            )

        return transcription
    
    except Exception as e:
        raise Exception(f"Error en la transcripción de voz a texto: {e}")
    

@app_audio.post("/speech_to_text", description="A esta ruta subimos audios .MP3 para poder obtener su transcripcion")
async def stt_endpoint(file: UploadFile = File(...)):
    try:
        # Guardar el archivo de audio en el directorio de almacenamiento
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Llamar a la función de transcripción de voz a texto con la ruta del archivo
        transcription = await speech_to_text_internal(file_path)

        # Eliminar el archivo después de procesarlo si es necesario
        os.remove(file_path)

        return {"text": transcription}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app_audio.post("/text_to_speech", description="En esta ruta podemos escribir un texto y generar el audio utilizado por el chatbot")
def generate_speech(request_body: TextToSpeechRequest ) -> bytes:
    try:
        text = request_body.text
        save_path = request_body.save_path


        response = text_to_speech(text, save_path)  # Llama a la función para generar el audio
        
        base_url = "https://mwy0tuecpg.execute-api.eu-central-1.amazonaws.com"

        # Construir la URL completa del archivo de audio
        #audio_file_path_mp3 = "audio_files/respuesta.mp3"
        #audio_url_mp3 = f"{base_url}/{audio_file_path_mp3}"

        audio_file_path_wav = "newAlvearium/audio_files/respuesta.wav"
        audio_url_wav = f"{base_url}/{audio_file_path_wav}"

        response_data = {
        #"audio_url_mp3": audio_url_mp3,
        "audio_url_wav": audio_url_wav,
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Manejar solicitudes para el ícono de favicon
@app_audio.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Puedes devolver una imagen de ícono si tienes una
    return