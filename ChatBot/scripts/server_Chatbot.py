import os
from typing import List, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from langserve import add_routes
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.vectorstores.faiss import FAISS
from langchain_community.callbacks import get_openai_callback
from operator import itemgetter
from extract_apis_keys import load
import tempfile
import pyaudio
import wave
from fastapi.responses import JSONResponse
from openai import OpenAI
import subprocess
import base64
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple API server using Langchain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir solicitudes desde cualquier origen
    allow_credentials=True,  # Permitir el envío de credenciales (por ejemplo, cookies, tokens)
    allow_methods=["*"],  # Permitir todos los métodos HTTP (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permitir cualquier encabezado en la solicitud
    expose_headers=["*"],  # Exponer cualquier encabezado en la respuesta
    max_age=600,  # Duración máxima en segundos para la que las credenciales se pueden mantener en caché
)

UPLOAD_DIRECTORY = "audio_files"

os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Configura las credenciales de autenticación de Google Cloud
GOOGLE_APPLICATION_CREDENTIALS = load()

# Carga de la clave de la API OpenAI
OPENAI_API_KEY = load()[1]
openai_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# Plantillas de conversación y respuesta
_TEMPLATE = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE = """"You are a personal assistant for Alvearium Company, tasked with responding to questions based on the provided context, with a friendly and warm tone. You must also follow these instructions when generating a response:

###Instructions###
Follow these instructions to the letter, do not skip any:

1. You will rely solely on the provided context to answer the questions (this instruction is the most important, use only the provided information).
2. The language in which the question is written is the language in which you must respond.
3. Responses must contain a maximum of 50 words, they cannot exceed this limit.
4. Keep in mind that the words "Alvearium, alvearium" have synonyms such as "Alveariun, albearium, albeariun, alvear, alveol, alveolar, salbearium, salvearium, alveary, albeary, alvearium, albearium, alveary, alveolo," among others.
5. You must answer the following questions according to the following examples:
Example 1:
Q: "Hola, ¿cómo estás?"
A: "Bien, gracias, estoy aquí esperando para ayudarte con lo que necesites."
Example 2:
Q: "Hola, ¿quién eres?"
A: "Hola, encantado de conocerte, soy Alvy, tu asistente personal, estoy aquí para ayudarte con lo que necesites."
Example 3:
Q: "Hola"
A: "¡Hola! ¿Quién eres? Soy Alvy, tu asistente personal, estoy aquí para ayudarte con lo que necesites."
6. If you don't know the answer based solely on the provided context, respond to the user according to the following examples:
A: "I didn't quite understand you, please repeat your question."
A: "Can you repeat the question, please?"
7. Double-check the information before responding, you can only respond based on the provided context.
8. You can only tell the truth.
###Your main objective using all the above information, instructions, and provided context###
Answer the following question based solely on the provided context, the information you will use to respond is the context we will provide, you can only tell the truth. The context is the following:" {context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)


DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

# Función para combinar documentos
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

MAX_CHAT_HISTORY_LENGTH = 6

# Función para formatear el historial del chat
def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    truncated_history = chat_history[-MAX_CHAT_HISTORY_LENGTH:]
    for dialogue_turn in truncated_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

# Carga del índice de vectores
index_directory = "./faiss_index"
persisted_vectorstore = FAISS.load_local(index_directory, openai_embeddings, allow_dangerous_deserialization=True)
retriever = persisted_vectorstore.as_retriever(search_type="mmr")

'''index_directory = "./faiss_index"
persisted_vectorstore = FAISS.load_local(index_directory, openai_embeddings)
retriever = persisted_vectorstore.as_retriever(search_type="mmr")'''

# Definición del mapeo de entrada y contexto
_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0.7)
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}

# Definición del modelo de entrada del historial de chat
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str

# Cadena de procesamiento de la conversación
conversational_qa_chain = (
    _inputs | _context | ANSWER_PROMPT | ChatOpenAI(model="gpt-4o", max_tokens=300, temperature=0.7) | StrOutputParser()
)
chain = conversational_qa_chain.with_types(input_type=ChatHistory)

# Variable global para almacenar el historial del chat
global_chat_history = []


# Función para grabar audio
def record_audio(file_path: str, duration: int = 10):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = duration
    
    audio = pyaudio.PyAudio()
    
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    print("Recording...")
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("Finished recording.")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Guardar el audio en formato WAV temporal
    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_file_path = wav_file.name
    wav_file.close()
    
    with wave.open(wav_file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    # Convertir el archivo WAV a MP3
    mp3_file_path = os.path.splitext(file_path)[0] + ".mp3"
    subprocess.run(['ffmpeg', "-y", "-i", wav_file_path, "-codec:a", "libmp3lame", mp3_file_path])
    
    # Leer el contenido del archivo MP3 como bytes
    with open(mp3_file_path, 'rb') as mp3_file:
        audio_content = mp3_file.read()
    
    # Eliminar el archivo WAV temporal
    os.remove(wav_file_path)
    
    return audio_content

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
    
# Ruta para la grabación de audio
@app.post("/record_audio")
async def record_audio_endpoint(duration: int = 10):
    file_path = os.path.join(UPLOAD_DIRECTORY, "recorded_audio.mp3")
    file_content = record_audio(file_path, duration)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio_file:
        tmp_audio_file.write(bytes(file_content))
        tmp_audio_file_path =  tmp_audio_file.name

    # Devolver el archivo temporal como respuesta
    return FileResponse(
        path=tmp_audio_file_path,
        filename="recorded_audio.mp3",
        media_type='audio/mpeg',
        background=BackgroundTasks([lambda: delete_file(tmp_audio_file_path)])
    )

# Define la ruta y la función controladora para manejar las solicitudes POST

@app.post("/answer")
async def get_answer(request_body: dict):
    global global_chat_history
    
    # Extraer la pregunta del cuerpo de la solicitud
    question = request_body.get("text")  # Cambiado de "question" a "text"
    if not question:
        raise HTTPException(status_code=400, detail="Transcripción no proporcionada en el cuerpo de la solicitud.")
    
    # Extraer el historial de chat del cuerpo de la solicitud, o usar una lista vacía si no está presente
    chat_history = request_body.get("chat_history", [])
    
    # Llama a tu lógica existente para obtener la respuesta
    with get_openai_callback() as cb:
        answer = chain.invoke({"chat_history": chat_history, "question": question})  # Cambiado "respuesta" por "answer"
        print(cb)
        # Si ocurrió algún error al obtener la respuesta, lanza una excepción HTTP
        if not answer:
            raise HTTPException(status_code=500, detail="Error al procesar la pregunta")
    
    # Convertir la respuesta del chatbot a audio utilizando la función text_to_speech
    file_path = os.path.join(UPLOAD_DIRECTORY, "respuesta.mp3")
    audio_content = text_to_speech(answer, file_path)

    
    # Actualizar el historial de chat global con la nueva conversación
    global_chat_history.append(("Usuario", question))
    global_chat_history.append(("Asistente", answer))
    
    base_url = "https://mwy0tuecpg.execute-api.eu-central-1.amazonaws.com"

    # Construir la URL completa del archivo de audio
    audio_file_path_mp3 = "audio_files/respuesta.mp3"
    audio_url_mp3 = f"{base_url}/{audio_file_path_mp3}"

    audio_file_path_wav = "audio_files/respuesta.wav"
    audio_url_wav = f"{base_url}/{audio_file_path_wav}"
    
    # Codificar el contenido de audio a Base64
    audio_base64 = base64.b64encode(audio_content).decode('utf-8')

    response_data = {
        "audio_url_mp3": audio_url_mp3,  # Cambiado de "audio_base64" a "audio_url"
        "text_response": answer,
        "audio_url_wav": audio_url_wav,
    }

    # Devolver el contenido del archivo temporal como respuesta
    return JSONResponse(content=response_data)

@app.post("/speech_to_text")
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

def delete_file(path: str):
    os.remove(path)

# Ruta para la conversión de texto a voz (TTS)
@app.post("/text_to_speech")
async def generate_speech(text: str) -> bytes:
    try:
        response = text_to_speech(text)  # Llama a la función para generar el audio
        
        # Lee el contenido del archivo temporal
        audio_content = response.read()

        # Devuelve el contenido de audio como respuesta
        return audio_content

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ruta para ver el historial del chat
@app.get("/chat_history")
async def view_chat_history():
    global global_chat_history
    # La función `view_chat_history` devuelve el historial global del chat
    return {"chat_history": global_chat_history}

# Ruta para servir archivos de audio
@app.get("/audio_files/{file_name}")
async def get_audio_file(file_name: str):
    file_path = os.path.join("audio_files", file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg")
    else:
        return {"error": "Archivo no encontrado"}

# Manejar solicitudes para el ícono de favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Puedes devolver una imagen de ícono si tienes una
    favicon_path = "ChatBot\scripts\cropped-cropped-favicon-01-32x32.png"
    return favicon_path

# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, chain, enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
