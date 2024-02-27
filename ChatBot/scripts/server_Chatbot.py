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
from google.cloud import texttospeech_v1 as texttospeech
from google.cloud import speech_v1 as speech
from operator import itemgetter
import librosa
from extract_apis_keys import load
import soundfile as sf
import tempfile
import pyaudio
import wave


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple API server using Langchain's Runnable interfaces",
)

UPLOAD_DIRECTORY = "audio_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Configura las credenciales de autenticación de Google Cloud
GOOGLE_APPLICATION_CREDENTIALS = load()

# Carga de la clave de la API OpenAI
OPENAI_API_KEY = load()[1]
openai_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Plantillas de conversación y respuesta
_TEMPLATE = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE = """Respond to the question based solely on the following context, ensuring that the response remains within the context of the provided conversation. Relate it to Alvearium, linking it to the following synonyms. The following synonyms are part of the meaning of Alvearium: Alvearium, Alveario, Albeo, Albio, Alvio, Avearium, Alveolar, Alveolado, Alveolario, Alveolaria, Alveolite, Alveari
Respond to the question based solely on the following context: {context}

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

# Función para formatear el historial del chat
def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

# Carga del índice de vectores
index_directory = "./faiss_index"
persisted_vectorstore = FAISS.load_local(index_directory, openai_embeddings)
retriever = persisted_vectorstore.as_retriever()

# Definición del mapeo de entrada y contexto
_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)
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
    _inputs | _context | ANSWER_PROMPT | ChatOpenAI(model="gpt-4-0125-preview") | StrOutputParser()
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
    
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

# Función Text-to-Speech (TTS) usando Google Cloud
def text_to_speech(text: str):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="es-ES", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)  # Ajustado para MP3
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return response.audio_content

async def speech_to_text(file: UploadFile = File(...)):
    try:
        # Guardar el archivo de audio temporalmente en el disco
        with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
            temp_audio.write(await file.read())
            temp_audio_path = temp_audio.name

        # Realizar la transcripción de voz
        transcription = await speech_to_text_internal(temp_audio_path)

        # Eliminar el archivo temporal
        os.remove(temp_audio_path)

        # Devolver directamente el texto transcribido
        return {"text": transcription}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Función Speech-to-Text (STT) actualizada para usar Google Cloud Speech-to-Text
async def speech_to_text_internal(audio_path: str) -> str:
    client = speech.SpeechClient()

    # Lee el archivo de audio
    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="es-ES",
    )

    # Detecta el texto del audio
    response = client.recognize(config=config, audio=audio)

    # Reúne los resultados de la transcripción
    if response.results:
        transcription = " ".join(result.alternatives[0].transcript for result in response.results)
        return transcription
    else:
        return "No se pudo transcribir el audio."
    
# Ruta para la grabación de audio
@app.post("/record_audio")
async def record_audio_endpoint(duration: int = 5):
    # Guardamos el archivo de audio grabado
    file_path_wav = os.path.join(UPLOAD_DIRECTORY, "recorded_audio.mp3")
    
    # Normalizar la ruta del archivo
    file_path_wav = os.path.normpath(file_path_wav)
    
    # Llama a la función de grabación de audio
    record_audio(file_path_wav, duration)
    
    return {"message": "Audio grabado exitosamente."}

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
    audio_content = text_to_speech(answer)
    
    # Actualizar el historial de chat global con la nueva conversación
    global_chat_history.append(("Usuario", question))
    global_chat_history.append(("Asistente", answer))
    
    # Crear un archivo temporal para almacenar el contenido de audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio_file:
        tmp_audio_file.write(audio_content)
        tmp_audio_file_path = tmp_audio_file.name

    # Devolver el archivo temporal como respuesta
    return FileResponse(tmp_audio_file_path, media_type="audio/mp3")

@app.post("/speech_to_text")
async def stt_endpoint(file: UploadFile = File(...)):
    try:
        # Crea un archivo temporal para guardar el contenido del archivo subido
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
        # Asegúrate de cerrar el archivo aquí, ya que with lo cierra automáticamente

        # Ahora que el archivo está cerrado, intenta acceder a él nuevamente
        transcription = await speech_to_text_internal(tmp_file_path)

        # Limpia el archivo temporal después de usarlo
        os.remove(tmp_file_path)

        return {"text": transcription}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ask_audio")
async def ask_question_audio(file: UploadFile = File(...)):
    try:
        # Leer el contenido del archivo de audio
        contents = await file.read()

        # Crear un archivo temporal para escribir el contenido
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            tmp_audio_file.write(contents)
            tmp_audio_file_path = tmp_audio_file.name

        # Resample a 16000 Hz si la frecuencia de muestreo no coincide
        y, sr = librosa.load(tmp_audio_file_path, sr=None)
        if sr != 16000:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        else:
            y_resampled = y

        # Escribir el audio resampleado en un archivo WAV
        output_path = "temp_audio_resampled.wav"
        sf.write(output_path, y_resampled, sr)

        # Devolver el archivo WAV como respuesta
        return FileResponse(output_path, media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def delete_file(path: str):
    os.remove(path)

# Ruta para la conversión de texto a voz (TTS)
@app.post("/text_to_speech")
async def generate_speech(text: str):
    # Configura el cliente de Google Cloud Text-to-Speech
    client = texttospeech.TextToSpeechClient()

    # Configuración de la solicitud
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code='en-US',
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Solicita la síntesis del texto
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    # Crea un archivo temporal para guardar el audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(response.audio_content)
        tmp_path = tmp.name

    # Devuelve el archivo de audio
    return FileResponse(path=tmp_path, filename="speech.mp3", media_type='audio/mpeg', background= BackgroundTasks(delete_file, tmp_path))

# Ruta para ver el historial del chat
@app.get("/chat_history")
async def view_chat_history():
    global global_chat_history
    # La función `view_chat_history` devuelve el historial global del chat
    return {"chat_history": global_chat_history}

# Manejar solicitudes para el ícono de favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Puedes devolver una imagen de ícono si tienes una
    return

add_routes(app, chain, enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
