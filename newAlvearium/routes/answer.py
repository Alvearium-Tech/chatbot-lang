from fastapi import APIRouter, HTTPException
import os
from chatbot import chain, UPLOAD_DIRECTORY
from langchain_community.callbacks import get_openai_callback
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel


url_servidor = "https://mwy0tuecpg.execute-api.eu-central-1.amazonaws.com"

app_answer = APIRouter()

class AnswerRequest(BaseModel):
    text: str

global_chat_history = []

@app_answer.post("/answer", description="Esta ruta es a la cual mandamos los mensajes escritos al chatbot para que los responda")
async def get_answer(request_body: AnswerRequest):
    global global_chat_history

    question = request_body.text
    if not question:
        raise HTTPException(status_code=400, detail="Transcripción no proporcionada en el cuerpo de la solicitud.")

    async with httpx.AsyncClient() as client:
        # Obtener el historial actual
        chat_history_response = await client.get(f"{url_servidor}/chat_history")
        if chat_history_response.status_code != 200:
            raise HTTPException(status_code=chat_history_response.status_code, detail="Error fetching chat history")
        chat_history_data = chat_history_response.json()

    # Obtener el historial de chat
    chat_history = chat_history_data.get('chat_history', [])

    # Verificar si el historial actual supera la longitud máxima deseada
    MAX_CHAT_HISTORY_LENGTH = 6
    if len(chat_history) >= MAX_CHAT_HISTORY_LENGTH:
        chat_history = chat_history[-MAX_CHAT_HISTORY_LENGTH:]  # Mantener solo los últimos 6 elementos

    # Agregar la nueva pregunta y respuesta al historial
    new_entry1 = ("question", question)
    chat_history.append(new_entry1)

    question = question + "\nEl chat history es el siguiente: " + str(chat_history)

    with get_openai_callback() as cb:
        try:
            answer = chain.invoke({"chat_history": chat_history, "question": question})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing the question: {repr(e)}")

        if not answer:
            raise HTTPException(status_code=500, detail="Error generating an answer")

    new_entry2 = ("answer", answer)
    chat_history.append(new_entry2)


    # Agregar la nueva entrada al historial global
    global_chat_history.append(new_entry1)
    global_chat_history.append(new_entry2)

    file_path = os.path.join(UPLOAD_DIRECTORY, "respuesta.wav")

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{url_servidor}/text_to_speech", json={"text": answer, "save_path": file_path})
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error en la conversión de texto a voz")

    base_url = "https://mwy0tuecpg.execute-api.eu-central-1.amazonaws.com"

    #audio_file_path_mp3 = "audio_files/respuesta.mp3"
    #audio_url_mp3 = f"{base_url}/{audio_file_path_mp3}"

    audio_file_path_wav = "newAlvearium/audio_files/respuesta.wav"
    audio_url_wav = f"{base_url}/{audio_file_path_wav}"

    response_data = {
        #"audio_url_mp3": audio_url_mp3,
        "text_response": answer,
        "audio_url_wav": audio_url_wav,
    }

    return JSONResponse(content=response_data)
