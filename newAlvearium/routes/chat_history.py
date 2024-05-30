from fastapi import APIRouter
from routes.answer import global_chat_history


app_chat_history = APIRouter()

# Ruta para ver el historial del chat
@app_chat_history.get("/chat_history", description="Esta ruta unicamente trae el historial del chatbot")
async def view_chat_history():
    global global_chat_history
    # La función `view_chat_history` devuelve el historial global del chat
    return {"chat_history": global_chat_history}