from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.audio import app_audio
from routes.chat_history import app_chat_history
from routes.answer import app_answer
from routes.data_preprocessor import app_data

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="API entera del chatbot Alvy",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir solicitudes desde cualquier origen
    allow_credentials=True,  # Permitir el envío de credenciales (por ejemplo, cookies, tokens)
    allow_methods=["*"],  # Permitir todos los métodos HTTP (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permitir cualquier encabezado en la solicitud
    expose_headers=["*"],  # Exponer cualquier encabezado en la respuesta
    max_age=1000,  # Duración máxima en segundos para la que las credenciales se pueden mantener en caché
)

app.include_router(app_audio)
app.include_router(app_chat_history)
app.include_router(app_answer)
app.include_router(app_data)