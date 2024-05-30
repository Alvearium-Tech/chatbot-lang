import os
from config import Config

# Define la ruta al archivo .env
env_file_path = os.path.join(os.path.dirname(__file__), '.env')

# Cargar la configuración desde el archivo .env
config = Config(stream_or_path=env_file_path)

def load():
    # Configurar la variable de entorno OPENAI_API_KEY
    os.environ['OPENAI_API_KEY'] = config.get("OPENAI_API_KEY") or "MY_OPENAI_API_KEY"        
    os.environ['PINECONE_API_KEY'] =  config.get("PINECONE_API_KEY") or "MY_PINECONE_API_KEY"
    os.environ['PINECONE_INDEX_NAME'] = config.get("PINECONE_INDEX_NAME") or "MY_PINECONE_INDEX_NAME"

    return (os.environ['OPENAI_API_KEY'], os.environ['PINECONE_API_KEY'], os.environ['PINECONE_INDEX_NAME'])
