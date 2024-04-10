import os
from config import Config

# Define la ruta al archivo .env
env_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')

# Cargar la configuración desde el archivo .env
config = Config(stream_or_path=env_file_path)

def load():
    # Configurar la variable de entorno OPENAI_API_KEY
    os.environ['OPENAI_API_KEY'] = config.get("OPENAI_API_KEY") or "MY_OPENAI_API_KEY"        
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.get("GOOGLE_APPLICATION_CREDENTIALS") or "MY_GOOGLE_APPLICATION_CREDENTIALS"
    os.environ['MONGODB_ATLAS_CLUSTER_URI'] =  config.get("MONGODB_ATLAS_CLUSTER_URI") or "MY_MONGODB_ATLAS_CLUSTER_URI"

    return (os.environ['GOOGLE_APPLICATION_CREDENTIALS'], os.environ['OPENAI_API_KEY'], os.environ['MONGODB_ATLAS_CLUSTER_URI'])
