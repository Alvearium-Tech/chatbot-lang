import os
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from extract_apis_keys import load
import re
import nltk
from unidecode import unidecode


def apiKeys():
    # Carga de la clave de la API OpenAI
    OPENAI_API_KEY = load()[1]
    openai_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    return openai_embeddings

class TextPreprocessor:
    
    def __init__(self, openai_embeddings) -> None:
        self.openai_embeddings = openai_embeddings
        self.charactersplit = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)

    def convert_to_utf8(self, no_utf8_directory, utf8_directory):
        for filename in os.listdir(no_utf8_directory):
            file_path = os.path.join(no_utf8_directory, filename)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                        content = unidecode(content)

                    utf8_file_path = os.path.join(utf8_directory, filename)
                    with open(utf8_file_path, 'w', encoding="utf-8") as f:
                        f.write(content)
                        
                except Exception as e:
                    print(f"Error al convertir '{filename}' a UTF-8: {e}")

    def text_transform(self, document):
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
        processed_text = sentence_tokenizer.tokenize(document)
        processed_text = ' '.join(processed_text)
        processed_text = re.sub(r'https?://\S+', '', processed_text)
        processed_text = re.sub(r'[^a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ\s]', '', processed_text)
        processed_text = processed_text.replace("a3", "ó")
        processed_text = processed_text.replace("A3", "ó")
        processed_text = processed_text.lower()
        
        return processed_text

    def preprocessor(self, utf8_directory):
        for filename in os.listdir(utf8_directory):
            file_path = os.path.join(utf8_directory, filename)
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        document = f.read()
                        processed_text = self.text_transform(document)

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(processed_text)

                except UnicodeEncodeError:
                    print(f"Error: No se pudo decodificar el archivo '{filename}' como UTF-8")
                    try:

                        with open(file_path, 'r', encoding="latin-1") as f:
                            document = f.read()
                            processed_text = self.text_transform(document)
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(processed_text)

                    except Exception as e:
                        print(f"Error al convertir '{filename}' a UTF-8: {e}")

                except FileNotFoundError:
                        print(f"Error: El archivo '{filename}' no se encontró")

                except IOError as e:
                        print(f"Error de E/S: {e}")
    

    def database(self, utf8_directory):
        embeddings = self.openai_embeddings
        all_documents = []
        for filename in os.listdir(utf8_directory):
            if filename.endswith(".txt"):
                file_path =  os.path.join(utf8_directory, filename)

                #Crear un text loader para cargar los textos
                loader = TextLoader(file_path, autodetect_encoding=True)
                documents = loader.load()

                # Dividir los documentos en fragmentos de texto
                texts = self.charactersplit.split_documents(documents)

                #almacenar todos los elementos en la lista
                all_documents.extend(texts)
        
        vectorstore =  FAISS.from_documents(all_documents, embeddings)
        vectorstore.save_local("faiss_index")


if __name__ == "__main__":
    openai_embeddings = apiKeys()
    text_processor = TextPreprocessor(openai_embeddings)
    no_utf8_directory = "TXT_no_UTF8"
    utf8_directory = "TXT_UTF8"
    text_processor.convert_to_utf8(no_utf8_directory, utf8_directory)
    text_processor.preprocessor(utf8_directory)
    text_processor.database(utf8_directory)