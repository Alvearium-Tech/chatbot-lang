from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
from extract_apis_keys import load

# Cargar la clave de la API de OpenAI
OPENAI_API_KEY = load()

class chatbot():
    def __init__(self):
        if OPENAI_API_KEY is None:
            raise ValueError("OPENAI_API_KEY no puede ser None")
        # Inicialización de los atributos del chatbot
        self._embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self._persisted_vectorstore = None
        self._retriever = None
        self._template = None
        self._prompt = None
        self._model = ChatOpenAI(model="gpt-3.5-turbo-0125")
        self._chain = None
        self._salida = None
        self._callback = None

    def load_database(self, _embeddings):
        # Cargar la base de datos de vectores utilizando el índice FAISS
        index_directory = "./faiss_index"
        persisted_vectorstore = FAISS.load_local(index_directory, _embeddings)
        retriever = persisted_vectorstore.as_retriever()
        return retriever

    def chain_template(self, template, prompt):
        # Crear una plantilla de solicitud de chat
        template = """Answer the question based only on the following context:
                    {context}

                    Question: {question}

                    System: Answer according to the language you are typing in the prompt {question}. 
                    """
        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    def invoke_chain(self):
        # Cargar el recuperador de la base de datos
        retriever = self.load_database(_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY))
        # Crear una plantilla de solicitud de chat
        prompt = self.chain_template(template=None, prompt=None)

        # Definir la cadena de operaciones del modelo de lenguaje
        chain = (
            {'context': retriever, "question" : RunnablePassthrough()}
            | prompt
            | self._model
            | StrOutputParser()
        )

        while True:
            # Obtener la pregunta del usuario
            question = input("Pregunta:")
            with get_openai_callback() as cb:
                # Invocar la cadena de operaciones para generar una respuesta
                salida = chain.invoke(question)
                # Imprimir información sobre el costo de la solicitud y la respuesta generada
                print(f"Total Cost (USD): ${format(cb.total_cost, '.20f')}")
                print(cb)
                print(salida)

if __name__ == "__main__":
    # Crear una instancia de la clase Chatbot
    my_chatbot = chatbot()

    # Llamar al método invoke_chain() para iniciar la interacción con el usuario
    my_chatbot.invoke_chain()

    # Esperar a que el usuario presione Enter para salir
    input("Presiona Enter para salir...")
