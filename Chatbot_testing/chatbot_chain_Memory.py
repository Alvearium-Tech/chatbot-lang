from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_community.callbacks import get_openai_callback
from extract_apis_keys import load
from langchain.schema import format_document
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes

# Cargar la clave de la API de OpenAI
OPENAI_API_KEY = load()

class Chatbot:
    def __init__(self):
        if OPENAI_API_KEY is None:
            raise ValueError("OPENAI_API_KEY no puede ser None")
        self._embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        # Inicializar la memoria del chatbot para almacenar el historial de conversaciones
        self._memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")

    def load_database(self):
        # Cargar la base de datos de vectores utilizando el índice FAISS
        index_directory = "./faiss_index"
        persisted_vectorstore = FAISS.load_local(index_directory, self._embeddings)
        retriever = persisted_vectorstore.as_retriever()
        return retriever

    def combine_documents(self, docs, document_separator="\n\n"):
        # Combinar documentos en una sola cadena separada por el separador especificado
        doc_strings = [format_document(doc) for doc in docs]
        return document_separator.join(doc_strings)

    def invoke_chain(self):
        # Cargar el recuperador de la base de datos
        retriever = self.load_database()

        # Definir la plantilla de la solicitud de chat
        prompt_template = """Answer the question based only on the following context, if you don't know the answer, just say that you don't know, don't try to make up an :
                            {context}
                            Question: {question}
                            System: Answer according to the language you are typing in the prompt {question}. 
                            """
        # Crear un objeto de plantilla de chat a partir de la plantilla definida
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Definir la cadena de operaciones del modelo de lenguaje
        chain = (
            # Definir el contexto de la conversación recuperado del historial de conversaciones
            {"context": itemgetter("question") | retriever, "question": RunnablePassthrough()}
            # Agregar una plantilla de solicitud de chat para proporcionar contexto al modelo
            | prompt
            # Utilizar el modelo de lenguaje de OpenAI para generar una respuesta
            | ChatOpenAI(model="gpt-4-0125-preview", temperature=0.5)
            # Analizar la salida del modelo en una cadena legible
            | StrOutputParser()
        )

        while True:
            # Obtener la pregunta del usuario
            question = input("Pregunta:")
            with get_openai_callback() as cb:
                # Invocar la cadena de operaciones para generar una respuesta
                salida = chain.invoke({"question": question})
                # Imprimir información sobre el costo de la solicitud y la respuesta generada
                print(f"Total Cost (USD): ${format(cb.total_cost, '.20f')}")
                print(cb)
                print(salida)

                # Guardar el contexto de la conversación en la memoria del chatbot
                memo = self._memory.save_context({"question": question}, {"answer": salida})
                # Imprimir el contenido de la memoria para verificar que se haya guardado correctamente
                self._memory.load_memory_variables(memo)
                

if __name__ == "__main__":
    # Crear una instancia de la clase Chatbot
    my_chatbot = Chatbot()

    # Llamar al método invoke_chain() para iniciar la interacción con el usuario
    my_chatbot.invoke_chain()

    # Detener la ejecución del chatbot y esperar a que el usuario presione Enter para salir
    input("Presiona Enter para salir...")
