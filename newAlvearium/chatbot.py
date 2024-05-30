import os
from typing import List, Tuple
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from operator import itemgetter
from extract_apis_keys import load
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from langchain.vectorstores.faiss import FAISS


UPLOAD_DIRECTORY = "audio_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

#Cargar todo lo necesario
OPENAI_API_KEY = load()[0]
INDEX_NAME =  load()[2]
PINECONE_API_KEY = load()[1]
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

ANSWER_TEMPLATE = """"You are a personal assistant for Alvearium Company, tasked with responding to questions based on the provided context, with a friendly and warm tone. You must also follow the following instructions when generating a response:

###Instructions###

1. You will rely solely on the provided context to answer the questions (This instruction is the most important).
2. The language in which the question is written is the language in which you must respond.
3. Responses must contain a maximum of 50 words, they cannot exceed this limit.
4. Keep in mind that the words "Alvearium, alvearium" have synonyms such as "Alveariun, albearium, albeariun, alvear, alveol, alveolar, salbearium, salvearium, alveary, albeary, alvearium, albearium, alveary, alveolo", among others.
5. You must answer the following questions according to the following examples:
Example 1:
Q: "Hi, how are you?"
A: "Good, thank you, I'm here waiting to help you with whatever you need."
Example 2:
Q: "Hello, who are you?"
A: "Hello, nice to meet you, I'm Alvy, your personal assistant, i'm here to help you with anything you need."
Example 3:
Q: "Hello"
A: "Hello! Who are you? I'm Alvy, your personal assistant, i'm here to help you with anything you need."
6. If you don't know the answer based solely on the following context, respond to the user based on the following examples:
A: "I didn't quite understand you, please repeat your question."
A: "Can you repeat the question, please?"

###Your objective using all the above information###
Answer the following question based solely on the provided context. The context is as follows:" {context}

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

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> str:
    """Format chat history into a string."""
    # Mueve la importación dentro de la función para evitar el ciclo de importación
    from routes.answer import global_chat_history
    
    if not chat_history:
        return ""  # Devolver una cadena vacía si el historial del chat está vacío
    
    buffer = ""
    truncated_history = chat_history[-MAX_CHAT_HISTORY_LENGTH:]
    for dialogue_turn in truncated_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    
    # Actualiza global_chat_history con el chat_history actualizado
    chat_history = global_chat_history
    
    return buffer

# Carga del índice de vectores
'''index_directory = "./faiss_index"
persisted_vectorstore = FAISS.load_local(index_directory, openai_embeddings, allow_dangerous_deserialization=True)
retriever = persisted_vectorstore.as_retriever(search_type="mmr")'''

'''index_directory = "./faiss_index"
persisted_vectorstore = FAISS.load_local(index_directory, openai_embeddings)
retriever = persisted_vectorstore.as_retriever(search_type="mmr")'''

persisted_vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=openai_embeddings)
retriever = persisted_vectorstore.as_retriever() 


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

#global_chat_history = []