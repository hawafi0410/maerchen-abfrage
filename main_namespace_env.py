from fastapi import FastAPI
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore  # NEU: neue Klasse aus neuem Paket
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

# Pinecone initialisieren
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# FastAPI App
app = FastAPI()

class FrageInput(BaseModel):
    frage: str

@app.post("/frage")
async def frage_stellen(payload: FrageInput):
    frage = payload.frage
    antworten = {}

    try:
        dokumente = index.describe_index_stats()["namespaces"].keys()
    except Exception as e:
        return {"antworten": {"Fehler beim Zugriff auf Index": str(e)}}

    for name in dokumente:
        print(f"🔍 Frage an: {name}")
        vectorstore = PineconeVectorStore.from_existing_index(  # NEU
            index_name=INDEX_NAME,
            embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
            namespace=name
        )
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        try:
            antwort = qa_chain.run(frage)
            if antwort and "keine Antwort" not in antwort.lower():
                antworten[name] = antwort
        except Exception as e:
            antworten[name] = f"Fehler: {str(e)}"

    return {"antworten": antworten}
