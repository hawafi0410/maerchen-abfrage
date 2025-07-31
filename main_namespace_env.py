from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# 🔐 .env ladenx
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # wichtig für neue SDK

# 🧠 Initialisierung Pinecone und Embeddings
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 📦 FastAPI App
app = FastAPI()

class FrageInput(BaseModel):
    frage: str

@app.post("/frage")
async def frage_stellen(payload: FrageInput):
    frage = payload.frage
    dokumente = index.describe_index_stats()["namespaces"].keys()
    antworten = {}

    for name in dokumente:
        print(f"🔍 Frage an: {name}")
        vectorstore = LangchainPinecone(
            index=index,
            embedding=embeddings,
            text_key="text",
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
