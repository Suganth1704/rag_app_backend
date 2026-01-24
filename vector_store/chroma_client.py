import os
import chromadb
from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from typing import Any

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

CHROMA_API_KEY = os.getenv('CHROMA_API_KEY')
TENANT = os.getenv('TENANT')
CHROMA_DB = os.getenv('CHROMA_DB')
RAG_APP_COLL = os.getenv('RAG_APP_COLL')
EMBED_MODEL = 'all-MiniLM-L6-v2'

class RagAppChromaClient(object):
    
    def __init__(self):
        self._db = CHROMA_DB
        self._collection_name = RAG_APP_COLL
        self._client = chromadb.CloudClient(
                        api_key=CHROMA_API_KEY,
                        tenant=TENANT,
                        database=self._db)
        self._embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        self._collection = self._client.get_or_create_collection(
                                                    name=self._collection_name,
                                                    embedding_function=self._embedding_function,
                                                    metadata={"hnsw:space": "cosine"},
                                                    )
        self._hug_embeddings = HuggingFaceEmbeddings(
                                                model_name=f"sentence-transformers/{EMBED_MODEL}"
                                                )
    def add_documents(self,texts:list[Document], metadata:list[dict[str, Any]], id:list[str]) -> None:
        self._collection.add(
            documents=texts,
            metadatas=metadata,
            ids=id
        )        

    def mmr_search(self, query:str) -> list[Document]:
        vector_db = Chroma(
            client=self._client,
            collection_name=self._collection_name,
            embedding_function=self._hug_embeddings
        )
        retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 12
            }
        )
        docs = retriever.invoke(query)
        return docs
    
    def qurey_chroma(self, query:str) -> dict[str, Any]:
        results = self._collection.query(
            query_texts=[query],
            n_results = 3
        )
        return results