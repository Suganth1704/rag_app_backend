import os
import chromadb
from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from typing import Any

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import logging
logger = logging.getLogger(__name__)

CHROMA_API_KEY = os.getenv('CHROMA_API_KEY')
TENANT = os.getenv('TENANT')
CHROMA_DB = os.getenv('CHROMA_DB')
RAG_APP_COLL = os.getenv('RAG_APP_COLL')
EMBED_MODEL = 'all-MiniLM-L6-v2'

class RagAppChromaClient(object):
    
    def __init__(self):
        self._db = CHROMA_DB
        self._collection_name = RAG_APP_COLL
        self._tries = 1
        while self._tries <= 3:
            try:
                logger.info("Connecting to chroma....")
                logger.info(f"Try: {self._tries}")
                self._client = chromadb.CloudClient(
                                api_key=CHROMA_API_KEY,
                                tenant=TENANT,
                                database=self._db)
                break
            except Exception as ex:
                logger.error(f"{ex}")
                self._tries += 1
        self._embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        self._collection = self._client.get_or_create_collection(
                                                    name=self._collection_name,
                                                    embedding_function=self._embedding_function,
                                                    metadata={"hnsw:space": "cosine"},
                                                    )
        self._hug_embeddings = HuggingFaceEmbeddings(
                                                model_name=f"sentence-transformers/{EMBED_MODEL}"
                                                )
        self._batch_size = 30
    def add_documents(self,texts:list[Document], metadata:list[dict[str, Any]], id:list[str]) -> None:
        # To avoid NUM_RECORDS error
        for i in range(0, len(texts),self._batch_size):
            if not self.source_exists(metadata):
                logger.info(f"Loading from the source {''.join(set(list(map(lambda x: x['source'], metadata))))} to {self._collection_name} collection in {self._db} chroma")
                self._collection.add(
                    documents=texts[i:i+self._batch_size],
                    metadatas=metadata[i:i+self._batch_size],
                    ids=id[i:i+self._batch_size]
                ) 
            else:
                logger.info(f"{''.join(set(list(map(lambda x: x['source'], metadata))))} source already exist in data base")
                logger.info("Skipping ingestion ...")

    def source_exists(self,metadata:list[dict[str, Any]]) -> bool:
        source = ''.join(set(list(map(lambda x: x['source'], metadata))))
        res = self._collection.get(
                    where={"source":source},
                    limit=1
        )
        return True if res['metadatas'] else False


    def mmr_search(self, query:str) -> list[Document]:
        logger.info(f"Starting MMR search... | Query : {query}")
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
        logger.info("Starting searching from chroma... | Query : {query}")
        results = self._collection.query(
            query_texts=[query],
            n_results = 3
        )
        return results