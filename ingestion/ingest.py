import os
from typing import List

from langchain_community.document_loaders import (Docx2txtLoader, PyPDFLoader,TextLoader, PyMuPDFLoader)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

from vector_store.chroma_client import RagAppChromaClient

import logging
logger = logging.getLogger(__name__)

DATA_DIR='data'

def load_docs() -> List[Document]:
    logger.info("Loading documents")
    docs = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        logger.info(f"Source : {path}")
        if path.endswith('.pdf'):
            docs.extend(PyMuPDFLoader(path).load())
        elif path.endswith('.txt'):
            docs.extend(TextLoader(path).load())
        elif path.endswith('.docx'):
            docs.extend(Docx2txtLoader(path).load())
        else:
            logger.info("Loading an unstructured document")
            loader = UnstructuredLoader(path,
                                mode="elements",
                                strategy="auto")
            docs.extend(loader.load())
        
    return docs

def get_chunks() -> List[Document]:
    logger.info("Splitting chunks")
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = splitter.split_documents(load_docs())
    return chunks

def start_ingest() -> None:
    logger.info("Starting ingestion ...")
    document_chunks = get_chunks()
    ragAppChromaDb = RagAppChromaClient()

    documents = [chunk.page_content for chunk in document_chunks]
    metadata = [{"source":chunk.metadata.get("source"), "page": chunk.metadata.get("page", 0)} for chunk in document_chunks]
    ids=[f"id{i}" for i in range(len(document_chunks))]
    ragAppChromaDb.add_documents(
        texts=documents,
        metadata=metadata,
        id=ids
    )
