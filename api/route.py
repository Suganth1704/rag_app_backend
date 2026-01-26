import os
from fastapi import APIRouter, UploadFile, File
from .schema import QueryRequest, QueryResponse, IngestionResponse
from ..llm.chat import get_rag_graph
from ..ingestion.ingest import start_ingest
import shutil

import logging
logger = logging.getLogger(__name__)

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(prefix="/api", tags=["RAG"])

def delete_file_after_ingest(path:str) -> None:
    os.remove(path)

@router.get("/health")
def health() -> dict:
    return {"status":"Okay"}

@router.post("/query", response_model=QueryResponse)
def query_rag(request:QueryRequest) -> QueryResponse:
    rag_graph = get_rag_graph()
    res = rag_graph.invoke({
        "question":request.query,
        "chat_history": request.chat_history
    })
    return QueryResponse(
        question=res['question'],
        chat_history=res['chat_history'],
        answer=res['answer']
    )

@router.post("/upload_and_ingest", response_model=IngestionResponse)
def upload_and_ingest(file: UploadFile = File(...)) -> IngestionResponse:
    exception = None
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"{file_path} saved")

        start_ingest()
        logger.info(f"Ingestion is done, deleting {file_path}...")
        delete_file_after_ingest(file_path)
    except Exception as ex:
        exception=ex
        logger.error(f"Ingestion Filed with the {ex}")

    return IngestionResponse(
        status="Success" if not exception else "Failed",
        filename=file.filename,   
        exception="NA" if not exception else str(exception)
    )