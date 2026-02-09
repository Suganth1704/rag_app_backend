import os
from fastapi import APIRouter, UploadFile, File
from .schema import * #QueryRequest, QueryResponse, IngestionResponse, AllSrcResponse,SessionCreateRequest
from ..llm.chat import get_rag_graph, get_history
from ..ingestion.ingest import start_ingest
from ..vector_store.chroma_client import RagAppChromaClient

import shutil
import uuid

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

@router.post("/session", response_model=SessionCreateResponse)
def create_session(request:SessionCreateRequest) -> SessionCreateResponse:
    session_id = str(uuid.uuid4())
    history =get_history(user_id=request.user_id, session_id=session_id)
    history.clear()
    return SessionCreateResponse(
        session_id=session_id
    )

@router.post("/query", response_model=QueryResponse)
def query_rag(request:QueryRequest) -> QueryResponse:
    rag_graph = get_rag_graph()
    history = get_history(user_id=request.user_id, session_id=request.session_id)
    res = rag_graph.invoke({
        "question":request.query,
        "chat_history": history.messages
    })
    history.add_user_message(request.query)
    history.add_ai_message(res['answer'])
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

@router.get("/get_all_src", response_model=AllSrcResponse)
def get_available_srcs() -> AllSrcResponse:
    chroma_client = RagAppChromaClient()
    res = chroma_client.get_available_srcs()
    return AllSrcResponse(
        source=res
    )