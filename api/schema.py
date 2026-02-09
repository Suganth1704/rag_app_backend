from pydantic import BaseModel
from typing import Optional, Any

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[list] = None
    session_id: str
    user_id: str

class QueryResponse(BaseModel):
    question: Optional[str]
    chat_history: list
    answer: str

class IngestionResponse(BaseModel):
    status: str
    filename: str
    exception: Optional[str]

class AllSrcResponse(BaseModel):
    source: list[str]

class SessionCreateRequest(BaseModel):
    user_id: str 

class SessionCreateResponse(BaseModel):
    session_id: str
