from pydantic import BaseModel
from typing import Optional, Any

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[list] = None

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
    #chat_tab: int | None = None 