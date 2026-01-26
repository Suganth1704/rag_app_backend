from pydantic import BaseModel
from typing import Optional, Any

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[list]

class QueryResponse(BaseModel):
    question: Optional[str]
    chat_history: list
    answer: str

class IngestionResponse(BaseModel):
    status: str
    filename: str
    exception: Optional[str]