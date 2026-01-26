
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph, END

from typing import TypedDict

from .groq_client import GroqClient
from ..vector_store.chroma_client import RagAppChromaClient

import logging
logger = logging.getLogger(__name__)

class RAGState(TypedDict):
    question:str
    chat_history:list[BaseMessage]
    context: list[Document]
    answer:str

def retriever(state:RAGState) -> RAGState:
    logger.info(f'Retrieving from Chroma ...')
    chromaClient=RagAppChromaClient()
    docs=chromaClient.mmr_search(query=state["question"])
    return {
        **state,
        "context": docs
    }

def generate(state: RAGState) -> RAGState:
    logger.info("Generating the answer on the given context")
    groqClient = GroqClient()
    llm = groqClient._llm
    context_prompt_tmpl = groqClient.get_context_prompt_template()
    resp = llm.invoke(
        context_prompt_tmpl.format_messages(
            question=state["question"],
            chat_history=state["chat_history"],
            context="\n\n".join(d.page_content for d in state["context"])
        )
    )
    return {
        **state,
        "answer": resp.content
    }


def get_rag_graph():
    graph = StateGraph(RAGState)

    logger.info("Building graph ...")
    graph.add_node("retriever", retriever)
    graph.add_node("generate", generate)
    
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "generate")
    graph.add_edge("generate", END)

    logger.info("Compiling graph ...")
    rag_graph = graph.compile()
    return rag_graph

store = {}
def get_history(session_id:str):
    if len(store) > 5:
        store.popitem()
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
