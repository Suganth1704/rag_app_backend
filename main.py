from ingestion.ingest import start_ingest
from vector_store.chroma_client import RagAppChromaClient
from llm.chat import get_rag_graph, get_history

from config.logging import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Start execution ...")
    start_ingest()
    cnt = 1
    session_id = f"user-{cnt}"
    hist = get_history(session_id=session_id)
    while True:
        
        question = input("Ask your question: \n")
        rag_graph = get_rag_graph()
        res = rag_graph.invoke({
            "question":question,
            "chat_history": hist.messages
        })
        hist.add_user_message(question)
        hist.add_ai_message(res["answer"])

        print(res["answer"])

        c = input("Do you want to continue ? (y/n)")
        if c.lower() == 'y':
            cnt+=1
        else:
            break

    print()
    