from ingestion.ingest import start_ingest
from vector_store.chroma_client import RagAppChromaClient
from llm.groq_client import GroqClient

if __name__ == "__main__":
    print('start')
    start_ingest()
    obj = RagAppChromaClient()
    groq_client = GroqClient()
    q = 'Importance of Character in Ethics?'
    r = obj.mmr_search(query=q)
    #r = obj.qurey_chroma(query=q)
    context = "\n".join([d.page_content for d in r])
    prompt = f"""
    
    Answer using only the context below.

    Context:
    {context}

    Question: {q}
    """

    resp = groq_client.getChatResponse(message=prompt)
    print()