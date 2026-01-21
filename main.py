from ingestion.ingest import start_ingest
from vector_store.chroma_client import RagAppChromaClient


if __name__ == "__main__":
    print('start')
    #start_ingest()
    obj = RagAppChromaClient()
    q = 'Basic Commands in Command Prompt'
    #r = obj.mmr_search(query=q)
    r = obj.qurey_chroma(query=q)
    print()