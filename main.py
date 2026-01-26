from fastapi import FastAPI
from .api.route import router

from .config.logging import setup_logging
setup_logging()

app = FastAPI(title="RAG APP")
app.include_router(router)