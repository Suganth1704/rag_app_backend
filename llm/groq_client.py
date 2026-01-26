import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import logging
logger = logging.getLogger(__name__)


GROQ_API_KEY=os.getenv("GROQ_API_KEY")
#self._model = "llama-3.3-70b-versatile"

class GroqClient(object):
    def __init__(self,):
        self._llm=ChatGroq(
                            model="llama-3.3-70b-versatile",
                            api_key=GROQ_API_KEY
                            )
    def get_context_prompt_template(self):
        logger.info("Getting context prompt template ...")
        prompt=ChatPromptTemplate.from_messages([
            ("system", "Answer using the provided context only."),
            MessagesPlaceholder("chat_history"),
            ("system", "Context:\n{context}"),
            ("human", "{question}")
        ])
        return prompt
    
    def get_prompt_template(self):
        logger.info("Getting prompt template ...")
        prompt=ChatPromptTemplate.from_messages([
            ("system", "Answer for provided question?"),
            ("human", "{question}")
        ])
        return prompt

if __name__ == "__main__":
    obj = GroqClient()
    # pt = obj.get_context_prompt_template()
    t=obj.get_prompt_template()
    resp = obj._llm.invoke(
        t.format_messages(question="What is python?")
    )
    print()