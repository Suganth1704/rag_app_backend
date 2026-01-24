import os
from groq import Groq
from groq.types.chat.chat_completion import ChatCompletion

GROQ_API_KEY=os.getenv("GROQ_API_KEY")


class GroqClient(object):
    def __init__(self):
        self._client = Groq(api_key=GROQ_API_KEY)
        self._model = "llama-3.3-70b-versatile"

    def getChatResponse(self, message:str) -> ChatCompletion:
        message = [{"role":"user", "content":message}]
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=message)
        return  resp

if __name__ == "__main__":
    c = GroqClient()
    message = "What is python?"
    resp = c.getChatReponse(message=message)
    print()