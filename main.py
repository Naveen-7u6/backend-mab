from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from thirdPartyApi import getFlightDetails
from chat import get_reponse

app = FastAPI()


class Message(BaseModel):
    content: str

@app.post("/chatbot/")
def chat_response(message: Message):
    response_content = generate_response(message.content)
    response = {
        "sender": "chatbot",
        "content": response_content
    }
    return response

def generate_response(user_message: str) -> str:
    return ""

@app.post("/")
def read_root(req : Message):
    #res = get_reponse(req.content)
    print(req.content)
    return {"message": "HI"}