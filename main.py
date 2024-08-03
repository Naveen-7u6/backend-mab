from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from thirdPartyApi import getFlightDetails
from chat import get_reponse
import json

app = FastAPI()

class Message(BaseModel):
    content: str

@app.post("/chat")
def read_root(req : Message):
    res = get_reponse(req.content)
    return res