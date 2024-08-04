from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from thirdPartyApi import getFlightDetails
from chat import get_reponse
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",  
    "https://your-ngrok-url.ngrok-free.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    content: str

@app.post("/chat")
def read_root(req : Message):
    res = get_reponse(req.content)
    return res