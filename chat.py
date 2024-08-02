import openai
import os
from check_rag import needs_rag, needs_agent
from agent import agent_response
from rag import rag_response

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    print("OPENAI_API_KEY environment variable is not set.")
else:
    openai.api_key  = os.environ['OPENAI_API_KEY']
    print("API key is set.")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


def general_answer(question):
    result = llm.invoke(question)
    print(result)
    return result


while True:
    prompt = input("User: ")
    print()
    print(f"User:\n {prompt}")

    if prompt.lower() == "exit":
        break
    else:
        print(f"\nBot:\n")
        if needs_rag(prompt):
            print("RAG")
            response = rag_response(prompt)
            print(response) # Print a new line for better spacing
        elif needs_agent(prompt):
            print("true")
            response = agent_response(prompt)
            print(response)
        else:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            response = general_answer(prompt)
            print(response) # Print a new line for better spacing
    
    import time
    time.sleep(3)


'''
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

# Define an endpoint for chat
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "")
    print(question)
    async def response_stream():
        response = get_answer(question)
        return response

# Run the server using `uvicorn` or similar ASGI server
'''