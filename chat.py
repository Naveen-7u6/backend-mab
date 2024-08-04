import openai
import os
from check_rag import is_common_question
from agent import agent_response
from rag import rag_response

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

load_dotenv(find_dotenv())

api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    print("OPENAI_API_KEY environment variable is not set.")
else:
    openai.api_key  = os.environ['OPENAI_API_KEY']
    print("API key is set.")



llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

def general_answer(question):
    result = llm.invoke(question)
    return result.content

def get_reponse(prompt):
    import json
    # if needs_agent(prompt) or needs_rag(prompt):
    if is_common_question(prompt):
        response = general_answer(prompt)
        return {"response":response}
    else:
        response, questions, backend_data = agent_response(prompt)
        if backend_data is None:
            return {"response":response, "questions": questions}
        else:
            return {"response": backend_data}
    # else:
    #     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    #     response = general_answer(prompt)
    #     return {"response":response}
