from langchain.agents import initialize_agent, load_tools
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from datetime import datetime, timedelta
from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions, format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field
import json
import os
import openai
from dotenv import load_dotenv, find_dotenv
from thirdPartyApi import getFlightDetails

load_dotenv(find_dotenv())

api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    print("OPENAI_API_KEY environment variable is not set.")
else:
    openai.api_key  = os.environ['OPENAI_API_KEY']
    print("API key is set.")

@tool
def get_flight_info(loc_origin: str, loc_destination: str) -> dict:
    """Get flight information between two locations."""

    missing_fields = []
    
    # Check each field and add to the list if missing
    if not loc_origin:
        missing_fields.append("loc_origin")
    if not loc_destination:
        missing_fields.append("loc_destination")
    
    if missing_fields:
        return f"Enter the correct details for flight info, missing {', '.join(missing_fields)}"
    
    flight_info = {
        "loc_origin": loc_origin,
        "loc_destination": loc_destination,
        "datetime": str(datetime.now() + timedelta(hours=2)),
        "airline": "KLM",
        "flight": "KL643",
    }

    res = getFlightDetails(flight_info)

    return json.dumps(flight_info)

@tool
def book_flight(loc_origin, loc_destination=None, passenger_name=None):
    """Simulate booking a flight between two locations. Required Parameters to book a flight is departure location, destination location, passenger_name"""
    # List of missing fields
    missing_fields = []
    
    # Check each field and add to the list if missing
    if not loc_origin:
        missing_fields.append("loc_origin")
    if not loc_destination:
        missing_fields.append("loc_destination")
    if not passenger_name:
        missing_fields.append("passenger_name")
    
    # If any fields are missing, return an error message
    if missing_fields:
        return f"Enter the correct details to book the flight, missing {', '.join(missing_fields)}"
    
    booking_reference = "BOOK123456"
    booking_datetime = datetime.now() + timedelta(hours=1)
    
    booking_info = {
        "booking_reference": booking_reference,
        "loc_origin": loc_origin,
        "loc_destination": loc_destination,
        "passenger_name": passenger_name,
        "datetime": booking_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "airline": "KLM",
        "flight": "KL643",
        "status": "Confirmed",
        "message": "Your flight has been booked successfully."
    }
    
    return json.dumps(booking_info)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

tools = [get_flight_info, book_flight]

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

# Define the prompt using SystemMessage and UserMessage
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful flight booking assistant who can book and provide details about flight."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

llm_with_tool = llm.bind(functions = [format_tool_to_openai_function(t) for t in tools])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        "memory": memory.load_memory_variables, 
    } | prompt | llm_with_tool | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools)

def agent_response(query):
    print("Agent query")
    response = agent_executor.invoke({"input":query})
    return response['output']