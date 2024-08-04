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

load_dotenv(find_dotenv())

api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    print("OPENAI_API_KEY environment variable is not set.")
else:
    openai.api_key  = os.environ['OPENAI_API_KEY']


backend_data = []
package_data = []

def read_json_objects_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)
    
@tool
def get_flight_info(loc_origin: str, loc_destination: str) -> dict:
    """Get flight information between two locations. Required loc_origin airport code, loc_destination airport code"""

    missing_fields = []
    
    # Check each field and add to the list if missing
    if not loc_origin:
        missing_fields.append("loc_origin")
    if not loc_destination:
        missing_fields.append("loc_destination")
    
    if missing_fields:
        return f"Enter the correct details for flight info, missing {', '.join(missing_fields)}"
    
    flight_jsons = {"DPS":"flights.json","MLE":"maldives_flights.json","COK":"mysore_flights.json"}

    file_path = f"./{flight_jsons[loc_destination]}"
    print(file_path)

    flights_data = read_json_objects_from_file(file_path)

    flight_data = list()
    i = 1
    for data in flights_data[:4]:
        flight_details = dict()
        flight_details['Origin'] = data['Origin']
        flight_details['Destination'] = data['Destination']
        flight_details['TotalFare'] = data['Fare']['TotalFare']
        flight_details['AgentPreferredCurrency'] = data['Fare']['AgentPreferredCurrency']

        for s in data['Segments']:
            flight_details['AirlineName'] = s[0]['AirlineName']
            flight_details['FlightNumber'] = s[0]['FlightNumber']
            flight_details['DepartureTime'] = s[0]['DepartureTime']
            flight_details['ArrivalTime'] = s[0]['ArrivalTime']
            flight_details["Availability"] = s[0]['NoOfSeatAvailable']
        
        key = f"flight_{i}"
        flight_data.append(flight_details)
        i+=1
    
    backend_data.append(flight_data)

    return json.dumps(flight_data)

@tool
def packages_list(places) -> dict:
    """Provide packages we offer from the places text given."""
    file_path = f"./available_packages.json"
    packages_data = read_json_objects_from_file(file_path)
    package_data.append(packages_data)
    return places

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

    print(booking_info)
    
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
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm_with_tool = llm.bind(functions = [format_tool_to_openai_function(t) for t in tools])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        "chat_history": memory.load_memory_variables, 
    } | prompt | llm_with_tool | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools)

def agent_response(query):

    response = agent_executor.invoke({"input":query})
    print(backend_data)
    
    if backend_data != []:
        return (response, backend_data[-1])
    else:
        return (response['output'], None)