import openai
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

api_key = os.getenv('OPENAI_API_KEY')
print(api_key)

if api_key is None:
    print("OPENAI_API_KEY environment variable is not set.")
else:
    openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain_community.document_loaders import TextLoader


loader = TextLoader(r"D:\backend\backend-mab\docs\maldives.txt")
document = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    chunk_size = 2500,
    chunk_overlap = 150,
)

split_doc = text_splitter.split_documents(document)

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embedding = OpenAIEmbeddings()
persist_directory = r'D:\backend\backend-mab\docs\chroma'


vectordb = Chroma.from_documents(
    documents=split_doc,
    embedding=embedding,
    persist_directory=persist_directory
)




from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


# Build prompt
from langchain.prompts import PromptTemplate

template = """
Use the following pieces of context to answer the question at the end. Refer to the provided context to answer the question.
If the details about the place are not in the context, respond with "There are no details about this place. Check the package details." 
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)


# Run chain
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})

qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
)

def rag_response(question):
    result = qa.invoke({"question": question})
    return result["answer"]