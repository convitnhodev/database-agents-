from langchain_core.messages import HumanMessage, SystemMessage
import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

try:
    llm_name = "gpt-3.5-turbo"
    model = ChatOpenAI(
        api_key=openai_key, 
        model_name=llm_name,
        temperature=0.7
    )

    messages = [ 
        SystemMessage(content="You are a helpful assistant that can answer questions and help with tasks."),
        HumanMessage(content="What is a bit, and tell me your name")
    ]

    response = model.invoke(messages)
    print(response)
except Exception as e:
    print(f"An error occurred: {str(e)}")