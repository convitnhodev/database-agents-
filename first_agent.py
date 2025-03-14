from langchain_core.messages import HumanMessage, SystemMessage
import os 
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from pathlib import Path


os.environ.clear()  
current_dir = Path(__file__).parent.absolute()
env_path = current_dir / '.env'
print(f"Looking for .env file at: {env_path}")
print(f"Does .env file exist? {env_path.exists()}")

load_dotenv(dotenv_path=env_path, override=True)

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

try:
    llm_name = "gpt-4-0125-preview"
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
    print("\nAssistant's response:")
    print(response.content)  
except Exception as e:
    print(f"An error occurred: {str(e)}")