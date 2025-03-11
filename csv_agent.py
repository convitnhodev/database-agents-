from langchain_core.messages import HumanMessage, SystemMessage
import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pathlib import Path
import pandas as pd

openai_key = os.getenv("OPENAI_API_KEY")

llm_name = "gpt-3.5-turbo"
model = ChatOpenAI(
    api_key=openai_key, 
    model_name=llm_name,
    temperature=0.7
)


