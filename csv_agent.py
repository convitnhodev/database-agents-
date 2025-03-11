from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
import os
import warnings

os.environ.clear()  
current_dir = Path(__file__).parent.absolute()
env_path = current_dir / '.env'
print(f"Looking for .env file at: {env_path}")
print(f"Does .env file exist? {env_path.exists()}")

load_dotenv(dotenv_path=env_path, override=True)

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Suppress urllib3 warnings (optional)
warnings.filterwarnings('ignore', category=Warning)

try:
    llm_name = "gpt-4-0125-preview"
    model = ChatOpenAI(
        api_key=openai_key, 
        model_name=llm_name,
        temperature=0.7
    )

    # Load and preprocess the CSV data
    csv_path = 'data/salaries_2023.csv'
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    
    df = pd.read_csv(csv_path).fillna(value=0)
    agent = create_pandas_dataframe_agent(model, df, verbose=True)
    
    query = "What is the average salary of the employees?"
    res = agent.invoke({"input": query})
    print(f"Query: {query}")
    print(f"Response: {res}")

except FileNotFoundError as e:
    print(f"File error: {str(e)}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    raise  # Re-raise the exception for debugging purposes

