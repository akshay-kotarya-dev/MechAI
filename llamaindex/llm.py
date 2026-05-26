from llama_index.llms.groq import Groq
from llama_index.core import Settings
import os
import dotenv

dotenv.load_dotenv()


def connect_llm():
    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )
    Settings.llm = llm
    return llm
