import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from app.key import GOOGLE_API_KEY

load_dotenv()

api_key = GOOGLE_API_KEY
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    api_key=GOOGLE_API_KEY
)

result = llm.invoke("Hello Gemini, can you confirm you are working?")
print(result)