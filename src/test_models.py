import os
import dotenv
dotenv.load_dotenv("../.env")
from langchain_google_genai import ChatGoogleGenerativeAI

models = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-3.1-flash-lite", 
    "gemini-3.5-flash-lite", 
    "gemini-flash-latest", 
    "gemini-flash-lite-latest", 
    "gemini-3.7-flash"
]

with open("models_result2.txt", "w", encoding="utf-8") as f:
    for m in models:
        try:
            llm = ChatGoogleGenerativeAI(model=m)
            res = llm.invoke("Hello, answer in 1 word.")
            out = f"SUCCESS: {m} -> {res.content.strip()}"
            print(out)
            f.write(out + "\n")
            f.flush()
        except Exception as e:
            out = f"FAILED: {m} -> {e}"
            print(out)
            f.write(out + "\n")
            f.flush()
