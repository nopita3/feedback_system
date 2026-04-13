from dotenv import load_dotenv
import os

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama


load_dotenv()
api_key = os.getenv('openai_api_key')
gemini_api_key = os.getenv('gemini_api_key')

def get_gemini_model(model="gemini-3.1-flash-lite-preview"):
    callback = UsageMetadataCallbackHandler()
    gemini_llm = ChatGoogleGenerativeAI(model=model, 
                                 temperature=0,  
                                 api_key=gemini_api_key, 
                                 callbacks=[callback])
    
    
    return gemini_llm, callback 

def get_ollama_model(model="gemma4:31b-cloud"):
    callback = UsageMetadataCallbackHandler()
    ollama_llm = ChatOllama(model=model,
                            temperature=0.1,
                            callbacks=[callback])
    
    return ollama_llm, callback



