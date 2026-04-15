from dotenv import load_dotenv
import os

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


load_dotenv()

gemini_api_key = os.getenv('gemini_api_key')
typhoon_api_key = os.getenv('typhoon_api_key')


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
                            temperature=0,
                            num_ctx=6000,
                            callbacks=[callback])
    
    return ollama_llm, callback

def get_typhoon_model(model="typhoon-v2.5-30b-a3b-instruct"):
    callback = UsageMetadataCallbackHandler()
    openai_llm = ChatOpenAI(model=model, 
                            temperature=0,  
                            max_tokens=12288,
                            api_key=typhoon_api_key,
                            base_url="https://api.opentyphoon.ai/v1", 
                            callbacks=[callback])
    
    return openai_llm, callback


