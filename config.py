from dotenv import load_dotenv
import os

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


load_dotenv()

gemini_api_key = os.getenv('gemini_api_key')
typhoon_api_key = os.getenv('typhoon_api_key')
qwen_api_key = os.getenv('DASHSCOPE_API_KEY')


def get_gemini_model(model="gemini-flash-lite-latest"):
    callback = UsageMetadataCallbackHandler()
    gemini_llm = ChatGoogleGenerativeAI(model=model, 
                                 temperature=0,  
                                 api_key=gemini_api_key, 
                                 callbacks=[callback],
                                 max_output_tokens=2700)    
    
    return gemini_llm, callback 

def get_ollama_model(model="gemma4:e4b"):
    callback = UsageMetadataCallbackHandler()
    ollama_llm = ChatOllama(model=model,
                            temperature=0,
                            num_ctx=6000,
                            callbacks=[callback],
                            reasoning=False,
                            num_gpu=45
                            )
    
    return ollama_llm, callback

def get_typhoon_model(model="qwen3.5-flash"):
    callback = UsageMetadataCallbackHandler()
    openai_llm = ChatOpenAI(model=model, 
                            temperature=0,
                            reasoning_effort="medium",
                            max_tokens=12288,
                            api_key=qwen_api_key,
                            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", 
                            callbacks=[callback])
    
    return openai_llm, callback


