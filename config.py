from dotenv import load_dotenv
import os

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama


load_dotenv()
api_key = os.getenv('openai_api_key')
gemini_api_key = os.getenv('gemini_api_key')

def get_gemini_model(model="gemini-3.1-flash-lite-preview", enable_cache=True):
    callback = UsageMetadataCallbackHandler()
    gemini_llm = ChatGoogleGenerativeAI(model=model, 
                                 temperature=0,  
                                 api_key=gemini_api_key, 
                                 callbacks=[callback])
    
    # Store cache flag in model for later use
    gemini_llm.enable_cache = enable_cache

    return gemini_llm, callback 

def get_ollama_model(model="qwen3.5:cloud"):
    callback = UsageMetadataCallbackHandler()
    ollama_llm = ChatOllama(model=model
                            , temperature=0
                            , format="json"
                            , callbacks=[callback])
    
    return ollama_llm, callback


def create_cached_message(content, cache_control=True):
    """
    สร้าง HumanMessage พร้อม cache control สำหรับการป้องกัน context caching
    
    Args:
        content: ข้อความหรือ list of message content
        cache_control: bool - enable cache control หรือไม่
    
    Returns:
        HumanMessage: message พร้อม cache control metadata
    """
    from langchain_core.messages import HumanMessage
    
    if isinstance(content, str):
        # ถ้า content เป็น string ให้แปลงเป็น dict ที่มี cache_control
        if cache_control:
            return HumanMessage(
                content=content,
                response_metadata={"cache_control": {"type": "ephemeral"}}
            )
        return HumanMessage(content=content)
    
    elif isinstance(content, list):
        # ถ้า content เป็น list ของ dict (เช่นมี text + image)
        if cache_control and content:
            # เพิ่ม cache control ลงใน element แรก (text part)
            modified_content = content.copy()
            if isinstance(modified_content[0], dict):
                modified_content[0] = modified_content[0].copy()
                modified_content[0]["cache_control"] = {"type": "ephemeral"}
            return HumanMessage(content=modified_content)
        return HumanMessage(content=content)
    
    return HumanMessage(content=content)
