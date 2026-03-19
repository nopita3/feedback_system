from langchain_core.messages import HumanMessage
from langgraph.constants import Send

import fitz  # PyMuPDF
import pandas as pd
import base64
import json
from time import perf_counter
from datetime import datetime
from fcntl import flock, LOCK_EX, LOCK_UN
import re

from Schemes.schema import OverallState, PageState
from config import get_ollama_model


# Node: ใช้ PyMuPDF อ่านไฟล์ PDF และแปลงแต่ละหน้าเป็น Base64
def read_and_split_pdf(state: OverallState):
    doc = fitz.open(state["pdf_path"])
    student_test = pd.read_csv(state['student_test_path'])
    pages_list = []
    key_col_list = [ {col: student_test.loc[0, col]} for col in student_test.columns.to_list() if col.startswith("PriKey") and col[-1].isdigit() ][:25]

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        zoom = 1.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        img_bytes = pix.tobytes("png")
        b64_img = base64.b64encode(img_bytes).decode("utf-8")
        pages_list.append(b64_img)
        
    return {"pages": pages_list , "key_answer": key_col_list}

# Conditional Edge (Fan-out): บอก LangGraph ให้แตก Node การทำงานแบบ Parallel ตามจำนวนหน้า
def continue_to_ocr(state: OverallState):
    return [
        Send("process_ocr_page", {"page_b64": page, "page_num": i + 1, "key_list": state["key_answer"]}) 
        for i, page in enumerate(state["pages"])
    ]

# Node: ประมวลผลแต่ละหน้าแบบ Parallel โดยรับ State แบบเดี่ยว (PageState)
def process_ocr_page(state: PageState):
    print("process_ocr_page is running...")
    """
    Process individual PDF pages in parallel using OCR.
    
    This node processes a single page asynchronously and can run concurrently 
    for multiple pages. It receives base64-encoded page data and page number,
    sends them to an LLM model configured to output JSON format only.
    
    Args:
        state (PageState): Contains page_b64 (base64 image) and page_num (page number)
    
    Returns:
        dict: Contains ocr_results with the OCR processing output for the page
    """
    
    page_b64 = state["page_b64"]
    page_num = state["page_num"]
    key_list = state["key_list"]

    # model_name = "qwen3.5:cloud"
    model_name = "qwen3.5:397b-cloud"
    # ระบุ format="json" เพื่อบังคับให้วิเคราะห์ผลออกมาเป็น JSON
    llm , callback  = get_ollama_model(model=model_name)
    strat_ocr_page = perf_counter()
    
    # สร้างโจทย์ (Prompt) เพื่อให้โมเดลทำความเข้าใจโครงสร้างภาพและอ่านไฟล์ข้อสอบ
    prompt_text = (
        "คุณคือผู้เชี่ยวชาญการทำ OCR และวิเคราะห์โจทย์ข้อสอบฟิสิกส์ "
        "โปรดสกัดข้อมูลจากรูปภาพข้อสอบและนำเสนอในรูปแบบ JSON Array เท่านั้น "
        "กรุณาตอบเป็น JSON ในรูปแบบนี้:\n"
        "[\n"
        "  {\n"
        '    "question_id": "เลขข้อ",\n'
        '    "question_content": "เนื้อหาเฉพาะส่วนคำถามของโจทย์ (ไม่เอาตัวเลือก)",\n'
        '    "skill_tags": ["ทักษะที่เกี่ยวข้อง เช่น การวิเคราะห์ระบบ, การคำนวณสูตร"],\n'
        '    "misconcept_type": [{"1": "คำนวณผิดพลาด"  , "2": "แยกประเภทววงจรขนานกับอนุกรมไม่ได้" , "3": "เข้าใจและแก้ปัญหาถูกต้อง" , ...(พิมพ์ให้ครบทุกตัวเลือกและวิเคราะห์ให้ถูกต้อง).... , "N":"ความเข้าใจผิดที่เกิดขึ้น"}],\n'
        '    "image_description": "คำอธิบายรูปภาพอย่างละเอียด (ถ้ามี) เช่น ประจุ a อยู่ตำแหน่ง x=1 หรือถ้าไม่มีรูปให้ใส่เว้นว่าง"\n'
        "  }\n"
        "]\n"
        "ห้ามเกริ่นนำใดๆ ตอบเป็น JSON Array เท่านั้น\n"
        "ถ้าหน้านั้นไม่ใช่ข้อสอบ เช่นระเบียบการสอบ สมการจำเป็น  ให้ข้ามไปเลยไม่ค้องส่งคำตอบของข้อมูลเหล่านั้นมาสิ่งที่ต้องการมีเพียงข้อมูลของข้อสอบ\n "
        "ข้อมูลที่ไม่่มีตัวเลือกก็ให้ข้ามได้เลย ไม่เอาข้อที่แสดงวิธีทำ\n "
        " ถ้าไม่มีไม่ต้องนำเสนอข้อมูลใดๆ มาเลยแม้แต่เครื่องหมายขีด - ไว้\n"
    )
    
    # ส่ง Message แบบระบุ base64 ใน image_url
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "text", "text": f"วิเคราะห์ข้อมูลข้อสอบจากหน้าที่ {page_num} (หน้านี้ไม่ใช่ข้อสอบทุกข้อเป็นเพียส่วนหนึ่งของข้อสอบ) แต่คำตอบขอข้อสอบทุกข้อมีดังนี้: {key_list} "},
            {"type": "image_url", "image_url": f"data:image/png;base64,{page_b64}"}
        ]
    )
    
    try:
        response = llm.invoke([message])
        content_text = response.content if hasattr(response, 'content') else str(response)
        
        # Debug: Check if response is empty
        if not content_text or content_text.strip() == "":
            print(f"WARNING: Empty response from LLM for page {page_num}")
            print(f"Full response object: {response}")
            content_text = "[]"
        
        result_data = {
                        "page_num": page_num,
                        "content": content_text
                    }
        
    except Exception as e:
        print(f"Error processing OCR for page {page_num}: {e}")
        print(f"Exception type: {type(e).__name__}")
        result_data = {
            "page_num": page_num,
            "content": "[]"
        }

    end_ocr_page = perf_counter()
    token_meatadata = {str(datetime.now()): callback.usage_metadata,
                        "processing_time": (end_ocr_page - strat_ocr_page),
                        "agent_work": "extract exam information each pagese",
                        "Platform": "Ollama" }
    try:
        s = json.dumps(token_meatadata, ensure_ascii=False, default=str)
        with open("Token_usage_log.txt", "a", encoding="utf-8") as log_file:
            flock(log_file.fileno(), LOCK_EX)
            log_file.write(s + "\n")
            flock(log_file.fileno(), LOCK_UN)
    except Exception as e:
        print(f"Token log write error: {e}")

    return {"ocr_results": [result_data]}




# Node: Agent รวมคำตอบทั้งหมดให้อยู่ใน List เดียวกัน โดยประมวลผลแต่ละหน้า
def aggregate_results(state: OverallState):
    print("aggregate_results is running...")
    
    with open("output_OCR_OllamaCloud_results.txt", "w", encoding="utf-8") as f:
        for result in state.get("ocr_results", []):
            f.write(f"--- Page {result['page_num']} ---\n{result['content']}\n")

    """
    Aggregate and consolidate OCR results from all processed pages.
    
    This node processes each page individually through an LLM and combines
    all results into a single structured list. It processes page-by-page
    and extends the compiled results with each page's output.
    
    Args:
        state (OverallState): Contains ocr_results list from all processed pages
    
    Returns:
        dict: Contains final_compiled_results as a consolidated list of OCR data
    """
    model_name="scb10x/llama3.2-typhoon2-3b-instruct:latest"
    # model_name = "qwen3.5:cloud"
    llm , callback  = get_ollama_model(model=model_name)
    
    results = sorted(state.get("ocr_results", []), key=lambda x: x["page_num"])
    
    compiled = []
    
    # ประมวลผลแต่ละหน้าทีละหน้า
    for result in results:
        prompt = (
            "คุณคือผู้ช่วยประมวลผลข้อมูล OCR จากหน้า PDF\n"
            "หน้าที่ของคุณ: นำผลลัพธ์ JSON จากหน้าหนึ่ง มาจัดรูปแบบให้เป็น JSON Array ของ objects\n\n"
            "**เงื่อนไขสำคัญ:**\n"
            
            "ต้องตอบกลับเป็น JSON Array เท่านั้น ตามตัวอย่างนี้:\n"
            "```json \n"
            "[\n"
            "  {\n"
            '    "question_id": "1",\n'
            '    "question_content": "เนื้อหาข้อ 1",\n'
            '    "skill_tags": ["ทักษะ 1", "ทักษะ 2"],\n'
            '    "miscon concept_type": [{"1": "คำนวณผิดพลาด"  , "2": "แยกประเภทววงจรขนานกับอนุกรมไม่ได้" , "3": "เข้าใจและแก้ปัญหาถูกต้อง" , ...}],\n'
            '    "image_description": "คำอธิบายรูปภาพ หรือ -"\n'
            "  }\n"
            "]\n\n"
            "ต้องระวังห้ามมี escape character ที่ไม่ได้ escape อย่างถูกต้องใน JSON เช่น backslash ที่ไม่ได้ตามด้วยตัวอักษรที่เป็น valid escape character (\" \ / b f n r t u)\n"
            "ห้ามเพิ่มข้อมูลใดๆ แค่จัดรูปแบบตามโครงสร้างเท่านั้น\n"
            f"ข้อมูลหน้านี้:\n{result['content']}"
        )
        
        try:
            strat_ocr_page = perf_counter()

            response = llm.invoke([HumanMessage(content=prompt)])
            content_text = response.content if hasattr(response, 'content') else str(response)
            
            # Debug: Check if response is empty
            if not content_text or content_text.strip() == "":
                print(f"WARNING: Empty response from LLM in aggregate_results for page {result['page_num']}")
                print(f"Full response object: {response}")
                content_text = "[]"

            # ลบ markdown formatting เผื่อ llm พ่น ```json มา
            if "```json" in content_text:
                content_text = content_text.split("```json")[-1].split("```")[0].strip()
            elif "```" in content_text:
                # เผื่อโมเดลพ่นแค่ ``` ครอบเฉยๆ แยกแค่ ```
                content_text = content_text.split("```")[1].strip()

            # Replace any backslash not followed by a valid JSON escape char: " \\ / b f n r t u
            content_text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', content_text)

            try:
                # ลองซ่อม string ก่อน parse ถ้ารูปแบบคล้าย Object เดี่ยว แต่เราบังคับ Array
                if content_text and not content_text.startswith("["):
                    if content_text.startswith("{") and content_text.endswith("}"):
                        content_text = f"[{content_text}]"
                    elif content_text.strip() == "":
                        content_text = "[]"
                
                # โหลด json แบบ strict=False เพื่อให้ยอมรับ control character ที่ไม่ได้ escape บางตัว
                data = json.loads(content_text, strict=False) if content_text else []
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON for page {result['page_num']}: {e}")
                print(f"Raw output was:\n{repr(content_text)}\n{'='*20}")
                data = []

            # จัดการกรณีต่างๆ และรวมผลลัพธ์เข้ากับ compiled
            if isinstance(data, list):
                compiled.extend(data)
            elif isinstance(data, dict) and "items" in data:
                compiled.extend(data["items"])
            elif isinstance(data, dict):
                compiled.append(data)
            
            end_ocr_page = perf_counter()
            token_meatadata = {str(datetime.now()): callback.usage_metadata,
                                "processing__time": (end_ocr_page - strat_ocr_page),
                                "agent_work": "aggregate information to json schema",
                                "Platform": "Ollama" }
            try:
                s = json.dumps(token_meatadata, ensure_ascii=False, default=str)
                with open("Token_usage_log.txt", "a", encoding="utf-8") as log_file:
                    flock(log_file.fileno(), LOCK_EX)
                    log_file.write(s + "\n")
                    flock(log_file.fileno(), LOCK_UN)
            except Exception as e:
                print(f"Token log write error: {e}")
                
        except Exception as e:
            print(f"Error processing page result: {e}")
            continue
    
        
    return {"final_compiled_results": compiled}



    
