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

from Schemes.schema import OverallState, PageState, OCRResult
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
        dict: Contains ocr_results with OCRResult objects from the page
    """
    
    page_b64 = state["page_b64"]
    page_num = state["page_num"]
    key_list = state.get("key_list", [])

    model_name = "qwen3.5:397b-cloud"
    llm, callback = get_ollama_model(model=model_name)
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
        '    "misconcept_type": [{"1": "คำนวณผิดพลาด", "2": "แยกประเภทวงจรขนานกับอนุกรมไม่ได้", "3": "เข้าใจและแก้ปัญหาถูกต้อง", ...(พิมพ์ให้ครบทุกตัวเลือกและวิเคราะห์ให้ถูกต้อง)...}],\n'
        '    "image_description": "คำอธิบายรูปภาพอย่างละเอียด (ถ้ามี) เช่น ประจุ a อยู่ตำแหน่ง x=1 หรือถ้าไม่มีรูปให้ใส่เว้นว่าง"\n'
        "  }\n"
        "]\n"
        "ห้ามเกริ่นนำใดๆ ตอบเป็น JSON Array เท่านั้น\n"
        "ถ้าหน้านั้นไม่ใช่ข้อสอบ เช่นระเบียบการสอบ สมการจำเป็น ให้ข้ามไปเลยไม่ต้องส่งคำตอบของข้อมูลเหล่านั้นมา สิ่งที่ต้องการมีเพียงข้อมูลของข้อสอบ\n"
        "ข้อมูลที่ไม่มีตัวเลือกก็ให้ข้ามได้เลย ไม่เอาข้อที่แสดงวิธีทำ\n"
        "ถ้าไม่มี misconcept_type ไม่ต้องพิมพ์เครื่องหมายขีด ให้เว้นว่างเป็น [] ไว้\n"
    )
    
    # ส่ง Message แบบระบุ base64 ใน image_url
    message_content = [
        {"type": "text", "text": prompt_text},
    ]
    
    if key_list:
        message_content.append(
            {"type": "text", "text": f"วิเคราะห์ข้อมูลข้อสอบจากหน้าที่ {page_num} (หน้านี้ไม่ใช่ข้อสอบทุกข้อเป็นเพียส่วนหนึ่งของข้อสอบ) แต่คำตอบขอข้อสอบทุกข้อมีดังนี้: {key_list} "}
        )
    
    message_content.append(
        {"type": "image_url", "image_url": f"data:image/png;base64,{page_b64}"}
    )
    
    message = HumanMessage(content=message_content)
    
    try:
        response = llm.invoke([message])
        
        # Handle response content (similar to Gemini)
        if isinstance(response.content, str):
            content_text = response.content
        elif isinstance(response.content, list):
            text_blocks = [block.get('text', '') if isinstance(block, dict) 
                           else str(block) for block in response.content]
            content_text = ''.join(text_blocks)
        else:
            content_text = str(response.content)
        
    except Exception as e:
        print(f"Error processing OCR for page {page_num}: {e}")
        content_text = "[]"

    end_ocr_page = perf_counter()
    
    token_metadata = {
        str(datetime.now()): callback.usage_metadata,
        "processing_time": (end_ocr_page - strat_ocr_page),
        "agent_work": "OCR and Extract information each page",
        "Platform": "Ollama"
    }
    try:
        with open("Token_usage_log.txt", "a", encoding="utf-8") as log_file:
            flock(log_file.fileno(), LOCK_EX)
            log_file.write(json.dumps(token_metadata, ensure_ascii=False, default=str) + "\n")
            flock(log_file.fileno(), LOCK_UN)
    except Exception as e:
        print(f"Token log write error: {e}")
    
    # Parse JSON string เป็น list ของ dict แล้วแปลงเป็น OCRResult objects
    ocr_results = []
    try:
        # Clean up markdown formatting if present
        if "```json" in content_text:
            content_text = content_text.split("```json")[-1].split("```")[0].strip()
        elif "```" in content_text:
            content_text = content_text.split("```")[1].strip() if "```" in content_text.split("```")[0] == "" else content_text
        
        # Replace any backslash not followed by a valid JSON escape char
        content_text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', content_text)
        
        ocr_data_list = json.loads(content_text, strict=False) if content_text else []
        if not isinstance(ocr_data_list, list):
            ocr_data_list = [ocr_data_list]
        
        # แปลง dict เป็น OCRResult objects - skip empty items
        for item in ocr_data_list:
            if item:  # ตรวจสอบว่า item ไม่ว่างเปล่า
                try:
                    ocr_results.append(OCRResult(**item))
                except Exception as item_error:
                    print(f"Skipping invalid OCR item on page {page_num}: {item_error}")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from page {page_num}: {e}")
        print(f"Content was: {content_text[:200]}...")
    except Exception as e:
        print(f"Error creating OCRResult objects on page {page_num}: {e}")

    return {"ocr_results": ocr_results}








    
