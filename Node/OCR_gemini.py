from langgraph.constants import Send
from langchain_core.messages import HumanMessage

import fitz  # PyMuPDF
import base64
import json 
from datetime import datetime
from time import perf_counter
from fcntl import flock, LOCK_EX, LOCK_UN

from Schemes.schema import OverallState, PageState, OCRResult
from config import get_gemini_model


# Node: ใช้ PyMuPDF อ่านไฟล์ PDF และแปลงแต่ละหน้าเป็น Base64
def read_and_split_pdf(state: OverallState):
    doc = fitz.open(state["pdf_path"])
    pages_list = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        zoom = 1.25
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        img_bytes = pix.tobytes("png")
        b64_img = base64.b64encode(img_bytes).decode("utf-8")
        pages_list.append(b64_img)
        
    return {"pages": pages_list}

# Conditional Edge (Fan-out): บอก LangGraph ให้แตก Node การทำงานแบบ Parallel ตามจำนวนหน้า
def continue_to_ocr(state: OverallState):
    return [
        Send("process_ocr_page", {"page_b64": page, "page_num": i + 1}) 
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

    strat_ocr_page = perf_counter()

    page_b64 = state["page_b64"]
    page_num = state["page_num"]

    model_name = "gemini-3.1-flash-lite-preview"
    llm , callback  = get_gemini_model(model=model_name)
    
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
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": f"data:image/png;base64,{page_b64}"}
        ]
    )
    
    response = llm.invoke([message])
    
    # Extract JSON text from response content
    # Response จาก Gemini อาจเป็น AIMessage ที่มี content เป็น text string โดยตรง
    # หรืออาจเป็น list ของ content blocks
    if isinstance(response.content, str):
        content_text = response.content
    elif isinstance(response.content, list):
        # ถ้า content เป็น list ให้หา text block
        text_blocks = [block.get('text', '') if isinstance(block, dict) 
                       else str(block) for block in response.content]
        content_text = ''.join(text_blocks)
    else:
        content_text = str(response.content)

    end_ocr_page = perf_counter()

    token_meatadata = {str(datetime.now()): callback.usage_metadata,
                        "processing_time": (end_ocr_page - strat_ocr_page),
                        "agent_work": "OCR and Extract information each page",
                        "Platform": "Google" }
    try:
        with open("Token_usage_log.txt", "a", encoding="utf-8") as log_file:
            flock(log_file.fileno(), LOCK_EX)
            log_file.write(json.dumps(token_meatadata, ensure_ascii=False, default=str) + "\n")
            flock(log_file.fileno(), LOCK_UN)
    except Exception as e:
        print(f"Token log write error: {e}")
    
    # Parse JSON string เป็น list ของ dict แล้วแปลงเป็น OCRResult objects
    ocr_results = []
    try:
        ocr_data_list = json.loads(content_text)
        if not isinstance(ocr_data_list, list):
            ocr_data_list = [ocr_data_list]
        
        # แปลง dict เป็น OCRResult objects - skip empty items
        for item in ocr_data_list:
            if item:  # ตรวจสอบว่า item ไม่ว่างเปล่า
                try:
                    ocr_results.append(OCRResult(**item))
                except Exception as item_error:
                    print(f"Skipping invalid OCR item on page {page_num}: {item_error}")
        
        with open("output_OCR_output_debug.txt", "a", encoding="utf-8") as debug_file:
            flock(debug_file.fileno(), LOCK_EX)
            debug_file.write(f"{json.dumps(ocr_data_list, ensure_ascii=False, indent=2)}\n\n")
            flock(debug_file.fileno(), LOCK_UN)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from page {page_num}: {e}")
        print(f"Content was: {content_text[:200]}...")  # Print first 200 chars for debugging
    except Exception as e:
        print(f"Error creating OCRResult objects on page {page_num}: {e}")

    return {"ocr_results": ocr_results}




  
    
    
