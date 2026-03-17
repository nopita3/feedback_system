from langgraph.constants import Send
from langchain_core.messages import HumanMessage

import fitz  # PyMuPDF
import base64
import json 
from datetime import datetime
from time import perf_counter
from fcntl import flock, LOCK_EX, LOCK_UN

from Schemes.schema import OverallState, PageState, OCRResultList
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
    
    llm , callback  = get_gemini_model(model="gemini-3.1-flash-lite-preview")
    
    # สร้างโจทย์ (Prompt) เพื่อให้โมเดลทำความเข้าใจโครงสร้างภาพและอ่านไฟล์ข้อสอบ
    prompt_text = (
        "คุณคือผู้เชี่ยวชาญการทำ OCR และวิเคราะห์โจทย์ข้อสอบฟิสิกส์ "
        "โปรดสกัดข้อมูลจากรูปภาพข้อสอบและนำเสนอในรูปแบบ JSON Array เท่านั้น "
        "เงื่อนไขสำคัญ: **ไม่ต้องพิมพ์ตัวเลือก (ก, ข, ค, ง) ให้พิมพ์เฉพาะท่อนที่เป็นคำถามหรือโจทย์เดี่ยวๆ**\n\n"
        "กรุณาตอบเป็น JSON ในรูปแบบนี้:\n"
        "[\n"
        "  {\n"
        '    "question_id": "เลขข้อ",\n'
        '    "question_content": "เนื้อหาโจทย์ (ไม่เอาตัวเลือก)",\n'
        '    "skill_tags": ["ทักษะที่เกี่ยวข้อง เช่น การวิเคราะห์ระบบ, การคำนวณสูตร"],\n'
        '    "error_type": "จุดผิดพลาดที่ผู้เรียนมักจะทำผิด หรือเว้นขีด - ไว้",\n'
        '    "image_description": "คำอธิบายรูปภาพอย่างละเอียด (ถ้ามี) เช่น ประจุ a อยู่ตำแหน่ง x=1 หรือถ้าไม่มีรูปให้ใส่เว้นว่าง"\n'
        "  }\n"
        "]\n"
        "ห้ามเกริ่นนำใดๆ ตอบเป็น JSON Array เท่านั้น\n"
        "ถ้าหน้านั้นไม่ใช่ข้อสอบ ให้ข้ามไปเลยไม่ต้องส่งคำตอบของข้อมูลเหล่านั้นมาสิ่งที่ต้องการมีเพียงข้อมูลของข้อสอบ\n "
    )
    
    # ส่ง Message แบบระบุ base64 ใน image_url พร้อม cache control
    content = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": f"data:image/png;base64,{page_b64}"}
    ]    
    message = HumanMessage(content=content)
    
    response = llm.invoke([message])
    
    # เมื่อใช้ Gemini แบบปกติ ค่าที่ได้จะเป็น AIMessage ที่มี content แต่ตอนนี้มันมี metadata ปนมาด้วย
    # เราจึงสกัดเฉพาะส่วนที่เป็น .content (ซึ่งเป็น text JSON) ออกมาก่อน 
    content_text = response.content if hasattr(response, 'content') else str(response)

    result_data = {
        "page_num": page_num,
        "content": content_text
    }
    end_ocr_page = perf_counter()

    token_meatadata = {str(datetime.now()): callback.usage_metadata,
                        "processing_ocr_each_page_time": (end_ocr_page - strat_ocr_page)}
    try:
        with open("Token_GeminiAPI_usage_log.txt", "a", encoding="utf-8") as log_file:
            flock(log_file.fileno(), LOCK_EX)
            log_file.write(json.dumps(token_meatadata, ensure_ascii=False, default=str) + "\n")
            flock(log_file.fileno(), LOCK_UN)
    except Exception as e:
        print(f"Token log write error: {e}")
    
    
    

    return {"ocr_results": [result_data]}

# Node: Agent รวมคำตอบทั้งหมดให้อยู่ใน List เดียวกัน โดยประมวลผลแต่ละหน้า
def aggregate_results(state: OverallState):

    with open("output_OCR_Gemini_results.txt", "w", encoding="utf-8") as f:
        # ดึงมา sort ให้สวยงามก่อนเขียนลงไฟล์
        sorted_results = sorted(state.get("ocr_results", []), key=lambda x: x["page_num"])
        for result in sorted_results:
            page = result["page_num"]
            content = result["content"]
            f.write(f"--- Page {page} ---\n{content}\n")

    print("aggregate_results is running...")
    
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
    
    # เปลี่ยนกลับเป็นโมเดลที่เสถียรกับการทำ Structured Output
    llm , callback  = get_gemini_model(model="gemini-3.1-flash-lite-preview")
    structured_model = llm.with_structured_output(schema=OCRResultList, method="json_schema")

    results = sorted(state.get("ocr_results", []), key=lambda x: x["page_num"])
    
    compiled = []
    
    # ประมวลผลแต่ละหน้าทีละหน้า
    for result in results:
        prompt = (
            "คุณคือผู้ช่วยประมวลผลข้อมูล OCR จากหน้า PDF\n"
            "หน้าที่ของคุณ: นำผลลัพธ์ JSON จากหน้าหนึ่ง มาจัดรูปแบบให้เป็น JSON Array ของ objects\n\n"
            "**เงื่อนไขสำคัญ:**\n"
            "ส่วนไหนเป็นสมการให้ใส่ format LaTeX ได้เลย เช่น $E=mc^2$ \n"
            "ต้องตอบกลับเป็น JSON Array เท่านั้น ตามตัวอย่างนี้:\n"
            "[\n"
            "  {\n"
            '    "question_id": "1",\n'
            '    "question_content": "เนื้อหาข้อ 1",\n'
            '    "skill_tags": ["ทักษะ 1", "ทักษะ 2"],\n'
            '    "error_type": "ข้อผิดพลาดที่พบบ่อย",\n'
            '    "image_description": "คำอธิบายรูปภาพ หรือ -"\n'
            "  }\n"
            "]\n\n"
            "ห้ามเพิ่มข้อมูลใดๆ แค่จัดรูปแบบตามโครงสร้างเท่านั้น\n"
            f"ข้อมูลหน้านี้:\n{result['content']}"
        )
        
        try:
            strat_ocr_page = perf_counter()
            
            # สร้าง message ปกติ
            
            message = HumanMessage(content=prompt)
            response = structured_model.invoke([message])
            
            try:
                with open("Token_GeminiAPI_usage_log.txt", "a", encoding="utf-8") as log_file:
                    flock(log_file.fileno(), LOCK_EX)
                    log_file.write(json.dumps(token_meatadata, ensure_ascii=False, default=str) + "\n")
                    flock(log_file.fileno(), LOCK_UN)
            except Exception as e:
                print(f"Token log write error: {e}")
            
            # เนื่องจากใช้ schema=OCRResultList เข้าไปตรงๆ response จะตีกลับมาเป็น Pydantic object
            # เราสามารถเรียกใช้ .items หรือแปลงเป็น dict ได้เลย
            if hasattr(response, 'items'):
                # แปลง Pydantic ข้อมูลย่อยให้อยู่ในรูป dict เพื่อเอาไป save JSON ได้
                data = [item.model_dump() for item in response.items]
            else:
                data = []

            # รวมผลลัพธ์เข้ากับ compiled
            compiled.extend(data)

            end_ocr_page = perf_counter()
            token_meatadata = {str(datetime.now()): callback.usage_metadata,
                        "processing_aggregate_time_seconds": (end_ocr_page - strat_ocr_page)}
                
        except Exception as e:
            print(f"Error processing page result: {e}") 
            continue

    

    return {"final_compiled_results": compiled}


  
    
    
