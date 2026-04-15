from langgraph.constants import Send
from langchain_core.messages import HumanMessage , SystemMessage

import fitz  # PyMuPDF
import pandas as pd
import base64
import json 
from datetime import datetime
from time import perf_counter
from fcntl import flock, LOCK_EX, LOCK_UN
from Schemes.schema import OverallState, PageState ,OCRExamResponse
from config import get_gemini_model , get_ollama_model

def llm_select(platform_name: str):
    
    if platform_name== "gemini":
        return get_gemini_model()
    elif platform_name == "ollama":
        return get_ollama_model()

    else:
        raise ValueError(f"Unsupported LLM name: {platform_name}")

# Node: ใช้ PyMuPDF อ่านไฟล์ PDF และแปลงแต่ละหน้าเป็น Base64
def read_and_split_pdf(state: OverallState):
    doc = fitz.open(state["pdf_path"])
    df = pd.read_csv(state["student_test_path"]).sample(5, random_state=42)
    labels_file = pd.read_csv(state["labels_path"])
    labels_list = [{str(row.iloc[0]):str(row.iloc[1])}  for _ ,row  in labels_file.iterrows()]

    # Convert numpy types to native Python types using str()
    key_list = [{col: str(df.iloc[0][col])} for col in df.columns.to_list() if col.startswith("PriKey") and col[-1].isdigit()][:25]
    pages_list = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        zoom = 1.25
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        img_bytes = pix.tobytes("png")
        b64_img = base64.b64encode(img_bytes).decode("utf-8")
        pages_list.append(b64_img)
        
    return {"pages": pages_list , "key_answer": key_list , "labels": labels_list}

# Conditional Edge (Fan-out): บอก LangGraph ให้แตก Node การทำงานแบบ Parallel ตามจำนวนหน้า
def continue_to_ocr(state: OverallState):
    return [
        Send("process_ocr_page", {"page_b64": page, 
                                  'progress': [i, len(state["pages"])], 
                                  "key_list": state["key_answer"] , 
                                  "labels": state["labels"] , 
                                  'llm_OCR_platform': state["llm_OCR_platform"]}) 
                                  
                                  for i, page in enumerate(state["pages"])
    ]

# Node: ประมวลผลแต่ละหน้าแบบ Parallel โดยรับ State แบบเดี่ยว (PageState)
def process_ocr_page(state: PageState):
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
    progress = state["progress"]
    key_list = state["key_list"]

    print(f"⏳Processing OCR page📸 {progress[0]+1} of {progress[1]}...")

    
    llm , callback  = llm_select(state["llm_OCR_platform"])
    llm_structured = llm.with_structured_output(OCRExamResponse)
    
    # สร้างโจทย์ (Prompt) เพื่อให้โมเดลทำความเข้าใจโครงสร้างภาพและอ่านไฟล์ข้อสอบ
    prompt_text = (
        "คุณคือผู้เชี่ยวชาญการทำ OCR และวิเคราะห์โจทย์ข้อสอบฟิสิกส์ที่อิงตามหลักสูตรแกนกลางของกระทรวงศึกษาธิการไทย ในส่วนของวิชาฟิสิกส์ (เพิ่มเติม) 4 เรื่องไฟฟ้าสถิตและไฟฟ้ากระแสตรง  "
        "โปรดสกัดข้อมูลจากรูปภาพข้อสอบและ Classify ข้อผิดพลาดของนักเรียนในแต่ละข้อ โดยอิงจาก class label ที่กำหนดให้ด้านล่าง"
        "กรุณาตอบในรูปแบบนี้:\n"
        "[\n"
        "  {\n"
        '    "question_id": "เลขข้อ",\n'
        '    "question_content": "เนื้อหาเฉพาะส่วนคำถามของโจทย์ (ไม่เอาตัวเลือก)",\n'
        '    "image_description": "คำอธิบายรูปภาพอย่างละเอียด (ถ้ามี) เช่น ประจุ a อยู่ตำแหน่ง x=1 หรือถ้าไม่มีรูปให้ใส่เว้นว่าง"\n'
        '   "weekness": "ผลลัพธ์การวิเคราะห์แล้ว classify ข้อผิดพลาดของนักเรียนในแต่ละข้อ"\n'
        '   "class_": "class ที่เป็นไปได้สำหรับการ classify ที่เป็นตัวเลข match กับ weekness เพื่อนำไปนำไปวิเคราะห์ accuracy ของการ classify "\n'
        "  },{...}\n"
        "]\n"
        "ข้อสอบ 1 หน้ามีได้มากกว่า 1 ข้อ ต้องสกัดออกมาให้ครบทุกข้อ และให้ระบุเลขข้อให้ชัดเจนเพื่อใช้ในการเชื่อมโยงกับคำตอบที่ถูกต้องและข้อมูลนักเรียนในภายหลัง"
        "การตอบไม่ต้องเกริ่นนำใด ๆ และให้ classify เพียง 1 class ต่อ 1 ข้อเท่านั้น"
        "พึงระวัง: ต้องตอบกลับมาเป็น JSON Object ที่มี key ชื่อ ocr_results และ value เป็นข้อมูล array เสมอ"
        
        
    )
    sys_prompt = SystemMessage(content =[{'type': 'text', 'text': prompt_text },
                                         {'type': 'text', 'text': f"นี่คือ class label ที่ใช้ในการ classify ข้อผิดพลาดของนักเรียนในแต่ละข้อ: {state['labels']}"}])
    # ส่ง Message แบบระบุ base64 ใน image_url
    message = HumanMessage(
        content=[
            {"type": "text", "text": f"คำตอบที่ถูกต้องของข้อสอบในวิชาฟิสิกส์เข้มข้น 4 (Intensive Physics 4) มีดังนี้: {key_list}\n"},
            {"type": "image_url", "image_url": f"data:image/png;base64,{page_b64}"}
        ]
    )
    
    response = llm_structured.invoke([sys_prompt, message])
    
    items = response.model_dump()

    end_ocr_page = perf_counter()

    token_meatadata = {str(datetime.now()): callback.usage_metadata,
                        "processing_time": (end_ocr_page - strat_ocr_page),
                        "agent_work": "OCR and Extract information each page",
                     }
    try:
        with open("Token_usage_log.txt", "a", encoding="utf-8") as log_file:
            flock(log_file.fileno(), LOCK_EX)
            log_file.write(json.dumps(token_meatadata, ensure_ascii=False, default=str) + "\n")
            flock(log_file.fileno(), LOCK_UN)
    except Exception as e:
        print(f"Token log write error: {e}")
    
    ocr_results = items.get("ocr_results", [])
    

    return {"ocr_results": ocr_results}




  
    
    
