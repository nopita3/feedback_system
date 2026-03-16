from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

from langchain_core.callbacks import UsageMetadataCallbackHandler

from pydantic import BaseModel , Field
from typing import Annotated, List, TypedDict
import fitz  # PyMuPDF

import os
import operator
import base64
import json
from pathlib import Path    
from time import perf_counter

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('openai_api_key')
gemini_api_key = os.getenv('gemini_api_key')



callback_ollama2 = UsageMetadataCallbackHandler()

# กำหนดรูปแบบ State ของระบบ (อัปเดตกลับมาเป็น Parallel)
class OverallState(TypedDict):
    pdf_path: Path
    pages: List[str]
    ocr_results: Annotated[List[dict], operator.add] # ใช้ operator.add เพื่อรวบรวม Results จาก Parallel Nodes
    final_compiled_results: List[dict] # เพิ่ม State สำหรับเก็บผลลัพธ์ที่รวมแล้ว

# State ย่อยสำหรับการทำ Map-Reduce (ส่งข้อมูลหน้าเดี่ยวไปในแต่ละ Node)
class PageState(TypedDict):
    page_b64: str
    page_num: int

class OCRResult(BaseModel):
    question_id: str = Field( description="เลขข้อ")
    question_content: str = Field( description="เนื้อหาโจทย์ (ไม่เอาตัวเลือก)")
    skill_tags: List[str] = Field( description="ทักษะที่เกี่ยวข้อง เช่น การวิเคราะห์ระบบ, การคำนวณสูตร")
    error_type: str = Field( description="จุดผิดพลาดที่ผู้เรียนมักจะทำผิด หรือเว้นขีด - ไว้")
    image_description: str = Field( description="คำอธิบายรูปภาพอย่างละเอียด (ถ้ามี) เช่น ประจุ a อยู่ตำแหน่ง x=1 หรือถ้าไม่มีรูปให้ใส่เว้นว่าง")

# เพิ่ม Schema แบบ List สำหรับ Agent ที่ทำหน้าที่รวมคำตอบ
class OCRResultList(BaseModel):
    items: List[OCRResult] = Field(description="รายการรวมโจทย์ข้อสอบทั้งหมดจากทุกหน้า")


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
    callback_ollama1 = UsageMetadataCallbackHandler()
    page_b64 = state["page_b64"]
    page_num = state["page_num"]
    
    # ระบุ format="json" เพื่อบังคับให้วิเคราะห์ผลออกมาเป็น JSON
    llm = ChatOllama(model="qwen3.5:cloud", temperature=0, format="json", callbacks=[callback_ollama1])
    # llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0,  api_key=gemini_api_key, callbacks=[callback_gemini1])
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
        "ถ้าหน้านั้นไม่ใช่ข้อสอบ ให้ข้ามไปเลยไม่ค้องส่งคำตอบของข้อมูลเหล่านั้นมาสิ่งที่ต้องการมีเพียงข้อมูลของข้อสอบ\n "
    )
    
    # ส่ง Message แบบระบุ base64 ใน image_url
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": f"data:image/png;base64,{page_b64}"}
        ]
    )
    
    try:
        response = llm.invoke([message])
        content_text = response.content if hasattr(response, 'content') else str(response)
        
        result_data = {
                        "page_num": page_num,
                        "content": content_text
                    }
        
    except Exception as e:
        print(f"Error processing OCR for page {page_num}: {e}")
        result_data = {
            "page_num": page_num,
            "content": "[]"
        }
        
    print(f"Total Ollama 1 (OCR) Tokens Used: {callback_ollama1.usage_metadata}\n")
    return {"ocr_results": [result_data]}




# Node: Agent รวมคำตอบทั้งหมดให้อยู่ใน List เดียวกัน โดยประมวลผลแต่ละหน้า
def aggregate_results(state: OverallState):
    print("aggregate_results is running...")
    
    with open("OCR_results.txt", "w", encoding="utf-8") as f:
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
    llm = ChatOllama(model="qwen3.5:cloud", temperature=0, format="json", callbacks=[callback_ollama2])
    
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
            response = llm.invoke([HumanMessage(content=prompt)])
            content_text = response.content if hasattr(response, 'content') else str(response)
            
            # ลบ markdown formatting เผื่อ llm พ่น ```json มา
            if "```json" in content_text:
                content_text = content_text.split("```json")[-1].split("```")[0].strip()
            elif "```" in content_text:
                # เผื่อโมเดลพ่นแค่ ``` ครอบเฉยๆ แยกแค่ ```
                content_text = content_text.split("```")[1].strip()

            # แก้ปัญหา Invalid \escape จาก LaTeX format ที่พ่น \ มาโดยไม่ escape เช่น \circ หรือ \n
            # แทนที่จะเจอ \c แล้วพัง เราต้อง escape \ ให้เป็น \\ เพื่อให้ json.loads มองเป็น literal backslash ใน json string
            # แต่ข้อความพวก \n หรือ \t เรายังต้องการเก็บความหมายของ newline/tab ไว้อยู่
            
            # ชดเชยกรณี \ ธรรมดาที่ไม่ได้ตามด้วยตัวอักษรพิเศษของ json
            # วิธีที่ง่ายที่สุดสำหรับเคส LaTeX ภายใน JSON string คือการหาและแทนที่ backslash เดี่ยวๆ 
            # ที่ไม่ใช่เริ่ม \n, \t, \r, \", \\ ด้วย double backslash
            # ใช้ raw string replace สำหรับคำยอดฮิตใน LaTeX
            content_text = content_text.replace(r"\circ", r"\\circ")
            content_text = content_text.replace(r"\n", r"\\n") # ใน string แบบ JSON อยากให้เป็นแค่ text ก็ต้อง escape
            content_text = content_text.replace(r"\\n", r"\\n") # เผื่อมันมาเป็น \\n อยู่แล้ว

            try:
                # ลองซ่อม string ก่อน parse ถ้ารูปแบบคล้าย Object เดี่ยว แต่เราบังคับ Array
                if content_text.startswith("{") and content_text.endswith("}"):
                    content_text = f"[{content_text}]"
                
                # โหลด json แบบ strict=False เพื่อให้ยอมรับ control character ที่ไม่ได้ escape บางตัว
                data = json.loads(content_text, strict=False)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON for page {result['page_num']}: {e}")
                print(f"Raw output was:\n{content_text}\n{'='*20}")
                data = []

            # จัดการกรณีต่างๆ และรวมผลลัพธ์เข้ากับ compiled
            if isinstance(data, list):
                compiled.extend(data)
            elif isinstance(data, dict) and "items" in data:
                compiled.extend(data["items"])
            elif isinstance(data, dict):
                compiled.append(data)
                
        except Exception as e:
            print(f"Error processing page result: {e}")
            continue
    print(f"Total Ollama 2 Tokens Used: {callback_ollama2.usage_metadata}\n")
        
    return {"final_compiled_results": compiled}

# ประกอบ Graph นำ Components ทั้งหมดมาร้อยเรียงกัน
builder = StateGraph(OverallState)

# เพิ่ม Nodes
builder.add_node("read_and_split_pdf", read_and_split_pdf)
builder.add_node("process_ocr_page", process_ocr_page)
builder.add_node("aggregate_results", aggregate_results) # เพิ่ม Node 

# เพิ่ม Edges
builder.add_edge(START, "read_and_split_pdf")

# หลังจากอ่านไฟล์เสร็จ ใช้ conditional edges จัดการทำ Send Fan-Out
builder.add_conditional_edges("read_and_split_pdf", continue_to_ocr)
# พอมันรัน process_ocr_page เสร็จแบบคู่ขนานครบทุกตัว ให้ไหลมารวมที่ aggregate_results
builder.add_edge("process_ocr_page", "aggregate_results")

builder.add_edge("aggregate_results", END)

# Compile LangGraph
graph = builder.compile()

if __name__ == "__main__":
    # ระบุพาทไปยังไฟล์ PDF ของคุณ
    pdf_file_path = Path("Documents/final_M5_022568.pdf")
    
    # เริ่มต้นการทำงาน (Invoke) แบบ Parallel แต่จำกัด request ป้องกัน Rate limit API
    strat = perf_counter()
    final_state = graph.invoke(
        {"pdf_path": pdf_file_path, "ocr_results": [], "final_compiled_results": []},
        config={"max_concurrency": 2} # ลดเหลือ 2 เพื่อป้องกัน 429 RESOURCE_EXHAUSTED จาก Gemini Free Tier
    )
    # บันทึก raw OCR results ลงไฟล์ (แต่ละหน้าต่อหนึ่งบรรทัด)
    

    with open("Aggregate_results.json", "w", encoding="utf-8") as f:
        # บันทึกข้อมูลที่ผ่านการรวมแล้วจาก Agent ลงไฟล์ JSON 
        json.dump(final_state.get("final_compiled_results", []), f, ensure_ascii=False, indent=2)
        
    end = perf_counter()
    print(f"Total processing time: {end - strat:.2f} seconds")
    
    print(f"Total Ollama 2 (Aggregate) Tokens Used: {callback_ollama2.usage_metadata}\n")
    
