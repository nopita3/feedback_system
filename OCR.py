import operator
import base64
import fitz  # PyMuPDF
from typing import Annotated, List, TypedDict
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from pathlib import Path    
from time import perf_counter
from pydantic import BaseModel , Field
import json

# กำหนดรูปแบบ State ของระบบ (อัปเดตกลับมาเป็น Parallel)
class OverallState(TypedDict):
    pdf_path: Path
    pages: List[str]
    ocr_results: Annotated[List[str], operator.add] # ใช้ operator.add เพื่อรวบรวม Results จาก Parallel Nodes
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
    
    # ระบุ format="json" เพื่อบังคับให้วิเคราะห์ผลออกมาเป็น JSON
    llm = ChatOllama(model="qwen3.5:cloud", temperature=0, format="json")
    
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
        "ถ้าหน้านั้นไม่ใช่ข้อสอบแบบตัวเลือก ให้ข้ามไปเลยไม่ค้องส่งคำตอบของข้อมูลเหล่านั้นมาสิ่งที่ต้องการมีเพียงข้อมูลของข้อสอบแบบเลือกตอบเท่านั้น\n "
    )
    
    # ส่ง Message แบบระบุ base64 ใน image_url
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": f"data:image/png;base64,{page_b64}"}
        ]
    )
    
    response = llm.invoke([message])
    
    formatted_result = f"--- Page {page_num} ---\n, {response.content}"
    
    return {"ocr_results": [formatted_result]}

# Node: Agent รวมคำตอบทั้งหมดให้อยู่ใน List เดียวกัน โดยใช้ Structured Output
def aggregate_results(state: OverallState):
    """
    Aggregate and consolidate OCR results from all processed pages.
    
    This node collects all OCR outputs from parallel page processing and uses
    an LLM to combine them into a single structured JSON format. It handles
    sorting of results to ensure consistent ordering despite parallel execution.
    
    Args:
        state (OverallState): Contains ocr_results list from all processed pages
    
    Returns:
        dict: Contains final_compiled_results as a consolidated list of OCR data
    """
    
    # ใช้ qwen3.5:cloud ที่ support JSON format ได้ดีกว่า
    # สำหรับ structured output กับ List type อาจมีปัญหา ดังนั้นจึงใช้ manual JSON parsing แทน
    llm = ChatOllama(model="scb10x/llama3.2-typhoon2-3b-instruct", temperature=0, format="json")
    
    # นำผลลัพธ์จากแต่ละหน้ามารวมกันเป็น Text เดียว
    # เรียงลำดับ string ใหม่ตาม page เพราะการทำงานแบบ Parallel ผลลัพธ์อาจจะดีดกลับมาเรียงไม่ถูกลำดับ
    results = sorted(state.get("ocr_results", [])) 
    combined_text = "\n".join(results)
    
    prompt = (
        "คุณคือผู้ช่วยรวบรวมข้อมูล OCR จากทุกหน้า PDF\n"
        "หน้าที่ของคุณ: นำผลลัพธ์ JSON ของแต่ละหน้า มารวบรวมและจัดรูปแบบให้เป็น Single JSON Object\n\n"
        "**เงื่อนไขสำคัญ:**\n"
        "1. ต้องตอบกลับเป็น JSON Object โครงสร้างเดียวเท่านั้น ตามตัวอย่างนี้:\n\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "question_id": "1",\n'
        '      "question_content": "เนื้อหาข้อ 1",\n'
        '      "skill_tags": ["ทักษะ 1", "ทักษะ 2"],\n'
        '      "error_type": "ข้อผิดพลาดที่พบบ่อย",\n'
        '      "image_description": "คำอธิบายรูปภาพ หรือ -"\n'
        "    },\n"
        "    {\n"
        '      "question_id": "2",\n'
        '      "question_content": "เนื้อหาข้อ 2",\n'
        '      "skill_tags": ["ทักษะ 3"],\n'
        '      "error_type": "ข้อผิดพลาด",\n'
        '      "image_description": "-"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "2. รวมข้อมูลทั้งหมดให้ครบถ้วน ห้ามตัดทอนข้อใดทิ้ง\n"
        "3. ห้ามเปลี่ยนแปลงข้อมูลแม้แต่ตัวอักษรเดียว\n"
        "4. เรียงลำดับตามหมายเลขข้อสอบจากน้อยไปมาก\n\n"
        f"ข้อมูลดิบทั้งหมดจากทุกหน้า:\n{combined_text}"
    )
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        # ดึง String ที่เป็น JSON มาแปลง (Parse) เอง จะปลอดภัยกับโมเดลมากกว่า
        data = json.loads(response.content)

        
        
        # จัดการกรณีที่เผื่อโมเดลแอบดื้อ คืนเป็น List ตรงๆ หรือคืนที่มี dict ครอบ
        if isinstance(data, list):
            compiled = data
        elif isinstance(data, dict) and "items" in data:
            compiled = data["items"]
        else:
            compiled = [data]
            
    except Exception as e:
        print(f"Error parsing final results: {e}")
        compiled = []
        
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
    
    # เริ่มต้นการทำงาน (Invoke) แบบ Parallel
    strat = perf_counter()
    final_state = graph.invoke(
        {"pdf_path": pdf_file_path, "ocr_results": [], "final_compiled_results": []},
        config={"max_concurrency": 6} # เพิ่ม max_concurrency กำจัดจำนวน request พร้อมกันเพื่อแก้ปัญหา 429
    )
    # บันทึก raw OCR results ลงไฟล์ (แต่ละหน้าต่อหนึ่งบรรทัด)
    with open("OCR_results.txt", "w", encoding="utf-8") as f:
        for result in final_state.get("ocr_results", []):
            f.write(result + "\n")

    with open("Aggregate_results.json", "w", encoding="utf-8") as f:
        # บันทึกข้อมูลที่ผ่านการรวมแล้วจาก Agent ลงไฟล์ JSON 
        json.dump(final_state.get("final_compiled_results", []), f, ensure_ascii=False, indent=2)
        
    end = perf_counter()
    print(f"Total processing time: {end - strat:.2f} seconds")
    
