from pydantic import BaseModel , Field
from typing import Annotated, List, TypedDict
import operator
from pathlib import Path   

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