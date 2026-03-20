from pydantic import BaseModel , Field
from typing import Annotated, List, TypedDict
import operator
from pathlib import Path

class Student(BaseModel):
    student_id: str = Field( description="รหัสประจำตัวนักเรียน")
    Earned_points: int = Field( description="คะแนนรวมที่ได้รับ")
    chosen_answers: list[dict] = Field( description="คำตอบที่นักเรียนเลือกในแต่ละข้อ เช่น { 'Stu1': '1', ... }")
    point_per_question: list[dict] = Field( description="คะแนนที่นักเรียนได้รับในแต่ละข้อ เช่น { 'Points1': 1, 'Points2': 0, ... }")   

class FeedbackResult(BaseModel):
    student_id: str = Field( description="รหัสประจำตัวนักเรียน")
    total_points: int = Field( description="คะแนนรวมที่ได้รับ")
    percentage: float = Field( description="ร้อยละของคะแนนที่ได้รับ")
    feedback_details: str = Field( description='รายละเอียด feedback ที่ต้องระบุใน feedback ดังนี้ {"concept/skill/ความเข้าใจ ที่นักเรียนทำได้ดีแล้วในแต่ละข้อ": "...", "จุดconcept/skill/ความเข้าใจ ที่ยังทำไม่ได้": "...", "แนวทางในการพัฒนาในจุดที่ยังทำไม่ได้": "...", "concern จาก error_types": "..." }')

class OCRResult(BaseModel):
    question_id: str = Field( description="เลขข้อ")
    question_content: str = Field( description="เนื้อหาโจทย์ (ไม่เอาตัวเลือก)")
    skill_tags: List[str] = Field( description="ทักษะที่เกี่ยวข้อง เช่น การวิเคราะห์ระบบ, การคำนวณสูตร")
    misconcept_type: List[dict] = Field( description='การจัดเก็บข้อมูลการ missconcept จากการตอบตัวลวง เช่น [{"1": "คำนวณผิดพลาด"  , "2": "แยกประเภทววงจรขนานกับอนุกรมไม่ได้" , "3": "เข้าใจและแก้ปัญหาถูกต้อง" , ...}]')
    image_description: str = Field( description="คำอธิบายรูปภาพอย่างละเอียด (ถ้ามี) เช่น ประจุ a อยู่ตำแหน่ง x=1 หรือถ้าไม่มีรูปให้ใส่เว้นว่าง")

# State ย่อยสำหรับการทำ Map-Reduce (ส่งข้อมูลหน้าเดี่ยวไปในแต่ละ Node)
class PageState(TypedDict):
    page_b64: str
    page_num: int
    key_list : List[dict]


# กำหนดรูปแบบ State ของระบบ (อัปเดตกลับมาเป็น Parallel)
class OverallState(TypedDict):
    pdf_path: bytes
    # pdf_path: Path
    pages: List[str]
    student_test_path: bytes
    # student_test_path: Path
    key_answer: List[dict] = Field(description="คำตอบที่ถูกต้องของข้อสอบในรูปแบบ List ของ Dict เช่น [ { 'question_id': '1', 'correct_answer': 'คำตอบที่ถูกต้องของข้อ 1' }, ... ]")
    student_information: list[Student] = Field(default_factory=list, description="ข้อมูลนักเรียนที่มีรหัสประจำตัวและคะแนนในแต่ละข้อสอบของนักเรียนแต่ละคน")

    ocr_results: Annotated[List[OCRResult], operator.add] = Field(default_factory=list) # ใช้ operator.add เพื่อรวบรวม Results จาก Parallel Nodes

    feedback : Annotated[list[FeedbackResult], operator.add] = Field(default_factory=list)
    

    










    