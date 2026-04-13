from pydantic import BaseModel , Field , RootModel
from typing import Annotated, TypedDict
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
    image_description: str = Field( description="คำอธิบายรูปภาพอย่างละเอียด (ถ้ามี) เช่น ประจุ a อยู่ตำแหน่ง x=1 หรือถ้าไม่มีรูปให้ใส่เว้นว่าง")
    weekness: str = Field( description='ผลลัพธ์การวิเคราะห์แล้ว classify ข้อผิดพลาดของนักเรียนในแต่ละข้อ')
    class_:int = Field( description="class ที่เป็นไปได้สำหรับการ classify ที่เป็นตัวเลข match กับ weekness เพื่อนำไปนำไปวิเคราะห์ accuracy ของการ classify ')")

class OCRExamResponse(BaseModel):
    ocr_results: list[OCRResult] = Field( description="ผลลัพธ์การทำ OCR และวิเคราะห์โจทย์ข้อสอบฟิสิกส์ที่อิงตามหลักสูตรแกนกลางของกระทรวงศึกษาธิการไทย ในส่วนของวิชาฟิสิกส์ (เพิ่มเติม) 4 เรื่องไฟฟ้าสถิตและไฟฟ้ากระแสตรง โดยมีรูปแบบเป็น list ของ OCRResult")

class Curriculm(RootModel):
    root: list[dict[str, list[str]]] = Field(default_factory=list, description="ผลการเรียนรู้ และสาระการเรียนรู้ที่เกี่ยวข้องของแต่ละข้อสอบ เช่น [{ 'Assessment1': ['สาระการเรียนรู้ที่เกี่ยวข้องของข้อสอบข้อ 1', ...] }, ...]")
    

# State ย่อยสำหรับการทำ Map-Reduce (ส่งข้อมูลหน้าเดี่ยวไปในแต่ละ Node)
class PageState(TypedDict):
    page_b64: str
    page_num: int
    key_list : list[dict]


# กำหนดรูปแบบ State ของระบบ (อัปเดตกลับมาเป็น Parallel)
class OverallState(TypedDict):
    
    pdf_path: Path
    pages: list[str]
    
    student_test_path: Path
    key_answer: list[dict] = Field(description="คำตอบที่ถูกต้องของข้อสอบในรูปแบบ list ของ Dict เช่น [ { 'question_id': '1', 'correct_answer': 'คำตอบที่ถูกต้องของข้อ 1' }, ... ]")
    student_information: list[Student] = Field(default_factory=list, description="ข้อมูลนักเรียนที่มีรหัสประจำตัวและคะแนนในแต่ละข้อสอบของนักเรียนแต่ละคน")

    ocr_results: Annotated[OCRExamResponse, operator.add] = Field(default_factory=list) # ใช้ operator.add เพื่อรวบรวม Results จาก Parallel Nodes
    ocr_user_corrections: list[OCRResult] = Field(default_factory=list, description="ข้อมูลแก้ไขจากผู้ใช้ในกรณีที่ OCR ผิดพลาด โดยมีรูปแบบเดียวกับ ocr_results")

    feedback : Annotated[list[FeedbackResult], operator.add] = Field(default_factory=list)
    labels : list[dict] = Field(default_factory=list, description="class ที่เป็นไปได้สำหรับการ classify ข้อผิดพลาดของนักเรียนในแต่ละข้อ")

class CurriculumState(TypedDict):
    curriculum_path: str
    pages: list[str]
    curriculum_analysis: list[Curriculm]= Field(default_factory=list)