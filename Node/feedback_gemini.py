from datetime import datetime
from time import perf_counter
import pandas as pd
from fcntl import flock, LOCK_EX, LOCK_UN
from langchain_core.messages import HumanMessage , SystemMessage
from langgraph.constants import Send
import json
from io import BytesIO
from config import get_gemini_model
from Schemes.schema import OverallState,  FeedbackResult, Student

def extract_student_information(state: OverallState):
    df = pd.read_csv(BytesIO(state["student_test_path"])).sample(5, random_state=42) #อย่าลืมเอา sample เวลาไปรวม node จริง ๆด้วยนะ
    
    # df = pd.read_csv(state["student_test_path"]).sample(5, random_state=42) #อย่าลืมเอา sample เวลาไปรวม node จริง ๆด้วยนะ
    point_col_list = [ col for col in df.columns.to_list() if col.startswith("Points") and col[-1].isdigit() ][:25]
    answer_col_list = [ col for col in df.columns.to_list() if col.startswith("Stu") and col[-1].isdigit() ][:25]
    

    student_info = []
    for _, row in df.iterrows():
        student_info.append(Student(
            student_id=str(row["StudentID"]),
            Earned_points=row["Earned Points"],
            chosen_answers=[{col: row[col]} for col in answer_col_list],
            point_per_question=[{col: row[col]} for col in point_col_list]
        ))
    
    return {"student_information": student_info}

def continue_to_feedback(state: OverallState):
    
    return [Send("process_feedback", {"student_information": student, "ocr_results": state["ocr_results"]  }) 
        for _, student in enumerate(state["student_information"])]


    
def process_feedback(state: OverallState):
    student = state['student_information']
    llm, callback = get_gemini_model(model="gemini-3.1-flash-lite-preview")
    percentage = student.Earned_points / len(student.point_per_question) * 100
    
    system_message = SystemMessage(content=f"""
        ระบบปัญญาประดิษฐ์ถูกใช้เพื่อประมวลผลทางภาษาเพื่อวิเคราะห์คะแนนที่นักเรียนได้รับจากการทำข้อสอบในแต่ละข้อ 
        ร่วมกับข้อมูลข้อสอบที่นักเรียนใช้สอบในวิชาฟิสิกส์เข้มข้น 4 (Intensive Physics 4) ที่อิงตามหลักสูตรแกนกลางของกระทรวงศึกษาธิการไทย
        ในส่วนของวิชาฟิสิกส์ (เพิ่มเติม) 4 เรื่องไฟฟ้าสถิตและไฟฟ้ากระแสตรง เพื่อให้ feedback ให้กับนักเรียนในการปรับปรุงการเรียนรู้และปิดช่องว่างการเรียนรู้ที่เกิดขึ้น
        แนวการตอบของระบบ
        1. ให้ระบุว่ามาจากการวิเคราะห์ด้วยปัญญาประดิษฐ์เสมอ
        2. ระบุข้อมูลประจำตัวนักเรียนดังนี้
        \t2.1 รหัสประจำตัวนักเรียน: {student.student_id}
        \t2.2 คะแนนรวมที่ได้รับ: {student.Earned_points} คะแนน จากคะแนนเต็ม {len(student.point_per_question)} คะแนน
        \t2.3 ร้อยละของคะแนนที่ได้รับ: {percentage:.2f}%
        3. รายระเอียดที่ต้องระบุใน feedback ดังนี้
        \t3.1 concept/skill/ความเข้าใจ ที่นักเรียนทำได้ดีแล้วในแต่ละข้อ (เช่น นักเรียนสามารถคำนวณข้อที่เกี่ยวกับพื้นฐานทางไฟฟ้าได้ดีแล้ว)
        \t3.2 จุดconcept/skill/ความเข้าใจ ที่ยังทำไม่ได้ โดยต้องระบุทุกจุด(เช่น ในจุดที่นักเรียนต้องพัฒนาต่อไป 1. ... 2. ... 3.ยังมีปัญหาในโจทย์ที่ซับซ้อนเรื่อง ... , 4. ... )
        \t3.3 แนวทางในการพัฒนาในจุดที่ยังทำไม่ได้ โดยต้องระบุทุกจุดที่ยังทำไม่ได้ (เช่น แนวทางในการพัฒนาตัวเอง ได้แก่ 1. ... 2. ... 3. ...)
        \t3.4 การระบุ concept/skill/ความเข้าใจ ไม่ต้องบอกเลขข้อว่าทำข้อไหนได้หรือไม่ได้ แต่ให้บอกเป็น concept/skill/ความเข้าใจที่นักเรียนทำได้หรือยังทำไม่ได้ในแต่ละข้อสอบแทน
        \t3.5 ต้องระบุทุก concept ห้ามตกหล่นรวมถึงข้อ concern จาก error_types
        5. ตอบด้วยน้ำเสียงที่เข้าถึงง่ายเหมาะกับเด็กนักเรียนไทยในช่วงมัธยมปลาย 
        6. ใช้ภาษาไทยเท่านั้นในการสื่อสารออกไป
        7. ถ้าน้องทำคะแนนรวมได้สูงมากให้ชื่นชม แต่ถ้าไม่สูงต้องให้กำลังใจการพัฒนาต่อไปอย่างเป็นธรรมชาติที่มนุษย์คุยกันทั่วไป

        การพิจารณาข้อมูลข้อสอบให้พิจารณาทุกด้านของข้อสอบ โดยข้อมูลของข้อสอบมีดังนี้: {state["ocr_results"]}
    """)
    
    human_message = HumanMessage(content=f"""นักเรียนที่มีรหัส {student.student_id} ได้รับคะแนนในแต่ละข้อดังนี้ {student.point_per_question}
        กรุณาวิเคราะห์คะแนนที่นักเรียนได้รับในแต่ละข้อ และให้คำแนะนำในการปรับปรุงการทำข้อสอบในอนาคต""")
    
    start_feedback = perf_counter()
    response = llm.invoke([system_message, human_message], temperature=0.2)
    end_feedback = perf_counter()
    
    feedback_info = FeedbackResult(
        student_id=student.student_id,
        total_points=student.Earned_points,
        percentage=percentage,
        feedback_details=response.content[0]['text'].strip()
    )

    with open(f"output_feedback_Gemini.txt", "a", encoding="utf-8") as f:
        flock(f.fileno(), LOCK_EX)
        f.write(feedback_info.model_dump_json(indent=2) + "\n")
        flock(f.fileno(), LOCK_UN)
    
    token_metadata = {
        str(datetime.now()): callback.usage_metadata,
        "processing_time": (end_feedback - start_feedback),
        "agent_work": "generate feedback for student",
        "Platform": "Google"
    }
    
    with open(f"Token_usage_log.txt", "a", encoding="utf-8") as f:
        flock(f.fileno(), LOCK_EX)
        f.write(json.dumps(token_metadata, ensure_ascii=False, default=str) + "\n")
        flock(f.fileno(), LOCK_UN)

    return {"feedback": [feedback_info]}


# def review_feedback_internal(feedback: FeedbackResult) -> tuple:
#     """Internal review function, returns (is_acceptable, quality_score, issues, suggestions)"""
#     llm, callback = get_gemini_model(model="gemini-3.1-flash-lite-preview")
    
#     review_system = SystemMessage(content="""
#         คุณคือผู้เชี่ยวชาญในการรีวิว feedback ที่สร้างโดยปัญญาประดิษฐ์สำหรับนักเรียนมัธยมปลายในวิชาฟิสิกส์เข้มข้น 4 (Intensive Physics 4) เรื่องไฟฟ้าสถิตและไฟฟ้ากระแสตรง
#         กรุณารีวิว feedback นี้โดยพิจารณาจากความถูกต้องของข้อมูลที่ให้ไป ความชัดเจนในการสื่อสาร ความครอบคลุมของ feedback ในการวิเคราะห์จุดแข็งและจุดที่ต้องพัฒนา และความเหมาะสมของคำแนะนำในการพัฒนาต่อไป
#         โดยมีการการพิจารณาดังนี้
#         •	ด้านความถูกต้องทางด้านวิชาการ (Academic Correctness): การเลือกใช้คำศัพท์เฉพาะทางวิทยาศาสตร์ได้อย่างถูกต้อง แม่นยำ และเหมาะสมกับระดับความรู้ของผู้เรียน 
#         •	ด้านรายละเอียดและความถูกต้องของข้อมูล (Detail and Accuracy of Information): การแสดงรายละเอียดครบถ้วน และมีความถูกต้องของข้อมูลที่กำหนดไว้ในวัตถุประสงค์การประเมินผลนอกจากนี้
#         •	ด้านความเหมาะสมของการใช้ภาษาในการให้คำแนะนำ (Appropriateness of Language):  ปัญญหาประดิษฐ์ควรใช้ภาษาที่สร้างสรรค์และให้กำลังใจ เพื่อรักษาสภาพแวดล้อมการเรียนรู้เชิงบวก โดยไม่ชมเชยจนเกินจริง 
#         •	ด้านความชัดเจนและการสื่อความหมาย (Clarity and Communication): ยึดตามมิติ Linguistic Clarity ของ Seßler et al. (2025) ที่ประเมินความชัดเจนของโครงสร้างประโยค เพื่อให้ผู้เรียนระดับเป้าหมายสามารถทำความเข้าใจได้ง่าย ไม่ซับซ้อน ซึ่งตรงกับเกณฑ์ Clarity of Feedback ในแบบประเมินผู้เชี่ยวชาญของ Chung (2025)
#         เพื่อให้การรีวิวมีความเป็นระบบและสามารถประเมินได้อย่างมีประสิทธิภาพ 
#         การรีวิวเนื้อหา feedback และตอบกลับเป็น JSON ในรูปแบบนี้:
#         {
#           "is_acceptable": true/false,
#           "quality_score": 1-10,
#           "issues": ["ภาษายังดูเหมือนหุ่นยนต์ ไม่มีความธนรรมชาติของมนุษย์", "ใช้คำที่เป็นทางการมากเกินไป ไม่เหมาะกับนักเรียนมัธยมปลาย", ...],
#           "suggestions": ["ปรับใช้ภาษาที่เข้าใจง่ายขึ้น", "เพิ่มคำแนะนำที่เป็นรูปธรรมมากขึ้น เช่น แนะนำแหล่งเรียนรู้เพิ่มเติม หรือวิธีการฝึกฝนที่ชัดเจน", ...]
#         }
#         ต้องสามารถนำไปพัฒนาให้เหมาะสมและดียิ่งขึ้นได้ต่อไปได้อย่างชัดเจน                          
#     """)
    
#     review_message = HumanMessage(content=f"""
#         กรุณารีวิว feedback นี้:
#         {feedback.feedback_details}
#     """)
#     start_review = perf_counter()
#     response = llm.invoke([review_system, review_message], temperature=0.3)
#     end_review = perf_counter()
#     token_meatadata = {
#             str(datetime.now()): callback.usage_metadata,
#             "processing_time": (end_review - start_review),
#             "agent_work": "review feedback for student",
#             "Platform": "Gemini"
#         }
#     with open(f"Token_usage_log.txt", "a", encoding="utf-8") as f:
#             flock(f, LOCK_EX)
#             f.write(json.dumps(token_meatadata, ensure_ascii=False, default=str) + "\n")
#             flock(f, LOCK_UN)
#     try:
#         import json as json_module
#         if isinstance(response.content, str):
#             content_text = response.content
#         elif isinstance(response.content, list):
#             # Extract text blocks from list format
#             text_blocks = [block.get('text', '') if isinstance(block, dict) else str(block) 
#                            for block in response.content]
#             content_text = ''.join(text_blocks)
#         else:
#             content_text = str(response.content)
#         review_data = json_module.loads(content_text)
#         return (
#             review_data.get('is_acceptable', False),
#             review_data.get('quality_score', 5),
#             review_data.get('issues', []),
#             review_data.get('suggestions', [])
#         )
#     except:
#         return (False, 5, ["Parse error"], ["Regenerate feedback"])
