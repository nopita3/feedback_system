from datetime import datetime
from io import BytesIO
from time import perf_counter
import pandas as pd
from fcntl import flock, LOCK_EX, LOCK_UN
from langchain_core.messages import HumanMessage , SystemMessage
from langgraph.constants import Send
import json
from google.genai.errors import ServerError , ClientError
from ollama._types import ResponseError 
from openai import RateLimitError , APITimeoutError , BadRequestError
from config import get_gemini_model , get_ollama_model , get_typhoon_model
from Schemes.schema import OverallState,  FeedbackResult, Student

def llm_select(platform_name: str):
    
    if platform_name== "gemini":
        return get_gemini_model()
    elif platform_name == "ollama":
        return get_ollama_model()
    elif platform_name == "typhoon":
        return get_typhoon_model()
    else:
        raise ValueError(f"Unsupported LLM name: {platform_name}")

def extract_student_information(state: OverallState):
    
    
    df = pd.read_csv(state["student_test_path"])
    point_col_list = [ col for col in df.columns.to_list() if col.startswith("Points") and col[-1].isdigit() ][:25]
    answer_col_list = [ col for col in df.columns.to_list() if col.startswith("Stu") and col[-1].isdigit() ][:25]
    

    student_info = []
    for _, row in df.iterrows():
        student_info.append(Student(
            student_id=str(row["ID"]),
            Earned_points=row["Earned Points"],
            chosen_answers=[{col: row[col]} for col in answer_col_list],
            point_per_question=[{col: row[col]} for col in point_col_list]
        ))
    
    return {"student_information": student_info}

def continue_to_feedback(state: OverallState):
    
    # ใช้ ocr_user_corrections ถ้ายูสเซอร์มีการแก้ไขให้เอามาใช้แทน ถ้าไม่มีให้ใช้ ocr_results ต้นฉบับ
    ocr_data = state.get("ocr_results") 
    
    return [Send("process_feedback", {"student_information": student, 
                                      "ocr_results": ocr_data , 
                                      'feed_progress': [i, len(state["student_information"]),] , 
                                      'llm_feedback_platform': state["llm_feedback_platform"]}) 
                                      for i, student in enumerate(state["student_information"])]
        


    
def process_feedback(state: OverallState):

    import time

    

    student = state['student_information']
    llm, callback = llm_select(state["llm_feedback_platform"])
    percentage = student.Earned_points / len(student.point_per_question) * 100

    progress = state['feed_progress']
    print(f"⏳Processing feedback✍🏻 {progress[0]+1} of {progress[1]}...")
    if (progress[0]+1) % 3 == 0:  # Every 2nd student
        time.sleep(40)  # Simulate processing time
    
    system_message = SystemMessage(content=f"""
        ระบบปัญญาประดิษฐ์ถูกใช้เพื่อประมวลผลทางภาษาเพื่อวิเคราะห์คะแนนที่นักเรียนได้รับจากการทำข้อสอบในแต่ละข้อ 
        ที่อิงตามหลักสูตรแกนกลางของกระทรวงศึกษาธิการไทยในส่วนของวิชาฟิสิกส์ (เพิ่มเติม) (ไม่มีการเรียนแคลคูลัสในระดับชั้นนี้ให้แนะนำด้วยวิธีที่ไม่ต้องใช้แคลคูลัส)
        เพื่อให้ feedback ให้กับนักเรียนในการปรับปรุงการเรียนรู้และปิดช่องว่างการเรียนรู้ที่เกิดขึ้น
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
        
        ถ้ามีสมการในเนื้อหาให้เขียนอยู่ในรูปแบบ LaTeX และใช้ MathJax ในการแสดงผลสมการนั้น ๆ ในส่วนของ feedback
        การพิจารณาข้อมูลข้อสอบให้พิจารณาทุกด้านของข้อสอบ โดยข้อมูลของข้อสอบโดยมี Exam_objecttive ที่ระบุวัตถุประสงค์ของแต่ละข้อ โดยมีดังนี้: {state["ocr_results"] }
    """)


    human_message = HumanMessage(content=f"""นักเรียนที่มีรหัส {student.student_id} ได้รับคะแนนในแต่ละข้อดังนี้ {student.point_per_question}
        กรุณาวิเคราะห์คะแนนที่นักเรียนได้รับในแต่ละข้อ และให้คำแนะนำในการปรับปรุงการทำข้อสอบในอนาคต""")
    
    try:
        start_feedback = perf_counter()
        response = llm.invoke([system_message, human_message])
        end_feedback = perf_counter()


        if state["llm_feedback_platform"] == "gemini":
            feedback_details = response.content[0]['text'].strip()
        else:
            feedback_details = response.content.strip() 

        
        feedback_info = FeedbackResult(
            student_id=student.student_id,
            total_points=student.Earned_points,
            percentage=percentage,
            
            feedback_details=feedback_details
        )

        token_metadata = {
            str(datetime.now()): callback.usage_metadata,
            "processing_time": (end_feedback - start_feedback),
            "agent_work": "generate feedback for student",
            'platform': state["llm_feedback_platform"]
        }

        with open(f"Token_usage_log.txt", "a", encoding="utf-8") as f:
            flock(f.fileno(), LOCK_EX)
            f.write(json.dumps(token_metadata, ensure_ascii=False, default=str) + "\n")
            flock(f.fileno(), LOCK_UN)
        return {"feedback": [feedback_info]}
    
    except ServerError as e:
        print(f"Error occurred while generating feedback for student {student.student_id}: {e}")
        time.sleep(120)
        feedback_info = FeedbackResult(
            student_id=student.student_id,
            total_points=student.Earned_points,
            percentage=percentage,
            
            feedback_details="Null"
        )
        return {"feedback": [feedback_info]}
    
    except BadRequestError as e:
        print(f"Bad request error occurred while generating feedback for student {student.student_id}: {e}")
        time.sleep(120)
        feedback_info = FeedbackResult(
            student_id=student.student_id,
            total_points=student.Earned_points,
            percentage=percentage,
            
            feedback_details="Null"
        )
        return {"feedback": [feedback_info]}
    
    except ResponseError as e:
        if e.status_code >= 500:
            print(f"Server error occurred while generating feedback for student {student.student_id}: {e}")
            time.sleep(120)
            feedback_info = FeedbackResult(
                student_id=student.student_id,
                total_points=student.Earned_points,
                percentage=percentage,
                
                feedback_details="Null"
            )
            return {"feedback": [feedback_info]}
        raise ValueError(f"Error occurred while generating feedback for student {student.student_id}: {e}")

    except ClientError as e:
        raise ValueError(f"Error occurred while generating feedback for student {student.student_id}: {e}")

    except RateLimitError as e:
        raise ValueError(f"Rate limit error occurred while generating feedback for student {student.student_id}: {e}")
    except APITimeoutError as e:
        raise ValueError(f"API timeout error occurred while generating feedback for student {student.student_id}: {e}")
    


