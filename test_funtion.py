from pathlib import Path
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage , SystemMessage
import json
from config import get_gemini_model

data = pd.read_csv(Path("Documents/Intensive_Physics_4.csv"))

answer_col_list = [ col for col in data.columns.to_list() if col.startswith("Stu") and col[-1].isdigit() ][:25]
point_col_list = [ col for col in data.columns.to_list() if col.startswith("Points") and col[-1].isdigit() ][:25]


data_complete = data[["StudentID", 'Earned Points'] + answer_col_list + point_col_list]
data_complete["StudentID"] = data_complete["StudentID"].astype(str)

sample = data_complete.sample(3, random_state=42)
test_json = json.load(open("output_Aggregate_Gemini_results.json", "r"))





for i, row in sample.iterrows():
    print(f"{"="*20} Analyzing StudentID: {i} {'='*20}")
    row = row.to_dict()
    row = {k: row[k] for k in ["StudentID", "Earned Points"] + point_col_list}
    # llm = ChatOllama(model="qwen3.5:cloud", temperature=0, format="json")
    from time import perf_counter
    start_ocr_page = perf_counter()
    llm, callback  = get_gemini_model(model="gemini-3.1-flash-lite-preview")
    system_message = SystemMessage(content=f"""
                                                ระบบปัญาประดิษฐ์ถูกใช้เพื่อประมวลผลทางภาษาเพื่อวิเคราะห์คะแนนที่นักเรียนได้รับจากการทำข้อสอบในแต่ละข้อ 
                                                ร่วมกับข้อมูลข้อสอบที่นักเรียนใช้สอบในวิชาฟิสิกส์เข้มข้น 4 (Intensive Physics 4) ที่อิงตามหลักสูตรแกนกลางของกระทรวงศึกษาธิการไทย
                                                ในส่วนของวิชาฟิสิกส์ (เพิ่มเติม) 4 เรื่องไฟฟ้าสถิตและไฟฟ้ากระแสตรง เพื่อให้ feedback ให้กับนักเรียนในการปรับปรุงการเรียนรู้และปิดช่องว่างการเรียนรู้ที่เกิดขึ้น
                                                แนวการตอบของระบบ
                                                1. ให้ระบุว่ามาจากการวิเคราะห์ด้วยปัญญาประดิษฐ์เสมอ
                                                2. ระบุข้อมูลประจำตัวนักเรียนดังนี้
                                                \t2.1 รหัสประจำตัวนักเรียน: {row['StudentID']}
                                                \t2.2 คะแนนรวมที่ได้รับ: {row['Earned Points']} คะแนน จากคะแนนเต็ม {len(point_col_list)} คะแนน
                                                \t2.3 ร้อยละของคะแนนที่ได้รับ: {row['Earned Points'] / len(point_col_list) * 100:.2f}%
                                                3. รายระเอียดที่ต้องระบุใน feedback ดังนี้
                                                \t3.1 concept/skill/ความเข้าใจ ที่นักเรียนทำได้ดีแล้วในแต่ละข้อ (เช่น นักเรียนสามารถคำนวณข้อที่เกี่ยวกับพื้นฐานทางไฟฟ้าได้ดีแล้ว)
                                                \t3.2 จุดconcept/skill/ความเข้าใจ ที่ยังทำไม่ได้ โดยต้องระบุทุกจุด(เช่น ในจุดที่นักเรียนต้องพัฒนาต่อไป 1. ... 2. ... 3.ยังมีปัญหาในโจทย์ที่ซับซ้อนเรื่อง ... , 4. ... )
                                                \t3.3 แนวทางในการพัฒนาในจุดที่ยังทำไม่ได้ โดยต้องระบุทุกจุดที่ยังทำไม่ได้ (เช่น แนวทางในการพัฒนาตัวเอง ได้แก่ 1. ... 2. ... 3. ...)
                                                \t3.4 การระบุ concept/skill/ความเข้าใจ ไม่ต้องบอกเลขข้อว่าทำข้อไหนได้หรือไม่ได้ แต่ให้บอกเป็น concept/skill/ความเข้าใจที่นักเรียนทำได้หรือยังทำไม่ได้ในแต่ละข้อสอบแทน
                                                \t3.5 ต้องระบุทุก concept ห้ามตกหล่นรวมถึงข้อ concern จาก error_types
                                                5. ตอบด้วยน้ำเสียงที่เข้าถึงง่ายเหมาะกับเด็กนักเรียนไทยในช่วงมัธยมปลาย 
                                                6. ใช้ภาษาไทยเท่านั้นในการสื่อสารออกไป
                                                7. ถ้าน้องทำคะแนนรวมได้สูงมากให้ชื่นชม แต่ถ้าไม่สูงต้องให้กำลังใจการพัฒนาต่อไปอย่างเป็นธรรมชาติที่มนุษย์คุยกันทั่วไป

                                                การพิจารณาข้อมูลข้อสอบให้พิจารณาทุกด้านของข้อสอบ โดยข้อมูลของข้อสอบมีดังนี้: {str(test_json)}
                                            """)
    
    human_message = HumanMessage(content=f"""นักเรียนที่มีรหัส {row["StudentID"]} ได้รับคะแนนในแต่ละข้อดังนี้ { {k: row[k] for k in point_col_list} } 
                                         กรุณาวิเคราะห์คะแนนที่นักเรียนได้รับในแต่ละข้อ และให้คำแนะนำในการปรับปรุงการทำข้อสอบในอนาคต""")
   
    response = llm.invoke([system_message, human_message] , temperature=0.2)
    
    with open(f"output_feedback.txt", "a", encoding="utf-8") as f:
        f.write(f"{"="*20} Analyzing StudentID: {row['StudentID']} {'='*20}\n")
        f.write(response.content[0]['text'].strip()+f"\n {"*"*80} \n\n")
    from datetime import datetime
    
    end_ocr_page = perf_counter()
    token_meatadata = {str(datetime.now()): callback.usage_metadata,
                        "processing_aggregate_time_seconds": (end_ocr_page - start_ocr_page)} 
    with open("Token_GeminiAPI_usage_log.txt", "a", encoding="utf-8") as log_file:
                    
        log_file.write(json.dumps(token_meatadata, ensure_ascii=False, default=str) + "\n")
                    
    