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
test_json = json.load(open("output_Aggregate_GeminiAPI_results.json", "r"))





for i, row in sample.iterrows():
    print(f"{"="*20} Analyzing StudentID: {i} {'='*20}")
    row = row.to_dict()
    row = {k: row[k] for k in ["StudentID"] + point_col_list}
    # llm = ChatOllama(model="qwen3.5:cloud", temperature=0, format="json")
    llm, callback  = get_gemini_model(model="gemini-3.1-flash-lite-preview")
    system_message = SystemMessage(content=f"""
                                                ระบบปัญาประดิษฐ์ถูกใช้เพื่อประมวลผลทางภาษาเพื่อวิเคราะห์คะแนนที่นักเรียนได้รับจากการทำข้อสอบในแต่ละข้อ 
                                                ร่วมกับข้อมูลข้อสอบที่นักเรียนใช้สอบในวิชาฟิสิกส์เข้มข้น 4 (Intensive Physics 4) ที่อิงตามหลักสูตรแกนกลางของกระทรวงศึกษาธิการไทย
                                                ในส่วนของวิชาฟิสิกส์ (เพิ่มเติม) 4 เรื่องไฟฟ้าสถิตและไฟฟ้ากระแสตรง 
                                                แนวการตอบของระบบ
                                                1. ให้ระบุว่ามาจากการวิเคราะห์ด้วยปัญญาประดิษฐ์เสมอ
                                                2. บอกผลการวิเคราะห์ว่าเป็นของนักเรียนรหัสใด และร้อยละของคะแนนที่ได้รับ เช่น จากการวิเคราะห์ข้อมูลของนักเรียนรหัส 12345 พบว่า ...
                                                3. ผลการวิเคราะห์ที่ไม่ต้องระบุว่าผิดข้อไหน(เช่น (ข้อ17) ไม่ต้องใส่มาแล้วนะ) แต่บอกเป็นภาพรวมในจุดที่ทำได้แล้ว และจุดที่ยังทำไม่ได้แ ต่ต้องแบบทุก concept นะอย่าสรุปจนมีสาระใดที่หายไปเพราะนั่นคือการปะรอยรั่วที่ไม่หมดยังไงก็จะรั่วต่อไป (เช่น นักเรียนสามารถคำนวณข้อที่เกี่ยวกับพื้นฐานทางไฟฟ้าได้ และ .... แต่ยังมีปัญหาในโจทย์ที่ซับซ้อนเรื่อง ... )
                                                4. ให้แนวทางในการพัฒนาในจุดที่ยังทำไม่ได้
                                                5. ตอบด้วยน้ำเสียงที่เข้าถึงง่ายเหมาะกับเด็กนักเรียนไทยในช่วงมัธยมปลาย และใช้ภาษาไทยเท่านั้นในการสื่อสารออกไป
                                                6. ถ้าน้องทำคะแนนรวมได้สูงมากให้ชื่อชน แต่ถ้าไม่สูงต้องให้กำลังใจการพัฒนาต่อไปอย่างเป็นธรรมชาติที่มนุษย์คุยกันทั่วไป

                                                โดยข้อมูลของข้อสอบมีดังนี้: {str(test_json)}
                                            """)
    
    human_message = HumanMessage(content=f"""นักเรียนที่มีรหัส {row["StudentID"]} ได้รับคะแนนในแต่ละข้อดังนี้ { {k: row[k] for k in point_col_list} } 
                                         กรุณาวิเคราะห์คะแนนที่นักเรียนได้รับในแต่ละข้อ และให้คำแนะนำในการปรับปรุงการทำข้อสอบในอนาคต""")
   
    response = llm.invoke([system_message, human_message])
    print(response.content[0]['text'].strip())
    
    print(f"token usage: {callback.usage_metadata}\n")
    print("\n")