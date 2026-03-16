from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("openai_api_key")






def aggregate_results(page_data):
    """
    Process a single OCR page result and extract structured data.
    
    Args:
        page_data (str): A single page OCR result (e.g., "--- Page 1 ---\n[...]")
    
    Returns:
        list: Extracted question data as list of dicts
    """
    
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0, format="json" , api_key=api_key)
    
    prompt = (
        "คุณคือผู้ช่วยแยกวิเคราะห์ JSON ที่ได้จากการทำ OCR\n"
        "หน้าที่ของคุณ: นำข้อมูล JSON ของหน้านี้มา และสกัดเฉพาะส่วน 'items' ออกมา\n\n"
        "ถ้าข้อมูลเป็น JSON Array อยู่แล้ว ให้ตอบกลับเป็น JSON Object ที่มีโครงสร้าง:\n"
        "{\n"
        '  "items": [...]\n'
        "}\n\n"
        "ตัวอย่างผลลัพธ์:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "question_id": "1",\n'
        '      "question_content": "เนื้อหาข้อ",\n'
        '      "skill_tags": ["ทักษะ"],\n'
        '      "error_type": "จุดผิดพลาด",\n'
        '      "image_description": "รูปภาพ"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"ข้อมูลของหน้านี้:\n{page_data}"
    )
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"LLM Response:\n{response.content}\n")
        
        data = json.loads(response.content)
        
        # ดึง items ออกมา
        if isinstance(data, dict) and "items" in data:
            items = data["items"]
        elif isinstance(data, list):
            items = data
        else:
            items = []
        
        print(f"Extracted {len(items)} items from this page\n")
        return items
            
    except Exception as e:
        print(f"Error processing page: {e}\n")
        return []


if __name__ == "__main__":
    import re
    
    with open("OCR_results.txt", "r", encoding="utf-8") as f:
        ocr_results_content = f.read()
    
    # แยก "--- Page X ---" sections โดยใช้ regex
    try:
        # Split by page markers แต่เก็บ markers ไว้
        parts = re.split(r'(--- Page \d+ ---)', ocr_results_content)
        
        ocr_pages = []
        # parts จะเป็น: ['', '--- Page 1 ---', 'content1', '--- Page 2 ---', 'content2', ...]
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                # รวม marker กับ content
                page_content = parts[i] + parts[i + 1]
                if page_content.strip():
                    ocr_pages.append(page_content.strip())
        
        if not ocr_pages:
            ocr_pages = [ocr_results_content]
            
    except Exception as e:
        print(f"Error parsing OCR_results.txt: {e}")
        ocr_pages = [ocr_results_content]
    
    print(f"Found {len(ocr_pages)} pages to process\n")
    
    # ประมวลผลแต่ละหน้าทีละหนึ่ง และเก็บผลลัพธ์
    all_results = []
    
    for idx, page_data in enumerate(ocr_pages, 1):
        print(f"========== Processing Page {idx} ==========")
        
        # ส่งหน้านี้ไปยัง LLM
        page_items = aggregate_results(page_data)
        
        # เพิ่มผลลัพธ์เข้าไปใน list
        all_results.extend( page_items)
        
        print(f"Total accumulated items so far: {len(all_results)}\n")
        print(f"Extracted items: {(all_results)}\n")
    
    # บันทึกผลลัพธ์ทั้งหมดลงไฟล์ JSON
    print("Writing final results to Aggregate_results.json...")
    with open("Aggregate_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nDone! Processed {len(ocr_pages)} pages and extracted {len(all_results)} total items.")