from langgraph.checkpoint.memory import MemorySaver

from pathlib import Path
from time import perf_counter
import pandas as pd
import json

from graph_process import graph_process
from Node import OCR_gemini
from Node import OCR_ollama
from Node import feedback_gemini

pdf_gemini = OCR_gemini.read_and_split_pdf
pdf_ollama = OCR_ollama.read_and_split_pdf

extracted_ollama = OCR_ollama.process_ocr_page
extracted_gemini = OCR_gemini.process_ocr_page

# summarize_test_ollama = OCR_ollama.aggregate_results


conditional_edges_gemini = OCR_gemini.continue_to_ocr
conditional_edges_ollama = OCR_ollama.continue_to_ocr

read_student_information = feedback_gemini.extract_student_information
conditional_feedback_gemini = feedback_gemini.continue_to_feedback
feedback_gemini_node = feedback_gemini.process_feedback


if __name__ == "__main__":
    # ระบุพาทไปยังไฟล์ PDF ของคุณ
    pdf_file_path = Path("Documents/final_M5_022568.pdf")
    student_test_path = Path("Documents/Intensive_Physics_4.csv")
    
    # เริ่มต้นการทำงาน (Invoke) แบบ Parallel แต่จำกัด request ป้องกัน Rate limit API
    start = perf_counter()
    api_model = "Gemini"

    try:

        memory = MemorySaver()
        config = {"configurable": {"thread_id": "1"}, "max_concurrency": 2}

        if api_model == "Gemini":
            graph = graph_process(pdf_gemini, extracted_gemini,  conditional_edges_gemini
                                  , read_student_information, conditional_feedback_gemini, feedback_gemini_node, memory=memory)
        else:
            graph = graph_process(pdf_ollama, extracted_ollama, conditional_edges_ollama, memory=memory)

        print("--- Starting Graph Execution ---")
        for event in graph.stream({"pdf_path": pdf_file_path, 
                                    "student_test_path": student_test_path,
                                    "key_answer": [],
                                    "ocr_results": [], },
                                    config=config):
            for k, v in event.items():
                print(f"Completed node: {k}")

        # Human in the loop step
        state = graph.get_state(config)
        next_step = state.next
        if next_step and "read_student_information" in next_step:
            print("\n--- Human Evaluation Step: OCR Results ---")
            ocr_results = state.values.get("ocr_results", [])
            print(f"Current OCR Results extracted {len(ocr_results)} pages.")
            
            # จำลองหน้าต่างแก้ไขด้วยการสร้างไฟล์ JSON ชั่วคราวให้ User แก้ไขผ่าน VS Code
            temp_edit_file = "temp_ocr_edit.json"
            
            # แปลงข้อมูลเป็น Dict เพื่อเซฟลง JSON
            ocr_results_dicts = [item.model_dump() if hasattr(item, 'model_dump') else item for item in ocr_results]
            with open(temp_edit_file, "w", encoding="utf-8") as f:
                json.dump(ocr_results_dicts, f, ensure_ascii=False, indent=2)
            
            print(f"\n[ACTION REQUIRED] ระบบได้จำลองข้อมูลให้แก้ไขไว้ที่ไฟล์: {temp_edit_file}")
            print(">> ให้คุณเปิดไฟล์นั้นขึ้นมา แก้ไขข้อความที่ต้องการ แล้วกด Save")
            
            user_input = input(">> กด [Enter] เพื่ออัปเดตข้อมูลและไปต่อ (หรือพิมพ์ 'abort' เพื่อยกเลิก): ")
            
            if user_input.lower().strip() != 'abort':
                # โหลดข้อมูลที่ User อาจจะแก้ไขไปแล้วกลับมา
                with open(temp_edit_file, "r", encoding="utf-8") as f:
                    edited_ocr_results = json.load(f)
                
                
                print("\n✅ โหลดข้อมูลแก้ไขเรียบร้อยแล้ว กำลังทำขั้นตอนต่อไป...")
                
                # ใช้ update_state เลี่ยงปัญหา append โดยส่งไปที่ฟิลด์ ocr_user_corrections ที่สร้างใหม่
                graph.update_state(config, {"ocr_user_corrections": edited_ocr_results}, as_node="process_ocr_page")

                # สั่งรันต่อให้จบการทำงานจากจุดที่ค้างไว้
                for event in graph.stream(None, config):
                    for k, v in event.items():
                        print(f"Completed node: {k}")
            else:
                print("Operation stopped by user.")
        
        final_state_values = graph.get_state(config).values
        
        # ถ้ายูสเซอร์แก้ไขก็จะเอา ocr_user_corrections ออกมาบึนทึกเป็น json ไม่ใช่ ocr_results เดิม
        saved_ocr = final_state_values.get("ocr_user_corrections") if final_state_values.get("ocr_user_corrections") else final_state_values.get("ocr_results", [])
        ocr_results_dicts = [item.model_dump() if hasattr(item, 'model_dump') else item for item in saved_ocr]
        
        # Create feedback table (1 row per student)
        feedback_list = final_state_values.get("feedback", [])
        feedback_dicts = [fb.model_dump() if hasattr(fb, 'model_dump') else fb for fb in feedback_list]
        feedback_table = pd.DataFrame(feedback_dicts)

        feedback_table.to_csv(f"output_feedback_{api_model}_results.csv", index=False, encoding="utf-8")
        
        with open(f"output_Aggregate_{api_model}_results.json", "w", encoding="utf-8") as f:
            json.dump(ocr_results_dicts,
                       f, 
                       ensure_ascii=False, 
                       indent=2)
        
        # Save feedback as CSV
        feedback_table.to_csv(f"output_feedback_{api_model}_results.csv", index=False, encoding="utf-8")
    

    except Exception as e:
        print(f"An error occurred: {e}")

    end = perf_counter()
    print(f"Total processing time: {end - start:.2f} seconds")