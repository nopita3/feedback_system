from pathlib import Path

import json
from graphs.graph_process import graph_process
from graphs.graph_process_sequencial import graph_process_seq
from graphs.graph_process_manual import graph_process as graph_process_manual
from Node import OCR_gemini

from Node import feedback_gemini
from Node import manual_result


pdf_gemini = OCR_gemini.read_and_split_pdf


extracted_gemini = OCR_gemini.process_ocr_page
conditional_edges_gemini = OCR_gemini.continue_to_ocr


read_student_information = feedback_gemini.extract_student_information
conditional_feedback_gemini = feedback_gemini.continue_to_feedback
feedback_gemini_node = feedback_gemini.process_feedback
manual_ocr_node = manual_result.Manual_process_ocr_page

if __name__ == "__main__":
    # ระบุพาทไปยังไฟล์ PDF ของคุณ
    pdf_file_path = Path("Documents/final_M4_022568.pdf")
    student_test_path = Path("Documents/Intensive_Physics_2.csv")
    labels_path = Path("Documents/finish_class_M4.csv")
    
    # เริ่มต้นการทำงาน (Invoke) แบบ Parallel แต่จำกัด request ป้องกัน Rate limit API

    mode = 'manual'
    if mode == 'sequential':
        graph = graph_process_seq(pdf_gemini, 
                                  extracted_gemini, 
                                  read_student_information ,
                                  feedback_gemini_node)
    elif mode == 'manual':
        graph = graph_process_manual(pdf_gemini, 
                                     manual_ocr_node, 
                                     read_student_information, 
                                     conditional_feedback_gemini, 
                                     feedback_gemini_node)
    else:
        graph = graph_process(pdf_gemini, 
                              extracted_gemini,  
                              conditional_edges_gemini,
                              read_student_information,
                              conditional_feedback_gemini,
                              feedback_gemini_node)
    
        
    config = {"configurable": {"thread_id": "1"}, "max_concurrency": 2}
    final_state = graph.invoke({"pdf_path": pdf_file_path, 
                    "student_test_path": student_test_path,
                    'labels_path': labels_path
                    }, 
                    config=config)
    
    ocr_results = final_state.get("ocr_results", [])

    feedback: list = final_state.get("feedback", [])


    with open('files_log/final_ocr_results.json', 'w', encoding='utf-8') as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=2)

    with open('files_log/final_feedback_results.json', 'w', encoding='utf-8') as f:
        feedback_list = [i.model_dump() for i in feedback]
        json.dump(feedback_list, f, ensure_ascii=False, indent=2)