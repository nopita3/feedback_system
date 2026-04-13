from pathlib import Path


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

    api_model = "Gemini"
    

    config = {"configurable": {"thread_id": "1"}, "max_concurrency": 2}

    if api_model == "Gemini":
        graph = graph_process(pdf_gemini, extracted_gemini,  conditional_edges_gemini
                                  , read_student_information, conditional_feedback_gemini, feedback_gemini_node)
    else:
        graph = graph_process(pdf_ollama, extracted_ollama, conditional_edges_ollama)
    
    graph.invoke({"pdf_path": pdf_file_path, 
                    "student_test_path": student_test_path,
                    }, 
                    config=config)
    
    

       