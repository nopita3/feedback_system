from langgraph.graph import StateGraph, START, END
from Schemes.schema import OverallState
from pathlib import Path
from time import perf_counter
import pandas as pd
import json

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


def graph_process(read_and_split_pdf, process_ocr_page ,continue_to_ocr ,read_student_information, continue_to_feedback, process_feedback):
    builder = StateGraph(OverallState)
    # เพิ่ม Nodes
    builder.add_node("read_and_split_pdf", read_and_split_pdf)
    builder.add_node("process_ocr_page", process_ocr_page)
    builder.add_node("read_student_information",read_student_information)
    builder.add_node("process_feedback", process_feedback)

    # เพิ่ม Edges
    builder.add_edge(START, "read_and_split_pdf")
    builder.add_conditional_edges("read_and_split_pdf", continue_to_ocr)
    builder.add_edge("process_ocr_page", "read_student_information")
    builder.add_conditional_edges("read_student_information", continue_to_feedback)
    builder.add_edge("process_feedback", END)

    # Compile LangGraph
    graph = builder.compile()

    return graph



if __name__ == "__main__":
    # ระบุพาทไปยังไฟล์ PDF ของคุณ
    pdf_file_path = Path("Documents/final_M5_022568.pdf")
    student_test_path = Path("Documents/Intensive_Physics_4.csv")
    
    # เริ่มต้นการทำงาน (Invoke) แบบ Parallel แต่จำกัด request ป้องกัน Rate limit API
    start = perf_counter()
    api_model = "Gemini"

    try:

        if api_model == "Gemini":
            graph = graph_process(pdf_gemini, extracted_gemini,  conditional_edges_gemini
                                  , read_student_information, conditional_feedback_gemini, feedback_gemini_node)
        else:
            graph = graph_process(pdf_ollama, extracted_ollama, conditional_edges_ollama)

        final_state = graph.invoke({"pdf_path": pdf_file_path, 
                                    "student_test_path": student_test_path,
                                    "key_answer": [],
                                    "ocr_results": [], },
                                    config={"max_concurrency": 2}
                                    )
        
        # Convert Pydantic objects to dicts for JSON serialization
        ocr_results = final_state.get("ocr_results", [])
        ocr_results_dicts = [item.model_dump() if hasattr(item, 'model_dump') else item for item in ocr_results]
        
        # Create feedback table (1 row per student)
        feedback_list = final_state.get("feedback", [])
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