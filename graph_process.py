from langgraph.graph import StateGraph, START, END
from Schemes.schema import OverallState
from pathlib import Path
from time import perf_counter
import json

from Node import OCR_gemini
from Node import OCR_ollama

pdf_gemini = OCR_gemini.read_and_split_pdf
pdf_ollama = OCR_ollama.read_and_split_pdf

extracted_ollama = OCR_ollama.process_ocr_page
extracted_gemini = OCR_gemini.process_ocr_page

summarize_test_ollama = OCR_ollama.aggregate_results
summarize_test_gemini = OCR_gemini.aggregate_results

conditional_edges_gemini = OCR_gemini.continue_to_ocr
conditional_edges_ollama = OCR_ollama.continue_to_ocr




def graph_process(read_and_split_pdf, process_ocr_page, aggregate_results,continue_to_ocr):
    builder = StateGraph(OverallState)
    # เพิ่ม Nodes
    builder.add_node("read_and_split_pdf", read_and_split_pdf)
    builder.add_node("process_ocr_page", process_ocr_page)
    builder.add_node("aggregate_results", aggregate_results) # เพิ่ม Node 

    # เพิ่ม Edges
    builder.add_edge(START, "read_and_split_pdf")

    # หลังจากอ่านไฟล์เสร็จ ใช้ conditional edges จัดการทำ Send Fan-Out
    builder.add_conditional_edges("read_and_split_pdf", continue_to_ocr)
    # พอมันรัน process_ocr_page เสร็จแบบคู่ขนานครบทุกตัว ให้ไหลมารวมที่ aggregate_results
    builder.add_edge("process_ocr_page", "aggregate_results")

    builder.add_edge("aggregate_results", END)

    # Compile LangGraph
    graph = builder.compile()

    return graph



if __name__ == "__main__":
    # ระบุพาทไปยังไฟล์ PDF ของคุณ
    pdf_file_path = Path("Documents/final_M5_022568.pdf")
    
    # เริ่มต้นการทำงาน (Invoke) แบบ Parallel แต่จำกัด request ป้องกัน Rate limit API
    start = perf_counter()
    api_model = "Gemini"

    try:

        if api_model == "Gemini":
            graph = graph_process(pdf_gemini, extracted_gemini, summarize_test_gemini, conditional_edges_gemini)
        else:
            graph = graph_process(pdf_ollama, extracted_ollama, summarize_test_ollama, conditional_edges_ollama)

        final_state = graph.invoke({"pdf_path": pdf_file_path, 
                                    "ocr_results": [], 
                                    "final_compiled_results": []},
                                    config={"max_concurrency": 3}
                                    )
        
        with open(f"output_Aggregate_{api_model}_results.json", "w", encoding="utf-8") as f:
            json.dump(final_state.get("final_compiled_results", []),
                       f, 
                       ensure_ascii=False, 
                       indent=2)

    except Exception as e:
        print(f"An error occurred: {e}")

    end = perf_counter()
    print(f"Total processing time: {end - start:.2f} seconds")