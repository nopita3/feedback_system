from langgraph.graph import StateGraph, START, END
from Schemes.schema import OverallState


def graph_process_seq(read_and_split_pdf, process_ocr_page, read_student_information, process_feedback, memory=None):
    
    # Wrapper function for sequential OCR execution
    def seq_process_ocr_page(state: OverallState):
        results = []
        pages = state.get("pages", [])
        for i, page in enumerate(pages):
            page_state = {
                "page_b64": page,
                "progress": [i, len(pages)],
                "key_list": state.get("key_answer", []),
                "labels": state.get("labels", []),
                "llm_OCR_platform": state.get("llm_OCR_platform", "")
            }
            res = process_ocr_page(page_state)
            if "ocr_results" in res:
                results.extend(res["ocr_results"])
        return {"ocr_results": results}

    # Wrapper function for sequential feedback execution
    def seq_process_feedback(state: OverallState):
        results = []
        ocr_data = state.get("ocr_user_corrections", state.get("ocr_results"))
        students = state.get("student_information", [])
        for i, student in enumerate(students):
            feed_state = {
                "student_information": student,
                "ocr_results": ocr_data,
                "feed_progress": [i, len(students)]
            }
            res = process_feedback(feed_state)
            if "feedback" in res:
                results.extend(res["feedback"])
        return {"feedback": results}

    builder = StateGraph(OverallState)
    # เพิ่ม Nodes แบบถูกครอบด้วย Wrapper ให้ทำงานตามลำดับ
    builder.add_node("read_and_split_pdf", read_and_split_pdf)
    builder.add_node("process_ocr_page", seq_process_ocr_page)
    builder.add_node("read_student_information", read_student_information)
    builder.add_node("process_feedback", seq_process_feedback)

    # เพิ่ม Edges - Sequential Process
    builder.add_edge(START, "read_and_split_pdf")
    builder.add_edge("read_and_split_pdf", "process_ocr_page")
    builder.add_edge("process_ocr_page", "read_student_information")
    builder.add_edge("read_student_information", "process_feedback")
    builder.add_edge("process_feedback", END)

    # Compile LangGraph
    if memory:
        graph = builder.compile(checkpointer=memory, interrupt_before=["read_student_information"])
    else:
        graph = builder.compile()

    return graph


