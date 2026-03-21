from langgraph.graph import StateGraph, START, END
from Schemes.schema import OverallState


def graph_process(read_and_split_pdf, process_ocr_page ,continue_to_ocr ,read_student_information, continue_to_feedback, process_feedback , memory=None):
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
    if memory:
        graph = builder.compile(checkpointer=memory, interrupt_before=["read_student_information"])
    else:
        graph = builder.compile()

    return graph


