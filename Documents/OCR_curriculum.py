import sys
from pathlib import Path
from fcntl import flock, LOCK_EX, LOCK_UN
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage , HumanMessage

# Allow running as a script from project root
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import get_ollama_model
from Schemes.schema import Curriculm , CurriculumState
from Node.pdf_base64 import read_and_split_pdf




def OCR_curriculum(state: CurriculumState):
    
    llm , _  = get_ollama_model()

    

    system_prompt =SystemMessage(content="""*ระบบ OCR หลักสูตรการเรียนรู้ ของวิชาฟิสิกส์เข้มข้น 4 (Intensive Physics 4) ที่อิงตามหลักสูตรแกนกลางของกระทรวงศึกษาธิการไทย*
    *สกัดข้อมูลตจากส่วนที่เป็นตารางในรูปภาพเท่านั้น*
    *ระบบนนี้มีหน้าที่สกัดผลการเรียนรู้ว่ามีสาระการเรียนอะไรบ้างที่เกี่ยวข้องกับข้อสอบแต่ละข้อ*
    *ระบบจะได้รับข้อมูลรูปภาพเป็น bytes และสกัดข้อมูลของข้อสอบแต่ละข้อที่ได้จากการทำ OCR มาแล้วในรูปแบบ นี้:*
    [{'*ผลการเรียนรู้....': ['*สาระการเรียนรู้ที่เกี่ยวข้องของข้อสอบข้อ 1', ...]}, ...]
    """)


    llm_structured = llm.with_structured_output(Curriculm)
    i=1

    for page in state['pages']:
        print(f'Processing page {i} with OCR...')
        
        print(page[:100])
        human_prompt = HumanMessage(content=[{"type": "text", "text": f"นี่คือรูปภาพที่รายละเอียดของหลักสูตร:"},
                                             {"type": "image_url", "image_url": f"data:image/png;base64,{page}"},])
        response = llm_structured.invoke([system_prompt, human_prompt])
        items = response.model_dump()
        print(f"Extracted curriculum analysis for page {i}: {items}")
        state['curriculum_analysis'].append({"assessments": items})
        i += 1
        
    
    return state

builder = StateGraph(CurriculumState)
builder.add_node(read_and_split_pdf, "read_and_split_pdf")
builder.add_node(OCR_curriculum, "OCR_curriculum")
builder.add_edge(START, "read_and_split_pdf")
builder.add_edge("read_and_split_pdf", "OCR_curriculum")
builder.add_edge("OCR_curriculum", END)


def main():
    base_dir = Path(__file__).resolve().parents[1]
    pdf_path = str(base_dir / "Documents/cur_intensive_physics_4.pdf")
    log_path = str(base_dir / "Curriculum_OCR_result.txt")
    graph = builder.compile()
    result = graph.invoke({'curriculum_path': pdf_path, 'pages': [], 'curriculum_analysis': []})
    output = result.get('curriculum_analysis', [])
    with open(log_path, "a", encoding="utf-8") as log_file:
        flock(log_file.fileno(), LOCK_EX)
        log_file.write(f"Curriculum OCR Result: {output}\n")
        flock(log_file.fileno(), LOCK_UN)


if __name__ == "__main__":
    main()
    