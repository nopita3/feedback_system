
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore", message=".*Deserializing.*")
from pathlib import Path
from time import perf_counter
import tempfile
import concurrent.futures
import time
import uuid
import gradio as gr

from langgraph.checkpoint.memory import MemorySaver

from Node import OCR_gemini, feedback_gemini
from graph_process import graph_process

# Configure models
pdf_gemini = OCR_gemini.read_and_split_pdf
extracted_gemini = OCR_gemini.process_ocr_page
conditional_edges_gemini = OCR_gemini.continue_to_ocr
read_student_information = feedback_gemini.extract_student_information
conditional_feedback_gemini = feedback_gemini.continue_to_feedback
feedback_gemini_node = feedback_gemini.process_feedback

# Output directory
OUTPUT_DIR = Path(tempfile.gettempdir()) / "feedback_system_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Global graph instance with MemorySaver enabled to support interruption & resuming
global_memory = MemorySaver()
graph = graph_process(
    pdf_gemini, extracted_gemini, conditional_edges_gemini,
    read_student_information, conditional_feedback_gemini, feedback_gemini_node,
    memory=global_memory
)

def run_ocr_process(pdf_file, csv_file, thread_id):
    try:
        if not pdf_file or not csv_file:
            return False, "❌ กรุณาอัปโหลดไฟล์ทั้งสองไฟล์"

        with open(pdf_file, 'rb') as f:
            pdf_content = f.read()
        with open(csv_file, 'rb') as f:
            csv_content = f.read()

        with open(pdf_file, 'rb') as f:
            pdf_content = f.read()
        with open(csv_file, 'rb') as f:
            csv_content = f.read()

        config = {"configurable": {"thread_id": thread_id}, "max_concurrency": 2}

        initial_state = {
            "pdf_path": pdf_content,
            "student_test_path": csv_content,
            "key_answer": [],
            "ocr_results": [],
        }

        # Run the graph until the breakpoint (interrupt_before=["read_student_information"])
        for event in graph.stream(initial_state, config=config):
            pass

        state = graph.get_state(config)
        ocr_results = state.values.get("ocr_results", [])
        
        ocr_results_dicts = [
            item.model_dump() if hasattr(item, 'model_dump') else item
            for item in ocr_results
        ]
        
        # Flatten the dictionary list into a DataFrame-friendly format for easy editing
        df_list = []
        for item in ocr_results_dicts:
            skill_tags = ", ".join(item.get("skill_tags", [])) if isinstance(item.get("skill_tags"), list) else ""
            
            misconcepts = []
            if isinstance(item.get("misconcept_type"), list):
                for m in item.get("misconcept_type"):
                    if isinstance(m, dict):
                        for k, v in m.items():
                            misconcepts.append(f"{k}: {v}")
            misc_str = " | ".join(misconcepts)
            
            df_list.append({
                "question_id": str(item.get("question_id", "")),
                "question_content": str(item.get("question_content", "")),
                "skill_tags": skill_tags,
                "misconcept_type": misc_str,
                "image_description": str(item.get("image_description", ""))
            })
            
        return True, pd.DataFrame(df_list)

    except Exception as e:
        return False, f"❌ เกิดข้อผิดพลาดระหว่าง OCR: {str(e)}"

def run_feedback_process(edited_df, thread_id):
    try:
        # Reconstruct the expected list of OCR dicts from the user's DataFrame edits
        edited_ocr_results = []
        for _, row in edited_df.iterrows():
            # Parse skill tags back to list
            tags_str = str(row.get("skill_tags", ""))
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            
            # Parse misconcept_type back to list of dicts
            misc_str = str(row.get("misconcept_type", ""))
            misc_list = []
            for part in misc_str.split("|"):
                if ":" in part:
                    k, v = part.split(":", 1)
                    misc_list.append({k.strip(): v.strip()})
                    
            edited_ocr_results.append({
                "question_id": str(row.get("question_id", "")),
                "question_content": str(row.get("question_content", "")),
                "skill_tags": tags,
                "misconcept_type": misc_list,
                "image_description": str(row.get("image_description", ""))
            })

        config = {"configurable": {"thread_id": thread_id}, "max_concurrency": 2}
        
        # Set the user corrections as state, pretending it came from OCR processing node
        graph.update_state(config, {"ocr_user_corrections": edited_ocr_results}, as_node="process_ocr_page")
        
        start_time = perf_counter()
        
        # Resume the graph from the point it was interrupted
        for event in graph.stream(None, config=config):
            pass
            
        final_state_values = graph.get_state(config).values
        end_time = perf_counter()
        processing_time = end_time - start_time
        
        # Process and save outcomes
        saved_ocr = final_state_values.get("ocr_user_corrections") if final_state_values.get("ocr_user_corrections") else final_state_values.get("ocr_results", [])
        ocr_results_dicts = [item.model_dump() if hasattr(item, 'model_dump') else item for item in saved_ocr]
        
        feedback_list = final_state_values.get("feedback", [])
        feedback_dicts = [
            fb.model_dump() if hasattr(fb, 'model_dump') else fb
            for fb in feedback_list
        ]

        feedback_table = pd.DataFrame(feedback_dicts)
        api_model = "Gemini"
        
        output_csv_path = OUTPUT_DIR / f"output_feedback_{api_model}_results_{thread_id}.csv"
        output_json_path = OUTPUT_DIR / f"output_Aggregate_{api_model}_results_{thread_id}.json"

        feedback_table.to_csv(output_csv_path, index=False, encoding="utf-8")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(ocr_results_dicts, f, ensure_ascii=False, indent=2)

        summary = f"✅ ประมวลผลสำเร็จใน {processing_time:.1f} วินาที"
        return True, summary, str(output_csv_path), feedback_table

    except Exception as e:
        return False, f"❌ เกิดข้อผิดพลาดระหว่างสร้าง Feedback: {str(e)}", None, pd.DataFrame([{"Status": "Error"}])

def create_ui():
    with gr.Blocks(
        title="Feedback System",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", neutral_hue="slate"),
        css="""
        .main-window {
            background-color: #ffffff;
            border-radius: 16px;
            max-width: 850px;
            margin: 40px auto;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.08);
            border: 1px solid #f1f5f9;
        }
        .header {
            background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 32px;
            font-weight: 800;
            text-align: center;
            padding: 10px;
            margin-bottom: 30px;
        }
        .btn-start {
            background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            padding: 14px 28px !important;
            transition: all 0.2s ease !important;
            width: 60% !important;
            margin: 10px auto !important;
            display: block !important;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
        }
        .btn-start:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important; }
        .btn-download { background: #10b981 !important; color: white !important; font-weight: 600 !important; border-radius: 8px !important; }
        .btn-download:hover { background: #059669 !important; }
        """
    ) as demo:
        session_id = gr.State(lambda: str(uuid.uuid4()))

        with gr.Column(elem_classes="main-window"):
            gr.HTML("<div class='header'>Feedback System</div>")

            # --- State 1: Input group ---
            with gr.Group() as input_group:
                with gr.Row():
                    pdf_file = gr.File(label="Upload Exam File.pdf", file_types=[".pdf"], type="filepath")
                with gr.Row():
                    csv_file = gr.File(label="Upload student File.csv", file_types=[".csv"], type="filepath")
                with gr.Row():
                    start_ocr_btn = gr.Button("Start Extracting OCR", elem_classes="btn-start")

            # --- Intermediate Processing View (Shared) ---
            with gr.Group(visible=False) as processing_group:
                file_info = gr.HTML()
                status_box = gr.HTML()

            # --- State 2: OCR Review Group ---
            with gr.Group(visible=False) as ocr_review_group:
                gr.HTML("<h3 style='text-align: center; color: #4f46e5;'>✅ OCR Extracted. Review and Correct Data Below</h3>")
                ocr_edit_df = gr.Dataframe(
                    label="OCR Results (Edit in Table)",
                    interactive=True,
                    wrap=True,
                    headers=["question_id", "question_content", "skill_tags", "misconcept_type", "image_description"],
                    datatype=["str", "str", "str", "str", "str"]
                )
                confirm_feedback_btn = gr.Button("Confirm & Generate Feedback", elem_classes="btn-start")

            # --- State 3: Output Group ---
            with gr.Group(visible=False) as result_group:
                result_table = gr.Dataframe(interactive=False, visible=True)
                with gr.Row(visible=False) as download_row:
                    gr.Textbox(value="Feedback_output.csv", show_label=False, interactive=False, scale=3, elem_classes="filename-box")
                    download_btn = gr.DownloadButton("Download CSV", value=None, scale=1, elem_classes="btn-download")


        # Handlers
        def handle_start_ocr(pdf, csv, s_id):
            if not pdf or not csv:
                yield (
                    gr.update(), gr.update(visible=True), gr.update(),
                    "<div style='color: red; text-align: center; padding: 20px;'>❌ กรุณาอัปโหลดไฟล์ทั้งสองไฟล์</div>",
                    gr.update(), gr.update(), gr.update()
                )
                return

            pdf_name = Path(pdf).name
            csv_name = Path(csv).name
            
            files_html = f"""
            <div style='display: flex; gap: 15px; justify-content: center; margin-bottom: 20px;'>
                <div style='border: 1px solid #e2e8f0; border-radius: 10px; padding: 8px 16px; display: flex; align-items: center; gap: 10px; background: #f8fafc; color: #334155;'>
                    <span>📄 {pdf_name}</span>
                </div>
                <div style='border: 1px solid #e2e8f0; border-radius: 10px; padding: 8px 16px; display: flex; align-items: center; gap: 10px; background: #f8fafc; color: #334155;'>
                    <span>📊 {csv_name}</span>
                </div>
            </div>
            """

            # Show processing
            yield (
                gr.update(visible=False), # input_group
                gr.update(visible=True),  # processing_group
                files_html,               # file_info
                "<div style='text-align: center; padding: 40px; border: 1px dashed #cbd5e1; border-radius: 12px; margin-top: 20px; font-size: 18px; color: #64748b;'><span style='animation: pulse 2s infinite;'>⏳ กำลังสกัดข้อความด้วย OCR...<br><br><span style='font-size: 15px; color: #ef4444;'>ขั้นตอนนี้อาจใช้เวลาหลายนาที โปรดรอสักครู่เผื่อให้ทำงานเสร็จสมบูรณ์</span></span></div>", # status_box
                gr.update(visible=False), # ocr_review_group
                gr.update(),              # ocr_edit_df
                gr.update(visible=True, interactive=True) # confirm_feedback_btn
            )

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_ocr_process, pdf, csv, s_id)
                success, result = future.result()

            if success:
                # Finished OCR, switch to Review state
                yield (
                    gr.update(), 
                    gr.update(visible=False), 
                    gr.update(), gr.update(),
                    gr.update(visible=True), 
                    gr.update(value=result),
                    gr.update(visible=True, interactive=True, value="Confirm & Generate Feedback")
                )
            else:
                yield (
                    gr.update(), 
                    gr.update(visible=True), 
                    gr.update(), 
                    f"<div style='color: red; text-align: center; padding: 20px;'>{result}</div>",
                    gr.update(), 
                    gr.update(),
                    gr.update()
                )

        start_ocr_btn.click(
            fn=handle_start_ocr,
            inputs=[pdf_file, csv_file, session_id],
            outputs=[input_group, processing_group, file_info, status_box, ocr_review_group, ocr_edit_df, confirm_feedback_btn]
        )


        def handle_confirm_feedback(edited_df, s_id):
            # Back to processing UI
            yield (
                gr.update(visible=False), # ocr_review_group
                gr.update(visible=True),  # processing_group
                "<div style='text-align: center; padding: 40px; border: 1px dashed #cbd5e1; border-radius: 12px; margin-top: 20px; font-size: 18px; color: #64748b;'><span style='animation: pulse 2s infinite;'>⏳ กำลังวิเคราะห์คำตอบและสร้าง Feedback...<br><br><span style='font-size: 15px; color: #ef4444;'>ขออภัยที่ต้องให้รอนาน ตัวแบบกำลังพิจารณาอย่างละเอียด โปรดรอสักครู่</span></span></div>", # status_box
                gr.update(visible=False), # result_group
                gr.update(),              # result_table
                gr.update(),              # download_row
                gr.update(),              # download_btn
                gr.update(interactive=False, value="⏳ Processing...") # confirm_feedback_btn
            )

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_feedback_process, edited_df, s_id)
                success, summary, csv_path, df = future.result()

            if success:
                status_done = f"<div style='text-align: center; padding: 20px; border: 1px solid #10b981; border-radius: 12px; margin-top: 20px; font-size: 16px; color: #10b981; background: #ecfdf5;'>{summary}</div>"
                yield (
                    gr.update(), 
                    gr.update(visible=True), 
                    status_done,
                    gr.update(visible=True),
                    gr.update(value=df), 
                    gr.update(visible=True), 
                    gr.update(value=csv_path),
                    gr.update(interactive=False, value="✅ Complete")
                )
            else:
                yield (
                    gr.update(visible=True), 
                    gr.update(visible=True), 
                    f"<div style='color: red; text-align: center;'>{summary}</div>",
                    gr.update(visible=True),
                    gr.update(value=pd.DataFrame()),
                    gr.update(visible=False), 
                    gr.update(),
                    gr.update(visible=True, interactive=True, value="Confirm & Generate Feedback")
                )


        confirm_feedback_btn.click(
            fn=handle_confirm_feedback,
            inputs=[ocr_edit_df, session_id],
            outputs=[ocr_review_group, processing_group, status_box, result_group, result_table, download_row, download_btn, confirm_feedback_btn]
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)

