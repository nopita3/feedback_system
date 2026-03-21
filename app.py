import gradio as gr
import pandas as pd
import json
from pathlib import Path
from time import perf_counter
import tempfile
import concurrent.futures
import time
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

# ❌ ถอด gr.Progress ออกเพื่อไม่ให้เกิด Progress bar ซ้ำซ้อน
def process_with_progress(pdf_file, csv_file):
    try:
        if not pdf_file or not csv_file:
            return "❌ กรุณาอัปโหลดไฟล์ทั้งสองไฟล์", None, pd.DataFrame()

        with open(pdf_file, 'rb') as f:
            pdf_content = f.read()
        with open(csv_file, 'rb') as f:
            csv_content = f.read()

        start_time = perf_counter()
        api_model = "Gemini"

        graph = graph_process(
            pdf_gemini, extracted_gemini, conditional_edges_gemini,
            read_student_information, conditional_feedback_gemini, feedback_gemini_node
        )

        initial_state = {
            "pdf_path": pdf_content,
            "student_test_path": csv_content,
            "key_answer": [],
            "ocr_results": [],
        }

        final_state = graph.invoke(initial_state, config={"max_concurrency": 2})

        ocr_results = final_state.get("ocr_results", [])
        ocr_results_dicts = [
            item.model_dump() if hasattr(item, 'model_dump') else item
            for item in ocr_results
        ]
        feedback_list = final_state.get("feedback", [])
        feedback_dicts = [
            fb.model_dump() if hasattr(fb, 'model_dump') else fb
            for fb in feedback_list
        ]

        feedback_table = pd.DataFrame(feedback_dicts)
        output_csv_path = OUTPUT_DIR / f"output_feedback_{api_model}_results.csv"
        output_json_path = OUTPUT_DIR / f"output_Aggregate_{api_model}_results.json"

        feedback_table.to_csv(output_csv_path, index=False, encoding="utf-8")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(ocr_results_dicts, f, ensure_ascii=False, indent=2)

        end_time = perf_counter()
        processing_time = end_time - start_time

        summary = f"✅ ประมวลผลสำเร็จใน {processing_time:.1f} วินาที"
        return summary, str(output_csv_path), feedback_table

    except Exception as e:
        error_msg = f"❌ เกิดข้อผิดพลาดระหว่างประมวลผล: {str(e)}"
        # ส่ง DataFrame สรุป Error คืนไปเพื่อให้หน้าจอไม่ค้าง
        return error_msg, None, pd.DataFrame([{"Status": "Error", "Message": error_msg}])


def create_ui():
    with gr.Blocks(
        title="Feedback System",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
            neutral_hue="slate"
        ),
        css="""
        .main-window {
            background-color: #ffffff;
            border-radius: 16px;
            max-width: 750px;
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
            letter-spacing: -0.5px;
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
        .btn-start:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
        }
        .btn-download {
            background: #10b981 !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 8px !important;
            transition: background 0.2s ease !important;
        }
        .btn-download:hover {
            background: #059669 !important;
        }
        .filename-box {
            border: 1px solid #e2e8f0 !important;
            background: #f8fafc !important;
            border-radius: 8px !important;
            color: #475569 !important;
        }
        /* Custom file upload styling */
        .svelte-11zloen {
            border-radius: 12px !important;
            border: 2px dashed #cbd5e1 !important;
            transition: all 0.2s ease;
        }
        .svelte-11zloen:hover {
            border-color: #6366f1 !important;
            background-color: #f8fafc !important;
        }
        """
    ) as demo:
        with gr.Column(elem_classes="main-window"):
            gr.HTML("<div class='header'>Feedback System</div>")

            # ---------------- State 1: อัปโหลดและปุ่มเริ่ม ----------------
            with gr.Group() as input_group:
                with gr.Row():
                    pdf_file = gr.File(label="Upload Exam File.pdf", file_types=[".pdf"], type="filepath")
                with gr.Row():
                    csv_file = gr.File(label="Upload student File.csv", file_types=[".csv"], type="filepath")
                with gr.Row():
                    start_btn = gr.Button("Start Generate", elem_classes="btn-start")

            # ---------------- State 2 & 3: สถานะรอประมวลผลและผลลัพธ์ ----------------
            with gr.Group(visible=False) as output_group:
                file_info = gr.HTML()
                
                gr.HTML("<div style='margin-top: 15px; font-size: 14px;'>Progression bar...</div>")
                progress_html = gr.HTML()
                
                status_box = gr.HTML()
                
                result_table = gr.Dataframe(interactive=False, visible=False)
                
                # --- แก้ไขส่วนปุ่ม Download ให้ตรงกับ Design ---
                with gr.Row(visible=False) as download_row:
                    gr.Textbox(value="Feedback_output.csv", show_label=False, interactive=False, scale=3, elem_classes="filename-box")
                    download_btn = gr.DownloadButton("download", value=None, scale=1, elem_classes="btn-download")

        # ---------------- Logic การเปลี่ยนสถานะ ----------------
        def handle_process(pdf, csv):
            if not pdf or not csv:
                yield (
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    "<div style='color: red; text-align: center; padding: 20px;'>❌ กรุณาอัปโหลดไฟล์ทั้งสองไฟล์</div>",
                    gr.update(), gr.update(), gr.update(), gr.update()
                )
                return

            pdf_name = Path(pdf).name
            csv_name = Path(csv).name
            
            files_html = f"""
            <div style='display: flex; gap: 15px; justify-content: center; margin-bottom: 20px;'>
                <div style='border: 1px solid #e2e8f0; border-radius: 10px; padding: 8px 16px; display: flex; align-items: center; gap: 10px; background: #f8fafc; color: #334155; font-weight: 500; box-shadow: 0 2px 4px rgba(0,0,0,0.02);'>
                    <span style='font-size:14px;'>📄 {pdf_name}</span>
                </div>
                <div style='border: 1px solid #e2e8f0; border-radius: 10px; padding: 8px 16px; display: flex; align-items: center; gap: 10px; background: #f8fafc; color: #334155; font-weight: 500; box-shadow: 0 2px 4px rgba(0,0,0,0.02);'>
                    <span style='font-size:14px;'>📊 {csv_name}</span>
                </div>
            </div>
            """
            
            prog_25 = """
            <div style='display: flex; align-items: center; gap: 15px;'>
                <div style='flex-grow: 1; background-color: #f1f5f9; border-radius: 999px; height: 12px; overflow: hidden;'>
                    <div style='width: 25%; background: linear-gradient(90deg, #6366f1, #a855f7); height: 100%; border-radius: 999px; transition: width 0.5s ease;'></div>
                </div>
                <span style='font-size: 14px; font-weight: 600; color: #64748b;'>25%</span>
            </div>
            """

            # 1. เปลี่ยนหน้าเริ่มต้น
            yield (
                gr.update(visible=False), # input_group
                gr.update(visible=True),  # output_group
                files_html,               # file_info
                prog_25,                  # progress_html (will be updated)
                "<div style='text-align: center; padding: 40px; border: 1px dashed #cbd5e1; border-radius: 12px; margin-top: 20px; font-size: 15px; color: #64748b; background: #f8fafc;'><span style='display:inline-block; animation: pulse 2s infinite;'>⏳ Analyzing and generating feedback...</span></div>", # status_box
                gr.update(visible=False), # result_table
                gr.update(visible=False), # download_row
                gr.update(value=None),    # download_btn
                gr.update(interactive=False) # start_btn 
            )

            # ประมวลผลจริงโดยใช้ ThreadPoolExecutor เพื่ออัปเดตแกน Progress 1% ต่อวินาที
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(process_with_progress, pdf, csv)
                
                progress = 1
                while not future.done():
                    if progress < 99:
                        progress += 1
                        
                    prog_html = f"""
                    <div style='display: flex; align-items: center; gap: 15px;'>
                        <div style='flex-grow: 1; background-color: #f1f5f9; border-radius: 999px; height: 12px; overflow: hidden;'>
                            <div style='width: {progress}%; background: linear-gradient(90deg, #6366f1, #a855f7); height: 100%; border-radius: 999px; transition: width 0.5s ease;'></div>
                        </div>
                        <span style='font-size: 14px; font-weight: 600; color: #64748b;'>{progress}%</span>
                    </div>
                    """
                    
                    yield (
                        gr.update(), gr.update(), gr.update(),
                        prog_html,
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                    )
                    time.sleep(1)
                
                # เมื่อเสร็จสิ้น นำผลลัพธ์ออกมา
                summary, csv_path, result_df = future.result()

            prog_100 = """
            <div style='display: flex; align-items: center; gap: 15px;'>
                <div style='flex-grow: 1; background-color: #f1f5f9; border-radius: 999px; height: 12px; overflow: hidden;'>
                    <div style='width: 100%; background: linear-gradient(90deg, #10b981, #059669); height: 100%; border-radius: 999px; transition: width 0.5s ease;'></div>
                </div>
                <span style='font-size: 14px; font-weight: 600; color: #10b981;'>100%</span>
            </div>
            """
            
            status_done = "<div style='text-align: center; padding: 40px; border: 1px dashed #10b981; border-radius: 12px; margin-top: 20px; font-size: 18px; color: #10b981; font-weight: bold; background: #ecfdf5;'>✅ Feedback Generated Done</div>"

            # 2. แสดงผล 100% 
            if csv_path is None:
                # กรณี Error
                yield (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    files_html,
                    prog_100,
                    f"<div style='color: #ef4444; text-align: center; padding: 20px; border: 1px solid #f87171; border-radius: 12px; border-radius: 12px; margin-top: 20px; background: #fef2f2; font-weight: 500;'>{summary}</div>",
                    gr.update(visible=True, value=result_df),
                    gr.update(visible=False),
                    gr.update(value=None),
                    gr.update(interactive=True)
                )
            else:
                # กรณีสำเร็จ แสดงผลตาราง และแสดงปุ่ม Download ที่พร้อมโหลดไฟล์
                yield (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    files_html,
                    prog_100,
                    status_done,
                    gr.update(visible=True, value=result_df),
                    gr.update(visible=True),            # เปิดให้เห็นแถวปุ่มดาวน์โหลด
                    gr.update(value=csv_path),          # ยัดไฟล์ CSV ใส่เข้าไปในปุ่ม
                    gr.update(interactive=True)
                )

        start_btn.click(
            fn=handle_process,
            inputs=[pdf_file, csv_file],
            outputs=[input_group, output_group, file_info, progress_html, status_box, result_table, download_row, download_btn, start_btn]
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )