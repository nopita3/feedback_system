"""
🎓 AI-Powered Student Feedback System
วปรับปรุง UI/UX - ทำให้ใช้งานง่ายและสวยงาม
"""

import gradio as gr
import pandas as pd
import json
from pathlib import Path
from time import perf_counter
import tempfile
from io import BytesIO
import fitz

from Schemes.schema import OverallState
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


def process_with_progress(pdf_file, csv_file, progress: gr.Progress = gr.Progress()):
    """Process files with progress tracking"""
    try:
        if not pdf_file or not csv_file:
            return "❌ กรุณาอัปโหลดไฟล์ทั้งสองไฟล์", None

        progress(0.1, desc="📖 กำลังอ่านไฟล์...")

        # Read files as bytes
        with open(pdf_file, 'rb') as f:
            pdf_content = f.read()
        with open(csv_file, 'rb') as f:
            csv_content = f.read()

        # Analyze files
        pdf_stream = BytesIO(pdf_content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        num_pages = len(doc)
        df = pd.read_csv(BytesIO(csv_content))
        num_students = len(df)

        progress(0.15, desc=f"📊 พบ {num_pages} หน้า, {num_students} นักเรียน")

        start_time = perf_counter()
        api_model = "Gemini"

        # Build and invoke graph
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

        progress(0.25, desc="⚙️ เริ่มประมวลผล...")
        final_state = graph.invoke(initial_state, config={"max_concurrency": 2})

        progress(0.85, desc="💾 บันทึกผลลัพธ์...")

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

        progress(1.0, desc="✅ เสร็จสิ้น!")

        end_time = perf_counter()
        processing_time = end_time - start_time

        # Create summary
        summary = f"""
        ✅ **ประมวลผลสำเร็จ!**
        
        📊 **ผลสรุป:**
        • หน้าข้อสอบ: {num_pages}
        • จำนวนนักเรียน: {len(feedback_dicts)}
        • ข้อสอบที่ OCR: {len(ocr_results_dicts)}
        • ระยะเวลา: {processing_time:.1f} วินาที
        
        💾 **ไฟล์บันทึก:**
        - CSV Feedback: `output_feedback_Gemini_results.csv`
        - JSON OCR: `output_Aggregate_Gemini_results.json`
        """

        return summary, str(output_csv_path)

    except Exception as e:
        error_msg = f"❌ เกิดข้อผิดพลาด: {str(e)}"
        return error_msg, None


def create_ui():
    """Create modern, user-friendly interface"""
    
    with gr.Blocks(
        title="📊 ระบบประเมินผลนักเรียน",
        theme=gr.themes.Soft(),
        css="""
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            border-radius: 12px;
            margin-bottom: 30px;
            text-align: center;
        }
        .header h1 { color: white; margin: 0; }
        .step-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #667eea;
            margin-bottom: 20px;
        }
        .result-box {
            background: #f0f7ff;
            padding: 20px;
            border-radius: 12px;
            border: 2px solid #667eea;
        }
        .success-box {
            background: #f0fdf4;
            padding: 20px;
            border-radius: 12px;
            border: 2px solid #10b981;
        }
        .file-box {
            background: white;
            padding: 20px;
            border: 2px dashed #ccc;
            border-radius: 12px;
            text-align: center;
        }
        """
    ) as demo:

        # Header
        gr.HTML("""
        <div class="header">
            <h1>📊 ระบบประเมินผลนักเรียน AI</h1>
            <p style="margin: 10px 0 0 0; font-size: 16px;">อ่านข้อสอบ OCR และสร้าง Feedback อัตโนมัติ</p>
        </div>
        """)

        # Main content
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📁 ขั้นตอนที่ 1: อัปโหลด")
                
                with gr.Group():
                    pdf_file = gr.File(
                        label="📄 ไฟล์ข้อสอบ (PDF)",
                        file_types=[".pdf"],
                        type="filepath"
                    )

                with gr.Group():
                    csv_file = gr.File(
                        label="📊 ไฟล์ข้อมูลนักเรียน (CSV)",
                        file_types=[".csv"],
                        type="filepath"
                    )

            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ ขั้นตอนที่ 2: ประมวลผล")
                
                process_btn = gr.Button(
                    "🚀 เริ่มประมวลผล",
                    variant="primary",
                    size="lg",
                    scale=1
                )

                gr.Markdown("## 📊 ความคืบหน้า")
                progress_bar = gr.Progress(track_tqdm=False)

        # Results section
        gr.Markdown("---")
        gr.Markdown("## 📋 ผลลัพธ์")

        with gr.Row():
            with gr.Column(scale=2):
                status_display = gr.Markdown(
                    value="⏳ พร้อมประมวลผล...",
                )

            with gr.Column(scale=1):
                download_btn = gr.File(
                    label="📥 ดาวน์โหลด CSV",
                    interactive=False
                )

        # Info section (collapsible)
        with gr.Accordion("📖 คำแนะนำและข้อกำหนด", open=False):
            gr.Markdown("""
            ### ✅ ข้อกำหนดไฟล์ CSV
            
            ไฟล์ CSV ต้องมีคอลัมน์ต่อไปนี้:
            - `StudentID` - รหัสนักเรียน
            - `Earned Points` - คะแนนรวม
            - `Stu1, Stu2, ...` - คำตอบที่เลือก
            - `Points1, Points2, ...` - คะแนนแต่ละข้อ
            - `PriKey1, PriKey2, ...` - คำตอบที่ถูกต้อง
            
            ### ⏱️ ระยะเวลาประมวลผล
            - 10 หน้า + 5 นักเรียน = ~3-5 นาที
            
            ### 📊 ผลลัพธ์
            - CSV: Feedback สำหรับแต่ละนักเรียน
            - JSON: ผลการ OCR (บันทึกในเซิร์ฟเวอร์)
            """)

        # Event handler
        def handle_process(pdf, csv, progress=gr.Progress()):
            return process_with_progress(pdf, csv, progress)

        process_btn.click(
            fn=handle_process,
            inputs=[pdf_file, csv_file],
            outputs=[status_display, download_btn]
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
