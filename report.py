import json
import asyncio
import markdown
import os
from playwright.async_api import async_playwright

def convert_to_html(data):
    """Convert the student data into a single HTML document with MathJax and proper Thai font styling"""
    
    html_content = """
    <!DOCTYPE html>
    <html lang="th">
    <head>
        <meta charset="UTF-8">
        <title>Student Feedback</title>
        
        <!-- Google Fonts: Sarabun -->
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap" rel="stylesheet">
        
        <!-- MathJax for rendering academic physics equations -->
        <script>
            MathJax = {
                tex: {
                    inlineMath: [['\\\\(', '\\\\)'], ['$', '$']],
                    displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']]
                },
                svg: { fontCache: 'global' }
            };
        </script>
        <script type="text/javascript" id="MathJax-script" async
          src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
        </script>

        <style>
            @page {
                size: A4;
                margin: 20mm 15mm;
            }
            body {
                font-family: 'Sarabun', sans-serif;
                font-size: 14px;
                line-height: 1.6;
                color: #333;
                background-color: #fff;
                margin: 0;
                padding: 0;
            }
            .page {
                page-break-after: always;
                position: relative;
            }
            .page:last-child {
                page-break-after: auto;
            }
            .title {
                text-align: center;
                font-size: 22px;
                font-weight: bold;
                color: #002b5e;
                margin-bottom: 20px;
            }
            .student-info {
                display: flex;
                justify-content: space-around;
                background-color: #e6f3ff;
                border: 1px solid #99c2ff;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 25px;
                text-align: center;
            }
            .info-box {
                flex: 1;
            }
            .info-box b {
                color: #002b5e;
                font-size: 15px;
            }
            .info-box span {
                display: block;
                margin-top: 5px;
                font-size: 14px;
            }
            .feedback-title {
                font-size: 18px;
                font-weight: bold;
                color: #1f4788;
                border-bottom: 2px solid #1f4788;
                padding-bottom: 5px;
                margin-bottom: 15px;
            }
            .content {
                text-align: left;
            }
            /* Make Markdown output look beautiful */
            .content h1, .content h2, .content h3 {
                color: #1f4788;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            .content p {
                margin-bottom: 10px;
            }
            .content ul, .content ol {
                margin-top: 5px;
                margin-bottom: 15px;
                padding-left: 25px;
            }
            .content li {
                margin-bottom: 5px;
            }
            .content blockquote {
                border-left: 4px solid #1f4788;
                margin: 15px 0;
                padding: 10px 20px;
                background-color: #f8f9fa;
                font-style: italic;
            }
            hr {
                border: 0;
                height: 1px;
                background: #ccc;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
    """
    
    # Process each student
    for _, student in enumerate(data):
        s_id = student.get('student_id', 'N/A')
        pts = student.get('total_points', 'N/A')
        pct = student.get('percentage', 0)
        
        try:
            pct_formatted = f"{float(pct):.2f}%"
        except (ValueError, TypeError):
            pct_formatted = f"{pct}%"
            
        feedback_raw = student.get('feedback_details', '')
        
        # Convert Markdown to HTML
        md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br'])
        feedback_html = md.convert(feedback_raw)
        
        html_content += f"""
        <div class="page">
            <div class="title">รายงานผลการวิเคราะห์ฟิสิกส์ (AI Feedback)</div>
            
            <div class="student-info">
                <div class="info-box">
                    <b>รหัสประจำตัวนักเรียน</b>
                    <span>{s_id}</span>
                </div>
                <div class="info-box">
                    <b>คะแนนที่ได้</b>
                    <span>{pts} / 25</span>
                </div>
                <div class="info-box">
                    <b>คิดเป็นร้อยละ</b>
                    <span>{pct_formatted}</span>
                </div>
            </div>
            
            <div class="feedback-title">รายละเอียดคำแนะนำจาก AI</div>
            <div class="content">
                {feedback_html}
            </div>
        </div>
        """
        
    html_content += """
    </body>
    </html>
    """
    
    return html_content

async def create_student_pdf(data):
    """Use Playwright to render the HTML (which correctly loads fonts, emojis, and MathJax) to PDF"""
    html_content = convert_to_html(data)
    
    # Save temporary html
    temp_html_path = "files_log/temp_render.html"
    pdf_path = "files_log/feedback_results.pdf"
    
    os.makedirs("files_log", exist_ok=True)
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print("Launching Chromium browser to render document with MathJax and proper Emojis...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Load the HTML
        file_url = f"file://{os.path.abspath(temp_html_path)}"
        await page.goto(file_url, wait_until="networkidle")
        
        # Wait for MathJax to finish rendering
        await page.wait_for_function("() => window.MathJax && window.MathJax.startup && window.MathJax.startup.document")
        # Give it a tiny bit extra time just in case to draw SVG
        await page.wait_for_timeout(2000)
        
        # Generate the PDF
        await page.pdf(
            path=pdf_path,
            format="A4",
            print_background=True,
            margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"}
        )
        
        await browser.close()
        
    # Clean up temp file
    if os.path.exists(temp_html_path):
        os.remove(temp_html_path)
        
    print(f"🎉 PDF successfully created natively: {pdf_path}")

def main():
    try:
        with open("files_log/final_feedback_results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]
        
        asyncio.run(create_student_pdf(data))
    except FileNotFoundError:
        print("Error: JSON file not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")

if __name__ == "__main__":
    main()
