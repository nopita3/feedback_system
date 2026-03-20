# 🎓 ระบบประเมินผลนักเรียนอัจฉริยะ
## AI-Powered Student Feedback System

ระบบอัตโนมัติสำหรับประมวลผล OCR ข้อสอบและสร้าง Feedback ให้กับนักเรียนโดยใช้ **Google Gemini Vision API**

---

## 📋 คุณสมบัติหลัก

✅ **OCR อัตโนมัติ** - อ่านข้อสอบจากไฟล์ PDF  
✅ **Feedback ส่วนบุคคล** - สร้าง Feedback เฉพาะตัวสำหรับแต่ละนักเรียน  
✅ **Progress Tracking** - ติดตามความคืบหน้าของการประมวลผลแบบ Real-time  
✅ **Export CSV** - ส่งออกผลลัพธ์เป็นไฟล์ CSV ได้อย่างง่ายดาย  
✅ **Web Interface** - ใช้ Gradio สำหรับ UI ที่ง่ายใช้  

---

## 🚀 วิธีการเรียกใช้

### 1. ติดตั้งการอ้างอิง

```bash
pip install -r requirements.txt
```

### 2. ตั้งค่า API Key

ต้องมี API Key สำหรับ Google Gemini:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 3. เรียกใช้แอปพลิเคชัน

#### วิธี A: ผ่าน Gradio Interface (ประมาณณ)

```bash
python app.py
```

จากนั้นเปิดเบราว์เซอร์ไปที่: `http://localhost:7860`

#### วิธี B: ผ่าน Script โดยตรง

```bash
python graph_process.py
```

---

## 📄 ข้อกำหนดไฟล์อินพุต

### ไฟล์ PDF
- ไฟล์ข้อสอบ (สามารถแยกหน้าได้)
- ความชัดเจน: ขยาย 125% สำหรับการอ่านที่ดีขึ้น

### ไฟล์ CSV
ต้องมีคอลัมน์ต่อไปนี้:

| คอลัมน์ | ประเภท | ตัวอย่าง | หมายเหตุ |
|--------|--------|---------|---------|
| `StudentID` | String | "S001" | รหัสประจำตัวนักเรียน |
| `Earned Points` | Integer | 8 | คะแนนรวมที่ได้รับ |
| `Stu1, Stu2, ...` | String | "1", "2", "3" | คำตอบที่นักเรียนเลือก |
| `Points1, Points2, ...` | Integer | 1, 0, 1 | คะแนนแต่ละข้อ (0 หรือ 1) |
| `PriKey1, PriKey2, ...` | String | "1", "1", "3" | คำตอบที่ถูกต้อง |

#### ตัวอย่าง CSV:
```csv
StudentID,Earned Points,Stu1,Stu2,Stu3,Points1,Points2,Points3,PriKey1,PriKey2,PriKey3
S001,2,1,2,3,1,0,1,1,1,3
S002,3,1,1,3,1,1,1,1,1,3
```

---

## 📊 ผลลัพธ์ที่ได้

### 1. CSV Output: `output_feedback_Gemini_results.csv`
```csv
student_id,total_points,percentage,feedback_details
S001,8,80,"{\""concept/skill/ความเข้าใจ ที่นักเรียนทำได้ดีแล้วในแต่ละข้อ\"": \"...\", ...}"
```

### 2. JSON Output: `output_Aggregate_Gemini_results.json`
```json
[
  {
    "question_id": "1",
    "question_content": "รายละเอียดของโจทย์",
    "skill_tags": ["การวิเคราะห์", "การคำนวณ"],
    "misconcept_type": [{...}],
    "image_description": "รูปภาพประกอบ (ถ้ามี)"
  }
]
```

---

## ⏱️ ระยะเวลาประมวลผล

| สถานการณ์ | เวลา |
|---------|------|
| 1 หน้า + 1 นักเรียน | ~20-30 วินาที |
| 5 หน้า + 5 นักเรียน | ~3-5 นาที |
| 10 หน้า + 10 นักเรียน | ~5-10 นาที |
| 20 หน้า + 20 นักเรียน | ~10-20 นาที |

**หมายเหตุ**: เวลาขึ้นอยู่กับ:
- ความซับซ้อนของข้อสอบ
- ความเร็วการเชื่อมต่อ Internet
- Rate Limit ของ Gemini API

---

## 🏗️ โครงสร้างระบบ (Graph)

```
START
  ↓
[read_and_split_pdf] 📄
  ↓
[process_ocr_page] (Parallel) 🔄
  ↓
[read_student_information] 📊
  ↓
[process_feedback] (Map-Reduce) 💬
  ↓
END ✅
```

### ขั้นตอนการทำงาน

1. **read_and_split_pdf**
   - อ่านไฟล์ PDF
   - แปลงแต่ละหน้าเป็น Base64
   - อ่านคำตอบที่ถูกต้องจาก CSV

2. **process_ocr_page** (Parallel)
   - ส่งแต่ละหน้าไปให้ Gemini Vision API
   - ดึงเนื้อหาโจทย์
   - วิเคราะห์ misconception จากตัวเลือก

3. **read_student_information**
   - อ่านข้อมูลนักเรียน
   - ประมวลผลคะแนนและคำตอบ

4. **process_feedback** (Map-Reduce)
   - สร้าง Feedback สำหรับแต่ละนักเรียน
   - ใช้ข้อมูล OCR เพื่อวิเคราะห์

---

## ⚙️ การตั้งค่าและตัวเลือก

### ไฟล์ `config.py`

```python
# ตั้งค่า Model
MODEL_NAME = "gemini-3.1-flash-lite-preview"
API_KEY = os.getenv("GEMINI_API_KEY")

# ตั้งค่าการประมวลผล
MAX_CONCURRENCY = 2  # จำนวน parallel tasks
ZOOM_FACTOR = 1.25   # ปรับขนาด PDF สำหรับ OCR
```

### ขีดจำกัด Rate Limit
- **Gemini API** มีมาตรฐาน Rate Limit สำหรับฟรี
- ตั้งค่า `MAX_CONCURRENCY = 2` เพื่อเลี่ยง Rate Limit

---

## 🔍 Gradio Interface Guide

### 📁 ขั้นตอนที่ 1: อัปโหลดไฟล์

1. คลิกที่ช่อง "📄 ไฟล์ข้อสอบ (PDF)"
2. เลือกไฟล์ PDF ของข้อสอบ
3. คลิกที่ช่อง "📊 ไฟล์ข้อมูลนักเรียน (CSV)"
4. เลือกไฟล์ CSV ของข้อมูลนักเรียน

### ⚙️ ขั้นตอนที่ 2: เริ่มประมวลผล

1. คลิกปุ่ม "🚀 เริ่มประมวลผล"
2. ติดตามความคืบหน้าบน Progress Bar
3. รอให้ระบบประมวลผลเสร็จสิ้น

### 📥 ขั้นตอนที่ 3: ดาวน์โหลดผลลัพธ์

1. เมื่อเสร็จสิ้น จะเห็นข้อความ "✅ ประมวลผลสำเร็จ"
2. คลิกปุ่ม "📥 ดาวน์โหลด Feedback CSV"
3. ไฟล์ CSV จะถูกดาวน์โหลดไปยังเครื่องของคุณ

---

## 🐛 การแก้ไขปัญหาทั่วไป

### ❌ OpenAI موثر API Key ด้าน
```
Error: Authentication failed
```
**วิธีแก้:**
```bash
export GEMINI_API_KEY="your_valid_api_key"
```

### ❌ File not found
```
Error: [Errno 2] No such file or directory
```
**วิธีแก้:** ตรวจสอบว่าไฟล์ PDF และ CSV มีอยู่ในเส้นทาง

### ❌ Rate Limit Exceeded
```
Error: 429 Too Many Requests
```
**วิธีแก้:** ลดจำนวน `MAX_CONCURRENCY` ใน config.py

### ❌ CSV Column Missing
```
Error: KeyError: 'StudentID'
```
**วิธีแก้:** ตรวจสอบว่าไฟล์ CSV มีชื่อคอลัมน์ที่ถูกต้อง

---

## 📝 Feedback Format

Feedback สำหรับแต่ละนักเรียนจะอยู่ในรูปแบบ JSON:

```json
{
  "concept/skill/ความเข้าใจ ที่นักเรียนทำได้ดีแล้วในแต่ละข้อ": "...",
  "จุดconcept/skill/ความเข้าใจ ที่ยังทำไม่ได้": "...",
  "แนวทางในการพัฒนาในจุดที่ยังทำไม่ได้": "...",
  "concern จาก error_types": "..."
}
```

---

## 📚 Project Structure

```
feedback_system/
├── app.py                           # Gradio Interface (Main)
├── graph_process.py                 # LangGraph Workflow
├── config.py                        # Configuration
├── requirements.txt                 # Dependencies
│
├── Node/                            # Node Modules
│   ├── __init__.py
│   ├── OCR_gemini.py               # Gemini OCR Processing
│   ├── OCR_ollama.py               # Ollama OCR Processing (optional)
│   └── feedback_gemini.py           # Gemini Feedback Generation
│
├── Schemes/                         # Pydantic Schemas
│   ├── __init__.py
│   └── schema.py                   # Data Models
│
├── Documents/                       # Sample Data
│   ├── Intensive_Physics_4.csv
│   └── final_M5_022568.pdf
│
└── history_log/                     # Logs & Results
    ├── output_feedback_*.csv
    ├── output_Aggregate_*.json
    └── Token_GeminiAPI_usage_log.txt
```

---

## 🤝 Contributing

หากคุณพบปัญหา หรือมีข้อเสนอแนะ โปรดแจ้งให้ทราบ

---

## 📄 ใบอนุญาต

© 2024 - สำหรับการใช้งานการศึกษา

---

## 📞 ติดต่อสำหรับความช่วยเหลือ

กรุณาแจ้งรายละเอียดเกี่ยวกับปัญหาที่คุณพบ:
- ชื่อ Error
- ไฟล์ที่เกี่ยวข้อง
- ผลลัพธ์ที่คาดหวัง
