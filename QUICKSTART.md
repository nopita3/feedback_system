# 🚀 Quick Start Guide
# ⚡ คู่มือเริ่มต้นใช้งานอย่างรวดเร็ว

## Step 1️⃣: ตั้งค่า Environment

### ก. Clone หรือสำรองไฟล์โปรเจกต์

```bash
cd feedback_system
```

### ข. ติดตั้ง Python Dependencies

```bash
pip install -r requirements.txt
```

หรือ ใช้ Quick Run Script:

```bash
python run.py
```

### ค. ตั้งค่า Google Gemini API Key

```bash
# Linux / macOS
export GEMINI_API_KEY="your_gemini_api_key_here"

# Windows
set GEMINI_API_KEY=your_gemini_api_key_here

# PowerShell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
```

---

## Step 2️⃣: เริ่มแอปพลิเคชัน

### ตัวเลือก A: ใช้ Quick Runner (ขอแนะนำ)

```bash
python run.py
```

✨ Script นี้จะ:
- ✅ ตรวจสอบ Dependencies
- ✅ ตรวจสอบ API Key
- ✅ ตรวจสอบไฟล์ตัวอย่าง
- ✅ เรียกใช้ Gradio Interface

### ตัวเลือก B: เรียกใช้ App โดยตรง

```bash
python app.py
```

### ตัวเลือก C: ใช้ LangGraph CLI

```bash
python graph_process.py
```

---

## Step 3️⃣: เตรียมไฟล์อินพุต

### 📄 ไฟล์ PDF
- ไฟล์ข้อสอบในรูปแบบ PDF
- ตัวอย่าง: `Documents/final_M5_022568.pdf`

### 📊 ไฟล์ CSV
ตรวจสอบว่ามีคอลัมน์ต่อไปนี้:
- `StudentID` - รหัสนักเรียน
- `Earned Points` - คะแนนรวม
- `Stu1, Stu2, ...` - คำตอบที่เลือก
- `Points1, Points2, ...` - คะแนนแต่ละข้อ
- `PriKey1, PriKey2, ...` - คำตอบถูก

ตัวอย่าง: `Documents/Intensive_Physics_4.csv`

---

## Step 4️⃣: ใช้งาน Gradio Interface

### 🌐 เปิดเบราว์เซอร์

```
http://localhost:7860
```

### ⬆️ อัปโหลดไฟล์

1. คลิกที่ช่อง **📄 ไฟล์ข้อสอบ (PDF)**
2. เลือกไฟล์ PDF
3. คลิกที่ช่อง **📊 ไฟล์ข้อมูลนักเรียน (CSV)**
4. เลือกไฟล์ CSV

### ▶️ เริ่มประมวลผล

1. คลิกปุ่ม **🚀 เริ่มประมวลผล**
2. ติดตามความคืบหน้าบน Progress Bar
   - 0-15% 🔄 เตรียมข้อมูล
   - 15-60% 📄 OCR ข้อสอบ
   - 60-95% 💬 สร้าง Feedback
   - 100% ✅ เสร็จสิ้น

### 📥 ดาวน์โหลดผลลัพธ์

- เมื่อเสร็จสิ้น จะเห็น ✅ สำเร็จ
- คลิก **📥 ดาวน์โหลด Feedback CSV**

---

## 📊 ผลลัพธ์ที่ได้

### ไฟล์ที่สร้างขึ้น:

```
history_log/
├── output_feedback_Gemini_results.csv     # CSV ที่ดาวน์โหลดได้
├── output_Aggregate_Gemini_results.json   # JSON ผลการ OCR
└── Token_GeminiAPI_usage_log.txt         # Token Usage Log
```

### CSV Feedback Format:

```csv
student_id,total_points,percentage,feedback_details
S001,8,80.0,"{...}"
S002,7,70.0,"{...}"
```

---

## ⏱️ ระยะเวลาประมวลผล (โดยประมาณ)

| PDF Pages | Students | Time |
|-----------|----------|------|
| 5 | 2 | 1-2 นาที |
| 10 | 5 | 2-5 นาที |
| 20 | 10 | 5-10 นาที |
| 50+ | 20+ | 15-30 นาที |

💡 **ขึ้นอยู่กับ:**
- ความซับซ้อนของข้อสอบ
- ความเร็ว Internet
- Rate Limit ของ Gemini API

---

## 🐛 การแก้ปัญหา

### ❌ Error: ModuleNotFoundError

**ปัญหา:**
```
ModuleNotFoundError: No module named 'gradio'
```

**แก้ไข:**
```bash
pip install gradio
```

### ❌ Error: GEMINI_API_KEY not found

**ปัญหา:**
```
Error: GEMINI_API_KEY environment variable not set
```

**แก้ไข:**
```bash
export GEMINI_API_KEY="your_api_key"
python run.py
```

### ❌ Error: File not found

**ปัญหา:**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**แก้ไข:**
1. ตรวจสอบเส้นทางไฟล์
2. วางไฟล์ใน `Documents/` folder

### ❌ Error: 429 Too Many Requests

**ปัญหา:**
```
Error 429: Too Many Requests from Gemini API
```

**แก้ไข:**
1. ลดจำนวน concurrent tasks
2. แก้ไข `config.py`:
   ```python
   MAX_CONCURRENCY = 1
   ```

---

## 📞 ติดต่อเมื่อเกิดปัญหา

หากเกิดข้อผิดพลาด โปรดรวบรวมข้อมูล:

✅ ข้อความ Error ที่เกิดขึ้น  
✅ ชื่อไฟล์ที่อัปโหลด  
✅ ผลลัพธ์ที่คาดหวัง  
✅ Log message (ถ้ามี)

---

## 🎓 ตัวอย่างการใช้งาน

### ตัวอย่าง CSV:
```csv
StudentID,Earned Points,Stu1,Stu2,Stu3,Points1,Points2,Points3,PriKey1,PriKey2,PriKey3
S001,2,1,2,3,1,0,1,1,1,3
S002,3,1,1,3,1,1,1,1,1,3
S003,1,2,2,1,0,0,1,1,1,3
```

### ตัวอย่าง Feedback Output (JSON):
```json
{
  "student_id": "S001",
  "total_points": 8,
  "percentage": 80.0,
  "feedback_details": {
    "concept_success": "นักเรียนเข้าใจเรื่องกฎของ Ohm ได้ดีแล้ว",
    "concept_gap": "ยังไม่เข้าใจเรื่องการวิเคราะห์วงจรขนาน",
    "improvement": "ควรฝึกทำโจทย์เกี่ยวกับวงจรขนานและเกรที่เพิ่มเติม",
    "concern": "คำนวณหา voltage ผิดในข้อ 2"
  }
}
```

---

## 🎯 Tips & Tricks

### 💡 Tip 1: สร้าง CSV ที่ดี

ใช้ Excel หรือ Google Sheets:
1. สร้างแท็บใหม่
2. เพิ่มหมวดหมู่คอลัมน์
3. Export เป็น CSV (UTF-8)

### 💡 Tip 2: PDF ที่ดี

- ความชัดเจน อย่างน้อย 300 DPI
- ไม่มีเงา หรือ blur
- ใช้ภาษาไทยหรืออังกฤษ

### 💡 Tip 3: Batch Processing

สำหรับข้อสอบหลายฉบับ:
1. อัปโหลดทีละหนึ่งฉบับ
2. บันทึกผลลัพธ์
3. รวมล

ูกด้วย Python script

### 💡 Tip 4: API Quota Management

ตรวจสอบการใช้งาน API:
- ดูไฟล์ `Token_GeminiAPI_usage_log.txt`
- ติดตามการใช้ Token

---

## 📚 เอกสารเพิ่มเติม

- 📖 **README_TH.md** - เอกสารที่ครบถ้วน
- 📋 **config.py** - การตั้งค่าระบบ
- 🔗 **graph_process.py** - แสดงโฟลว์การประมวลผล

---

## ✅ Checklist เริ่มต้น

- [ ] ติดตั้ง Python 3.8+
- [ ] ติดตั้ง Dependencies (`pip install -r requirements.txt`)
- [ ] ตั้งค่า GEMINI_API_KEY
- [ ] เตรียมไฟล์ PDF และ CSV
- [ ] เรียกใช้ `python run.py`
- [ ] เปิด http://localhost:7860
- [ ] อัปโหลดไฟล์และทดสอบ

---

## 🎉 เริ่มใช้งาน!

```bash
python run.py
```

ขอให้สำเร็จ! 🚀
