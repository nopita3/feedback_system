# Context Caching Implementation Guide

## Overview
Context Caching ได้ถูก implement เพื่อลด token usage จาก Gemini API โดยการ cache prompt ที่ส่งไปแล้ว

## วิธีการทำงาน

### 1. **Automatic Cache Control**
ทุก request ไป Gemini API จะมี `cache_control {"type": "ephemeral"}` แนบมาซึ่งจะ:
- Cache system prompt และ instruction ที่นำเข้ามา
- ลด token cost ได้ถึง 90% สำหรับ repeated context
- ยังคงมีผลเฉพาะในระหว่าง conversation session เดียวกัน

### 2. **Configuration**
ใน `config.py`:
```python
# ไปที่ get_gemini_model() function
llm, callback = get_gemini_model(
    model="gemini-3.1-flash-lite-preview",
    enable_cache=True  # Enable context caching
)
```

### 3. **Using Cached Messages**
ใน `OCR_gemini.py`:

#### สำหรับ simple text messages:
```python
from config import create_cached_message

message = create_cached_message(
    content="Your prompt here",
    cache_control=True  # Enable cache_control
)
response = llm.invoke([message])
```

#### สำหรับ complex content (text + image):
```python
content = [
    {"type": "text", "text": prompt_text},
    {"type": "image_url", "image_url": f"data:image/png;base64,{page_b64}"}
]

# Add cache control to first element
if content:
    content[0]["cache_control"] = {"type": "ephemeral"}

message = HumanMessage(content=content)
response = llm.invoke([message])
```

## Token Usage Metrics

### Monitoring Cache Performance
```python
from cache_helper import extract_cache_metrics, calculate_cache_savings, print_cache_summary

# Get metrics from callback
metrics = extract_cache_metrics(callback.usage_metadata)

# Calculate savings
savings = calculate_cache_savings(metrics)

# Print summary
print_cache_summary(callback.usage_metadata, "process_name")
```

### Log Entry Format
ข้อมูลต่อไปนี้จะถูก track:
- `cache_creation_input_tokens`: Token ที่ใช้ในการสร้าง cache
- `cache_read_input_tokens`: Token ที่ประหยัดจากการอ่าน cache (cost 90% less)
- `input_tokens`: Token ทั่วไป
- `output_tokens`: Token output ที่สร้าง
- `total_tokens`: รวมทั้งหมด

## Cost Savings Example

หากทำ OCR บน 10 หน้า PDF ด้วย prompt เดียวกัน:

**ไม่มี Caching:**
- Page 1: 2,000 tokens (prompt = 1,000 + image = 1,000)
- Page 2-10: 19,000 tokens (9 × 2,000)
- **Total: 21,000 tokens**

**มี Caching (Ephemeral):**
- Page 1: 2,000 tokens (cache creation)
- Page 2-10: 1,900 tokens (9 × ~211 tokens) - prompt cached, only image + 90% discount
- **Total: ~3,900 tokens** 
- **Savings: ~82% reduction** ✨

## Implementation Details

### Files Modified:
1. **config.py**
   - Added `enable_cache` parameter to `get_gemini_model()`
   - Added `create_cached_message()` helper function

2. **Node/OCR_gemini.py**
   - Updated `process_ocr_page()` to use cache control in message
   - Updated `aggregate_results()` to use `create_cached_message()`
   - Imports `create_cached_message` from config

3. **cache_helper.py** (NEW)
   - `extract_cache_metrics()` - Extract cache info from usage metadata
   - `calculate_cache_savings()` - Calculate token/cost savings
   - `log_cache_metrics()` - Log metrics to file
   - `print_cache_summary()` - Display human-readable summary

## Cache Limitations

⚠️ **Important:**
- Ephemeral cache มีอายุ 5 นาที ต่อคำขอ API
- หากต้อง cache ยาวนาน ให้ใช้ multi-turn conversation แทน
- Cache เหมาะสำหรับ batch processing ที่มี prompt ซ้ำๆ

## Future Improvements

- [ ] Implement persistent cache (prompt caching)
- [ ] Add cache statistics to Token_analytics.py
- [ ] Create dashboard for cache performance monitoring
- [ ] Automatic prompt optimization for caching efficiency
