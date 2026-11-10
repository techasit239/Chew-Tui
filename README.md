# Chew-Tui — Mindful Chewing เคี้ยวช้าลงอย่างมีสติ

เว็บแอปนับการเคี้ยวผ่านกล้องในเบราว์เซอร์ ด้วย MediaPipe Face Mesh
วงแหวนเต็มเมื่อเคี้ยวครบตามเป้า แล้วค่อยกลืน อ้าปากตักคำใหม่ ตัวนับเริ่มใหม่เอง

## วิธีนับ

- วัดการขยับของคาง (ขึ้นลง/ซ้ายขวา) และริมฝีปาก ในกรอบพิกัดที่ติดกับหัว จึงไม่นับตอนพยักหน้าหรือหันหน้า
- กรองความถี่ให้เหลือช่วงจังหวะเคี้ยว แล้วนับยอดคลื่น (`chew_counter.py`)
- ค่าตั้งต้นมาจาก `data/best_settings.json` ซึ่ง `tune.py` หาจากรอบทดลองจริง

## รันในเครื่อง (Windows)

```
conda create -n chewing python=3.11 -y
conda activate chewing
pip install -r requirements.txt
streamlit run app.py
```

หรือดับเบิลคลิก `run.bat` (ใช้ env ชื่อ `chewing`)

## ขึ้นเว็บด้วย Streamlit Community Cloud

1. ไปที่ https://share.streamlit.io แล้วเข้าสู่ระบบด้วย GitHub
2. Create app → เลือก repo นี้, branch `main`, ไฟล์ `app.py`
3. Advanced settings → **Python version 3.11** (mediapipe 0.10.21 ยังไม่รองรับ 3.13)
4. Deploy

ถ้ากด "เริ่มใช้งาน" แล้ววิดีโอค้าง (บางเครือข่ายต่อวิดีโอตรงไม่ได้) ให้ใส่ TURN server ใน
App settings → Secrets:

```toml
TURN_URLS = "turn:your-turn-server:3478,turns:your-turn-server:443"
TURN_USERNAME = "..."
TURN_CREDENTIAL = "..."
```

## จูนค่าด้วยข้อมูลของตัวเอง (ในเครื่อง)

1. เปิด sidebar → โหมดทดลอง → บันทึกจำนวนที่เคี้ยวจริง แล้วเคี้ยวรอบละ 10 ครั้ง 20–30 รอบ
   (และรอบนั่งนิ่ง จำนวนจริง = 0 อีก 8–10 รอบ)
2. ดับเบิลคลิก `tune.bat` (หรือ `python tune.py`) ผลจะอยู่ที่ `data/best_settings.json`

ข้อมูลรายเฟรม (`data/raw/`) ไม่ถูกส่งขึ้น GitHub (ดู `.gitignore`)
