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

### กล้องเปิดช้า / วิดีโอค้าง

ถ้ามีแค่ STUN เบราว์เซอร์กับเซิร์ฟเวอร์ต้องลองหาเส้นทางหลายแบบจนกว่าจะเจอ บางเครือข่าย (มือถือ, Wi-Fi องค์กร)
จะช้ามากหรือต่อไม่ติด ใส่ TURN server ใน App settings → Secrets แล้วจะต่อได้เร็วขึ้น (เลือกอย่างใดอย่างหนึ่ง):

1. **Hugging Face (ฟรี ง่ายที่สุด)** สมัครที่ huggingface.co → Settings → Access Tokens → สร้าง token แบบ Read

   ```toml
   HF_TOKEN = "hf_..."
   ```

2. **Cloudflare Realtime TURN** (ฟรีตามโควตา)

   ```toml
   CLOUDFLARE_TURN_KEY_ID = "..."
   CLOUDFLARE_TURN_KEY_API_TOKEN = "..."
   ```

3. **TURN ของตัวเอง**

   ```toml
   TURN_URLS = "turn:your-turn-server:3478,turns:your-turn-server:443"
   TURN_USERNAME = "..."
   TURN_CREDENTIAL = "..."
   ```

แอปบน Streamlit Cloud ที่ไม่มีคนใช้สักพักจะหลับ การเปิดครั้งแรกหลังจากนั้นจะช้ากว่าปกติ

## จูนค่าด้วยข้อมูลของตัวเอง (ในเครื่อง)

1. เปิด sidebar → โหมดทดลอง → บันทึกจำนวนที่เคี้ยวจริง แล้วเคี้ยวรอบละ 10 ครั้ง 20–30 รอบ
   (และรอบนั่งนิ่ง จำนวนจริง = 0 อีก 8–10 รอบ)
2. ดับเบิลคลิก `tune.bat` (หรือ `python tune.py`) ผลจะอยู่ที่ `data/best_settings.json`

ข้อมูลรายเฟรม (`data/raw/`) ไม่ถูกส่งขึ้น GitHub (ดู `.gitignore`)
