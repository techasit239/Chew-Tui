import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import math
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import json
import av
from PIL import Image, ImageDraw, ImageFont

from chew_counter import ChewCounter, SIGNALS, DEFAULT_SETTINGS

# จุด Landmark ของ MediaPipe Face Mesh ที่ใช้
CHIN = 152
FOREHEAD = 10
FACE_LEFT = 234
FACE_RIGHT = 454
UPPER_LIP_TOP = 0
LOWER_LIP_BOTTOM = 17
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
NOSE_TIP = 1
EYE_LEFT_OUTER = 33    # ตาที่อยู่ซ้ายของภาพ
EYE_RIGHT_OUTER = 263  # ตาที่อยู่ขวาของภาพ
SUBNASALE = 2          # ใต้จมูก (ไม่ขยับตามขากรรไกร)
# จุดที่ไม่ขยับตามขากรรไกร ใช้สร้างกรอบพิกัดของหัว ใช้หลายจุดแล้วเฉลี่ย กรอบจะนิ่งกว่าใช้จุดเดียว
EYE_LEFT_POINTS = [33, 133, 159, 145]     # หางตา หัวตา เปลือกตาบน-ล่าง (ตาซ้ายของภาพ)
EYE_RIGHT_POINTS = [263, 362, 386, 374]
# หน้าผาก (กะโหลก ไม่ขยับ และอยู่บนระนาบหน้า) ใช้หาทิศ "ลง" ของหัว
# ไม่ใช้จมูก เพราะจมูกยื่นออกมา ทำให้แกนลงเอียงเข้าหากล้องจนการขยับของคางเห็นน้อยลง
FOREHEAD_POINTS = [10, 151, 9]
SCALE_SMOOTHING = 0.1  # ระยะหางตาเปลี่ยนแค่ตอนเข้าใกล้/ถอยห่างกล้อง จึงเฉลี่ยได้มาก
# จุดแก้มและมุมกรามที่สมมาตรซ้าย–ขวา (ซ้ายของภาพ, ขวาของภาพ) ใช้หลายคู่แล้วเฉลี่ย จุดสั่นจะลดลง
# คู่ท้าย 2 คู่อยู่ที่มุมกราม ใกล้กล้ามเนื้อเคี้ยว (Masseter)
CHEEK_PAIRS = [(50, 280), (101, 330), (118, 347), (187, 411), (205, 425),
               (207, 427), (147, 376), (213, 433), (58, 288), (172, 397)]

# ชื่อสัญญาณบนวิดีโอ (ตัวสัญญาณและวิธีนับอยู่ใน chew_counter.py)
SIGNAL_LABELS = {"lips": "Lips", "chin_v": "Chin up/down", "chin_h": "Chin side", "cheeks": "Cheeks"}

# โมเดลใบหน้า 3 มิติแบบทั่วไป (มม.) ใช้กับ solvePnP เพื่อหามุมหน้าเทียบกับกล้อง
# แกนเดียวกับกล้อง: x ขวา, y ลง, z พุ่งออกจากกล้อง (ปลายจมูกอยู่ใกล้กล้องที่สุด)
PNP_LANDMARKS = [NOSE_TIP, CHIN, EYE_LEFT_OUTER, EYE_RIGHT_OUTER, MOUTH_LEFT, MOUTH_RIGHT]
FACE_MODEL = np.array([
    (0.0, 0.0, 0.0),
    (0.0, 330.0, 65.0),
    (-225.0, -170.0, 135.0),
    (225.0, -170.0, 135.0),
    (-150.0, 150.0, 125.0),
    (150.0, 150.0, 125.0),
], dtype=np.float64)

DATA_DIR = Path(__file__).resolve().parent / "data"  # โฟลเดอร์เก็บไฟล์ผลการทดลอง
RAW_DIR = DATA_DIR / "raw"                            # สัญญาณรายเฟรมของแต่ละรอบทดลอง (ให้ tune.py ใช้)
BEST_SETTINGS_FILE = DATA_DIR / "best_settings.json"  # ค่าที่ tune.py หาได้
# รันบน Streamlit Community Cloud (ผู้ใช้ของระบบชื่อ appuser) ไฟล์ที่เขียนจะอยู่บนเซิร์ฟเวอร์และหายเมื่อแอปรีสตาร์ต
IS_CLOUD = Path.home().name == "appuser"

CALIBRATION_SECONDS = 3.0
POSE_SMOOTHING = 0.3     # ค่าเฉลี่ยถ่วงน้ำหนักของมุมหน้า ยิ่งน้อยยิ่งนิ่งแต่ตอบสนองช้าลง
MAX_FRAME_GAP = 1.0      # วินาที กันเวลาหยุดนับพุ่งเมื่อเฟรมขาดหายนาน

ORANGE = (0, 165, 255)
BRIGHT_GREEN = (80, 255, 80)
GRAY = (70, 70, 70)
WHITE = (255, 255, 255)
RED = (60, 60, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# สีของหน้าจอหลัก (BGR) โทนเดียวกับหน้าเว็บ
ACCENT = (207, 108, 10)        # #0A6CCF
ACCENT_SOFT = (255, 155, 91)   # #5B9BFF
DONE_GREEN = (153, 211, 52)    # #34D399
AMBER = (36, 191, 251)         # #FBBF24
CORAL = (113, 113, 248)        # #F87171
INK = (245, 237, 232)          # #E8EDF5
MUTED = (176, 154, 142)        # #8E9AB0
GUIDE_COLORS = {"ok": DONE_GREEN, "warn": AMBER, "bad": CORAL, "pending": INK}

_FONTS = {}


# ฟอนต์ตัวเลขแบบเรียบ (Pillow มีฟอนต์ในตัว ไม่ต้องดาวน์โหลด) ถ้าใช้ไม่ได้จะกลับไปใช้ฟอนต์ของ OpenCV
def ui_font(size):
    if size not in _FONTS:
        try:
            _FONTS[size] = ImageFont.load_default(size=size)
        except Exception:
            _FONTS[size] = None
    return _FONTS[size]


def put_text_center(image, text, center, size, color):
    font = ui_font(size)
    if font is None:
        scale = size / 30
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
        cv2.putText(image, text, (int(center[0] - tw / 2), int(center[1] + th / 2)), FONT, scale, color, 2, cv2.LINE_AA)
        return
    x0, y0, x1, y1 = font.getbbox(text)
    left = int(center[0] - (x1 - x0) / 2) - 2
    top = int(center[1] - (y1 - y0) / 2) - 2
    h, w = image.shape[:2]
    rx0, ry0 = max(left, 0), max(top, 0)
    rx1, ry1 = min(left + x1 - x0 + 4, w), min(top + y1 - y0 + 4, h)
    if rx1 <= rx0 or ry1 <= ry0:
        return
    region = Image.fromarray(np.ascontiguousarray(image[ry0:ry1, rx0:rx1, ::-1]))
    ImageDraw.Draw(region).text((left - rx0 + 2 - x0, top - ry0 + 2 - y0), text, font=font,
                                fill=(color[2], color[1], color[0]))
    image[ry0:ry1, rx0:rx1] = np.asarray(region)[:, :, ::-1]

# ตัวช่วยจัดมุมกล้อง: ลำดับหัวข้อ, ข้อความบนวิดีโอ (OpenCV เขียนไทยไม่ได้) และคำแนะนำภาษาไทยบนหน้าเว็บ
SETUP_ORDER = ["distance", "framing", "facing", "height", "chin_view", "light", "stability"]
SETUP_COLORS = {"ok": BRIGHT_GREEN, "warn": ORANGE, "bad": RED, "pending": GRAY}
SETUP_TEXT = {
    ("distance", "ok"): "Distance OK",
    ("distance", "far"): "Move closer",
    ("distance", "close"): "Move back a little",
    ("framing", "ok"): "Framing OK",
    ("framing", "low"): "Tilt camera down",
    ("framing", "high"): "Tilt camera up",
    ("framing", "side"): "Center your face",
    ("facing", "ok"): "Camera in front",
    ("facing", "side"): "Camera is off to the side",
    ("height", "ok"): "Camera height OK",
    ("height", "high"): "Camera too high",
    ("height", "low"): "Camera too low",
    ("light", "ok"): "Light OK",
    ("light", "dark"): "Too dark",
    ("light", "bright"): "Too bright",
    ("light", "backlit"): "Light is behind you",
    ("chin_view", "ok"): "Chin & cheek motion visible",
    ("chin_view", "vertical"): "Chin up/down hard to see",
    ("chin_view", "side"): "Chin side/cheeks hard to see",
    ("stability", "pending"): "Tracking check after calibration",
    ("stability", "ok"): "Tracking steady",
    ("stability", "noisy"): "Tracking jittery",
}
SETUP_LABEL_TH = {
    "distance": "ระยะห่าง",
    "framing": "ตำแหน่งหน้าในภาพ",
    "facing": "กล้องอยู่ตรงหน้า",
    "height": "ความสูงกล้อง",
    "light": "แสง",
    "chin_view": "กล้องเห็นการขยับของคางและแก้ม",
    "stability": "ความนิ่งของจุดที่ใช้นับ",
}
SETUP_ADVICE_TH = {
    ("distance", "ok"): "ดีแล้ว",
    ("distance", "far"): "ไกลเกินไป ริมฝีปากในภาพเล็กจนจุดสั่นกลบการเคี้ยว ขยับเข้าใกล้กล้อง (ประมาณ 40–60 ซม.)",
    ("distance", "close"): "ใกล้เกินไป ก้มลงตักอาหารแล้วหน้าจะหลุดกรอบ ถอยออกเล็กน้อย",
    ("framing", "ok"): "ดีแล้ว",
    ("framing", "low"): "หน้าอยู่ต่ำในภาพ ก้มลงแล้วคางจะหลุดกรอบ เอียงกล้องลงเล็กน้อย",
    ("framing", "high"): "หน้าอยู่ชิดขอบบน เอียงกล้องขึ้นเล็กน้อย",
    ("framing", "side"): "หน้าไม่อยู่กลางภาพ เลื่อนกล้องหรือที่นั่งให้หน้าอยู่กลางจอ",
    ("facing", "ok"): "ดีแล้ว",
    ("facing", "side"): "กล้องอยู่เยื้องไปด้านข้าง ย้ายกล้องมาไว้ตรงหน้า (หลังจานอาหาร)",
    ("height", "ok"): "ดีแล้ว",
    ("height", "high"): "กล้องสูงกว่าหน้า มองลงมาจนริมฝีปากดูสั้น ลดกล้องลงมาให้อยู่ระดับคาง",
    ("height", "low"): "กล้องต่ำเกินไป เห็นใต้คางมากกว่าริมฝีปาก ยกกล้องขึ้นมาให้อยู่ระดับคาง",
    ("light", "ok"): "ดีแล้ว",
    ("light", "dark"): "หน้ามืดเกินไป เปิดไฟหรือหันหน้าเข้าหาหน้าต่าง",
    ("light", "bright"): "หน้าสว่างจ้าเกินไป หลบแสงแดดหรือไฟที่ส่องตรงหน้า",
    ("light", "backlit"): "แสงอยู่ด้านหลัง หน้าจึงมืด ย้ายให้หน้าต่างหรือไฟอยู่ด้านหน้าแทน",
    ("chin_view", "ok"): "ดีแล้ว คางขยับขึ้นลงและซ้ายขวาเห็นชัดในภาพ",
    ("chin_view", "vertical"): "กล้องมองคางเฉียงจากด้านบนหรือล่าง การขยับขึ้นลงของคางแทบไม่เห็นในภาพ "
                               "ต้องเดาจากความลึกซึ่งไม่แม่น ปรับกล้องให้อยู่ระดับคางและหันตรงเข้าหน้า",
    ("chin_view", "side"): "กล้องมองหน้าเฉียงจากด้านข้าง การขยับซ้ายขวาของคางและแก้มแทบไม่เห็นในภาพ "
                           "ย้ายกล้องมาไว้ตรงหน้า",
    ("stability", "pending"): "จะตรวจหลังคาลิเบรตเสร็จ",
    ("stability", "ok"): "ดีแล้ว จุดที่ใช้นับนิ่งพอ นับได้แม่น",
    ("stability", "noisy"): "จุดในวงเล็บสั่นมาก อาจนับเกิน ลองเข้าใกล้กล้อง เพิ่มแสง แล้วคาลิเบรตใหม่โดยนิ่งปากไว้ "
                            "หรือเพิ่มเกณฑ์/ปิดสัญญาณนั้นในแถบตั้งค่า",
}

# ตั้งค่าหน้าเพจ Streamlit (สีหลักอยู่ใน .streamlit/config.toml)
st.set_page_config(page_title="Mindful Chewing", page_icon="🍽️", layout="wide",
                   initial_sidebar_state="collapsed")

# หน้าตาแบบเรียบ มืด มีแสงฟ้านุ่มๆ ปุ่มทรงแคปซูล การ์ดโปร่งแสง (อ้างอิงสไตล์ vibrant.noomoagency.com)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Ubuntu+Mono:wght@700&family=Noto+Sans+Thai:wght@300;400;500;600&display=swap');
:root { --accent: #0A6CCF; --accent-soft: #5B9BFF; --ink: #E8EDF5; --muted: #8E9AB0;
        --glass: rgba(255,255,255,0.04); --line: rgba(255,255,255,0.09); }
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, label, p, li {
  font-family: 'Plus Jakarta Sans', 'Noto Sans Thai', sans-serif !important; }
.stApp {
  background:
    radial-gradient(55% 45% at 50% -5%, rgba(10,108,207,0.45), transparent 70%),
    radial-gradient(35% 35% at 90% 85%, rgba(91,155,255,0.12), transparent 70%),
    radial-gradient(30% 30% at 5% 70%, rgba(91,155,255,0.08), transparent 70%),
    #05070D;
  background-attachment: fixed; }
[data-testid="stHeader"] { background: transparent; }
/* กว้างพอให้วิดีโอใหญ่บนจอคอม แต่ไม่ยืดจนหน้าตาเต็มจอเกินไป */
.block-container { padding-top: 3.5rem; max-width: 1120px; padding-left: 1.5rem; padding-right: 1.5rem; }
.hero { text-align: center; margin: 0.5rem 0 1.8rem; animation: fadeUp .8s ease both; }
.eyebrow { font-family: 'Ubuntu Mono', monospace !important; text-transform: uppercase; letter-spacing: .22em;
  font-size: .72rem; color: var(--accent-soft); display: inline-block; padding: .35rem .9rem;
  border: 1px solid rgba(91,155,255,0.35); border-radius: 999px; background: rgba(10,108,207,0.12); }
.hero h1 { font-size: clamp(2.2rem, 6vw, 3.4rem); font-weight: 600; letter-spacing: -0.03em; line-height: 1.08;
  margin: 1rem 0 .6rem; color: var(--ink); }
.hero p { color: var(--muted); font-weight: 300; font-size: 1.02rem; margin: 0 auto; max-width: 520px; }
.steps { display: flex; gap: .6rem; justify-content: center; flex-wrap: wrap; margin: 1.4rem 0 0; }
.step { background: var(--glass); border: 1px solid var(--line); border-radius: 999px; padding: .5rem 1rem;
  font-size: .85rem; color: var(--ink); backdrop-filter: blur(10px); }
.step b { font-family: 'Ubuntu Mono', monospace !important; color: var(--accent-soft); margin-right: .4rem; }
.section-label { font-family: 'Ubuntu Mono', monospace !important; text-transform: uppercase; letter-spacing: .2em;
  font-size: .7rem; color: var(--muted); margin: 2rem 0 .6rem; }
/* กรอบวิดีโอ */
[data-testid="stCustomComponentV1"], iframe[title*="webrtc"] { border-radius: 28px !important; }
.element-container:has(iframe[title*="webrtc"]) { border-radius: 28px; overflow: hidden;
  box-shadow: 0 0 0 1px var(--line), 0 30px 80px -20px rgba(10,108,207,0.45); }
/* ปุ่มทรงแคปซูล มีวงแสงรอบนอก */
.stButton > button, .stDownloadButton > button {
  border-radius: 999px !important; border: 1px solid var(--line) !important; background: var(--glass) !important;
  color: var(--ink) !important; padding: .6rem 1.2rem !important; font-weight: 500 !important;
  transition: all .25s ease !important; backdrop-filter: blur(10px); }
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: rgba(91,155,255,0.6) !important; box-shadow: 0 0 0 6px rgba(10,108,207,0.15) !important;
  transform: translateY(-1px); }
.stButton > button[kind="primary"] { background: #F5F8FF !important; color: #0A6CCF !important; border: none !important;
  box-shadow: 0 0 0 6px rgba(255,255,255,0.08) !important; }
/* การ์ดโปร่งแสง */
[data-testid="stExpander"] { border: 1px solid var(--line) !important; border-radius: 22px !important;
  background: var(--glass); backdrop-filter: blur(12px); overflow: hidden; }
[data-testid="stExpander"] summary { padding: .9rem 1.1rem; }
[data-testid="stMetric"] { background: var(--glass); border: 1px solid var(--line); border-radius: 18px; padding: .8rem 1rem; }
[data-testid="stSidebar"] { background: rgba(10,13,22,0.92); border-right: 1px solid var(--line); }
[data-testid="stAlert"] { border-radius: 16px; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
</style>
""",
    unsafe_allow_html=True,
)

# mediapipe 1.x เอา mp.solutions ออกแล้ว มักเกิดจากรันด้วย Anaconda หลักแทน env chewing
if not hasattr(mp, "solutions"):
    st.error(
        f"mediapipe {mp.__version__} ไม่มี mp.solutions ต้องใช้รุ่น 0.10.21 ใน env chewing\n\n"
        f"ตอนนี้รันด้วย: `{sys.executable}`\n\n"
        "ปิดหน้าต่างนี้ แล้วดับเบิลคลิก run.bat ในโฟลเดอร์แอปแทน"
    )
    st.stop()
st.markdown(
    """
<div class="hero">
  <span class="eyebrow">Mindful Chewing</span>
  <h1>เคี้ยวช้าลง<br>อย่างมีสติ</h1>
  <p>กล้องนับการเคี้ยวให้ทุกคำ วงแหวนเต็มเมื่อเคี้ยวครบ แล้วค่อยกลืน</p>
  <div class="steps">
    <span class="step"><b>01</b>นั่งให้ตัวทับเงา</span>
    <span class="step"><b>02</b>กดเริ่มใช้งาน หุบปากนิ่ง 3 วินาที</span>
    <span class="step"><b>03</b>อ้าปากตักคำใหม่ ตัวนับเริ่มเอง</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

METHOD_LABELS = {"peaks": "กรองความถี่ + ยอดคลื่น", "threshold": "เกณฑ์ (แบบเดิม)"}


# ค่าของวิธีนับ (รูปแบบเดียวกับ chew_counter / best_settings.json) -> ค่าของแถบเลื่อนแต่ละตัว
def settings_to_widgets(s):
    s = {**DEFAULT_SETTINGS, **s}
    return {
        "w_method": METHOD_LABELS[s["method"]],
        "w_band": (float(s["peak_band"][0]), float(s["peak_band"][1])),
        "w_noise_k": float(s["peak_noise_k"]),
        "w_prominence": float(s["peak_prominence"]),
        "w_use_lips": "lips" in s["enabled"],
        "w_lips": (round(s["chew_low_pct"] * 100), round(s["chew_high_pct"] * 100)),
        "w_use_chin_v": "chin_v" in s["enabled"],
        "w_chin_v": round(s["thresholds"]["chin_v"] * 100, 2),
        "w_use_chin_h": "chin_h" in s["enabled"],
        "w_chin_h": round(s["thresholds"]["chin_h"] * 100, 2),
        "w_use_cheeks": "cheeks" in s["enabled"],
        "w_cheeks": round(s["thresholds"]["cheeks"] * 100, 2),
        "w_bite": round(s["bite_pct"] * 100),
    }


def load_tuned():
    if not BEST_SETTINGS_FILE.exists():
        return None
    try:
        return json.loads(BEST_SETTINGS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# เริ่มจากค่าที่จูนแล้วถ้ามี (คนที่เปิดเว็บแอปได้ค่าที่แม่นที่สุดทันที ไม่ต้องกดปุ่ม)
tuned = load_tuned()
for widget_key, widget_value in settings_to_widgets(tuned["settings"] if tuned else DEFAULT_SETTINGS).items():
    st.session_state.setdefault(widget_key, widget_value)

# แถบตั้งค่าด้านข้าง
with st.sidebar:
    st.header("⚙️ ตั้งค่า")

    # ค่าที่ tune.py หาได้จากรอบทดลองที่บันทึกไว้ (ใช้เป็นค่าเริ่มต้นอยู่แล้ว ปุ่มนี้ไว้ย้อนกลับหลังลองปรับเอง)
    if tuned:
        test = tuned.get("metrics", {}).get("best", {}).get("test")
        accuracy = (f"ชุดทดสอบคลาดเฉลี่ย {test['mae']} ครั้ง/รอบ, คลาดไม่เกิน ±1 {test['within1_pct']}%"
                    if test else "ยังไม่มีชุดทดสอบ (รอบทดลองน้อยกว่า 10 รอบ)")
        st.caption(f"ค่าที่จูนอัตโนมัติ ({tuned.get('created', '')}, {tuned.get('n_trials', '?')} รอบ): {accuracy}")
        if st.button("⚡ ใช้ค่าที่จูนอัตโนมัติ", use_container_width=True):
            st.session_state.update(settings_to_widgets(tuned["settings"]))
            st.toast("โหลดค่าที่จูนแล้ว กดคาลิเบรตใหม่ก่อนทดลอง")

    target_chews = st.slider("จำนวนครั้งที่ต้องเคี้ยวต่อคำ", 5, 50, 20)

    st.subheader("วิธีนับ")
    method_label = st.radio(
        "วิธีนับการเคี้ยว", list(METHOD_LABELS.values()), key="w_method",
        help="กรองความถี่: เก็บสัญญาณย้อนหลัง 4 วินาที ตัดการขยับที่ช้าหรือเร็วกว่าจังหวะเคี้ยวทิ้ง "
             "แล้วนับยอดคลื่น ตัวเลขจะขึ้นช้ากว่าจริงประมาณครึ่งวินาที | "
             "เกณฑ์: ขยับเกินเกณฑ์แล้วกลับมา = 1 ครั้ง",
    )
    method = "peaks" if method_label.startswith("กรอง") else "threshold"
    peak_band = st.slider(
        "ช่วงความถี่ที่เก็บไว้ (Hz)", 0.2, 4.0, step=0.1, key="w_band",
        disabled=method != "peaks",
        help="การเคี้ยวปกติอยู่ที่ 1–2 Hz (1–2 ครั้งต่อวินาที) ต่ำกว่าขอบล่าง = หน้าเลื่อนช้าๆ "
             "สูงกว่าขอบบน = จุดสั่นหรือกล้องสั่น",
    )
    peak_noise_k = st.slider(
        "ยอดคลื่นต้องสูงกว่าจุดสั่นกี่เท่า", 4.0, 12.0, step=0.5, key="w_noise_k",
        disabled=method != "peaks",
        help="วัดจุดสั่นจากตอนคาลิเบรต ค่าสูง = ไม่นับตอนนั่งนิ่ง แต่อาจพลาดการเคี้ยวเบาๆ "
             "ค่าต่ำ = จับการเคี้ยวเบาได้ แต่อาจนับตอนนิ่ง ลองนั่งนิ่ง 10 วินาทีดูว่าตัวเลขขึ้นไหม",
    )
    # ฟิลเตอร์ต้องมีช่วงความถี่กว้างพอ ไม่อย่างนั้นสร้างไม่ได้
    if peak_band[1] - peak_band[0] < 0.3:
        peak_band = (peak_band[0], peak_band[0] + 0.3)
    peak_prominence = st.slider(
        "ยอดคลื่นต้องสูงแค่ไหนถึงนับ (เท่าของเกณฑ์แต่ละสัญญาณ)", 0.1, 2.0, step=0.1, key="w_prominence",
        disabled=method != "peaks",
        help="ใช้เกณฑ์ของแต่ละสัญญาณด้านล่างเป็นหน่วย 0.5 = ขยับแค่ครึ่งของเกณฑ์ก็นับ "
             "ถ้านับไม่ขึ้นให้ลด ถ้านับตอนอยู่นิ่งให้เพิ่ม",
    )

    st.subheader("สัญญาณที่ใช้นับการเคี้ยว")
    st.caption("ส่วนใดส่วนหนึ่งขยับเกินเกณฑ์แล้วกลับมา = นับ 1 ครั้ง ดูแถบสัญญาณด้านขวาของวิดีโอประกอบการปรับ")
    move_help = ("หน่วยเป็น % ของระยะหางตาซ้าย–ขวา (ประมาณ 9 ซม. ดังนั้น 1% ≈ 1 มม.) "
                 "ค่าน้อย = ขยับนิดเดียวก็นับ ถ้านับเกินให้เพิ่มค่า ถ้านับไม่ขึ้นให้ลดค่า")
    use_lips = st.checkbox("ริมฝีปาก", key="w_use_lips")
    chew_low_pct, chew_high_pct = st.slider(
        "ช่วงที่ริมฝีปากขยับได้โดยยังไม่นับ (% ของค่าปกติ)", 70, 130, key="w_lips",
        disabled=not use_lips,
        help="ริมฝีปากอ้าเกินขอบบน หรือเม้มต่ำกว่าขอบล่าง แล้วกลับเข้ามาในช่วง = นับ 1 ครั้ง "
             "ถ้านับไม่ขึ้นให้บีบช่วงให้แคบลง (เข้าใกล้ 100) ถ้านับเกินให้ขยายช่วงออก",
    )
    # ขอบทั้งสองต้องอยู่คนละฝั่งของ 100% ไม่อย่างนั้นริมฝีปากนิ่งๆ ก็นับ
    chew_low_pct = min(chew_low_pct, 99)
    chew_high_pct = max(chew_high_pct, 101)
    use_chin_v = st.checkbox("คางขยับขึ้นลง", key="w_use_chin_v")
    chin_v_pct = st.slider("คางขึ้นลงเกินเท่าไรถึงนับ (%)", 0.5, 10.0, step=0.5, key="w_chin_v",
                           disabled=not use_chin_v, help=move_help)
    use_chin_h = st.checkbox("คางขยับซ้ายขวา", key="w_use_chin_h")
    chin_h_pct = st.slider("คางซ้ายขวาเกินเท่าไรถึงนับ (%)", 0.5, 10.0, step=0.5, key="w_chin_h",
                           disabled=not use_chin_h, help=move_help)
    use_cheeks = st.checkbox("แก้มขยับ", key="w_use_cheeks")
    cheeks_pct = st.slider("แก้มขยับเกินเท่าไรถึงนับ (%)", 0.5, 10.0, step=0.5, key="w_cheeks",
                           disabled=not use_cheeks, help=move_help)
    if not (use_lips or use_chin_v or use_chin_h or use_cheeks):
        st.warning("ไม่ได้เลือกสัญญาณใดเลย แอปจะใช้ริมฝีปากแทน")

    st.subheader("คำใหม่")
    bite_pct = st.slider(
        "อ้าปากกว้างแค่ไหนถึงนับเป็นคำใหม่ (% ของค่าปกติ)", 120, 250, key="w_bite",
        help="ถ้าตัวนับรีเซ็ตเองระหว่างเคี้ยวให้เพิ่มค่า (ข้อมูลทดลองจริง: ตอนเคี้ยวริมฝีปากอ้าถึงราว 125–150% "
             "ตอนตักอาหารราว 155–295% ค่ากลางคือ 215%)",
    )
    st.subheader("ช่วงที่ขยับหน้าได้")
    max_yaw_deg = st.slider(
        "หันซ้าย/ขวาได้ไม่เกิน (องศา)", 5, 30, 15,
        help="เกินค่านี้แอปจะหยุดนับ ถ้าหยุดบ่อยเกินไปให้เพิ่มค่า ถ้านับเกินตอนหันหน้าให้ลดค่า",
    )
    max_pitch_deg = st.slider(
        "ก้ม/เงยได้ไม่เกิน (องศา)", 5, 30, 12,
        help="เทียบกับท่าตอนคาลิเบรต ถ้าหยุดบ่อยเกินไปให้เพิ่มค่า ถ้านับขาดตอนก้มให้ลดค่า",
    )
    show_debug = st.checkbox(
        "แสดงข้อมูลวิเคราะห์บนวิดีโอ (สำหรับปรับค่า)", value=False,
        help="ปกติบนวิดีโอมีแค่ตัวนับกับเงาคนนั่ง เปิดเพื่อดูจุดบนหน้า รายการตรวจมุมกล้อง "
             "และแถบสัญญาณของแต่ละส่วน (แถบเกินเส้นส้ม = ส่วนนั้นขยับเกินเกณฑ์)",
    )

    st.subheader("🧪 โหมดทดลอง")
    if IS_CLOUD:
        st.caption("บนเว็บจะบันทึกสัญญาณรายเฟรมไว้บนเซิร์ฟเวอร์ ซึ่งหายเมื่อแอปรีสตาร์ต "
                   "ถ้าจะเก็บข้อมูลไปจูนด้วย tune.py ให้รันในเครื่องด้วย run.bat")
    record_actual = st.checkbox(
        "บันทึกจำนวนที่เคี้ยวจริงลงตาราง", value=False,
        help="ทุกแถวจะมีคอลัมน์ actual_chews (จำนวนจริง) และ error (นับได้ − จริง) ใช้เทียบความแม่นยำ "
             "และสัญญาณรายเฟรมของทุกรอบจะถูกบันทึกลง data/raw ให้ tune.py ใช้หาค่าที่ดีที่สุด (ไม่เก็บภาพ)",
    )
    actual_chews = st.number_input(
        "เคี้ยวจริงรอบละกี่ครั้ง", 0, 100, 10, disabled=not record_actual,
        help="ทุกครั้งที่แอปนับได้ วงแหวนจะกะพริบขาวพร้อม +1 ถ้าเคี้ยวแล้วไม่กะพริบ = แอปนับพลาด "
             "ใส่ 0 สำหรับรอบนั่งนิ่งหรือรอบพูดคุย (ช่วยให้ tune.py รู้ว่าไม่ควรนับตอนไหน)",
    )
    if record_actual:
        n_raw = len(list(RAW_DIR.glob("*.json"))) if RAW_DIR.exists() else 0
        st.caption(f"บันทึกสัญญาณแล้ว {n_raw} รอบ — ได้ 20–30 รอบแล้วดับเบิลคลิก tune.bat")


# เก็บข้อมูลรายคำของมื้อนี้ไว้ใน session_state ข้อมูลจึงไม่หายเมื่อกด STOP หรือคาลิเบรตใหม่
class MealLog:
    def __init__(self):
        self.lock = threading.Lock()
        self.rows = []

    def add(self, row):
        with self.lock:
            self.rows.append({"bite_no": len(self.rows) + 1, **row})

    def snapshot(self):
        with self.lock:
            return list(self.rows)

    def clear(self):
        with self.lock:
            self.rows = []


if "meal_log" not in st.session_state:
    st.session_state.meal_log = MealLog()
meal_log = st.session_state.meal_log


# คลาสประมวลผลวิดีโอแบบ Real-time (วิธีนับการเคี้ยวสืบทอดมาจาก ChewCounter ใน chew_counter.py)
class ChewProcessor(ChewCounter, VideoProcessorBase):
    def __init__(self, meal_log):
        ChewCounter.__init__(self)
        # โหลดโมเดล MediaPipe ไว้ใน Init เพื่อให้โหลดแค่ครั้งเดียว
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # recv() ทำงานคนละ Thread กับหน้าเว็บ จึงต้องล็อกก่อนแก้ค่าสถานะ
        self.lock = threading.Lock()
        self.meal_log = meal_log

        # ค่าที่ปรับได้จากหน้าเว็บ (ค่าของวิธีนับตั้งไว้ใน ChewCounter)
        self.target_chews = 20
        self.max_yaw_deg = 15
        self.max_pitch_deg = 12
        self.show_debug = False         # แสดงข้อมูลวิเคราะห์บนวิดีโอ (ปกติมีแค่ตัวนับกับเงาคนนั่ง)
        self.ring_display = 0.0         # ความคืบหน้าวงแหวนที่วาดจริง ค่อยๆ ไล่ตามค่าจริงให้ดูลื่น
        self.pulse_start = None         # เวลาที่เริ่มวงคลื่นกระจายรอบวงแหวน (ทุกครั้งที่นับได้)
        self.setup_report = None
        self.actual_chews = None        # โหมดทดลอง: จำนวนที่เคี้ยวจริงต่อรอบ
        self.flash_count = 0            # จำนวนล่าสุดที่วาดแล้ว ใช้รู้ว่าเพิ่งนับได้
        self.flash_until = 0.0
        self.head_axes_px = None        # แกนหัวที่ฉายลงภาพ ไว้วาดลูกศร
        self.scale_smooth = None

        # ข้อมูลของคำที่กำลังเคี้ยว
        self.bite_start = None
        self.paused_seconds = 0.0
        self.last_frame_time = None
        # โหมดทดลอง: สัญญาณรายเฟรมของรอบนี้ บันทึกลง data/raw ตอนจบรอบ ให้ tune.py ใช้หาค่าที่ดีที่สุด
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frames = None

        self.start_calibration()

    def start_calibration(self):
        with self.lock:
            self.finish_bite("recalibrated")
            self.baseline = None      # ค่าปกติของแต่ละสัญญาณ {ชื่อ: ค่า}
            self.signal_std = None    # ความสั่นตอนนิ่งของแต่ละสัญญาณ
            self.baseline_yaw = None
            self.baseline_pitch = None
            self.calib_start = None
            self.calib_signals = {key: [] for key in SIGNALS}
            self.calib_yaw = []
            self.calib_pitch = []
            self.yaw = None
            self.pitch = None
            self.sig_smooth = None
            self.last_dev = {}
            self.yaw_delta = 0.0
            self.pitch_delta = 0.0
            self.pose_ok = True
            self.reset_count()
            self.reset_bite_state()

    def reset_bite(self):
        with self.lock:
            self.finish_bite("manual_reset")
            self.reset_count()
            if self.baseline is not None:
                self.start_bite(time.time())

    # streamlit-webrtc เรียกเมื่อผู้ใช้กด STOP ให้บันทึกคำสุดท้ายก่อนปิด
    def on_ended(self):
        with self.lock:
            self.finish_bite("stopped")

    def start_bite(self, now, reason="manual"):
        self.bite_start = now
        self.paused_seconds = 0.0
        self.reset_count()
        self.frames = {key: [] for key in ["t", "counted", "skip", *SIGNALS]}
        # สถานะการจับคำใหม่ตอนเริ่มรอบ ให้ tune.py เล่นซ้ำจากจุดเดียวกับแอปได้เป๊ะ
        self.start_state = {
            "reason": reason,
            "is_biting": self.is_biting,
            "bite_frames": self.bite_frames,
            "close_frames": self.close_frames,
            "last_bite_age": None if self.last_bite_time is None else now - self.last_bite_time,
        }

    # เก็บสัญญาณดิบหนึ่งเฟรม (sig = None ถ้าหาหน้าไม่เจอ) พร้อมบอกว่าเฟรมนี้แอปนับหรือข้าม tune.py จะข้ามเฟรมเดียวกัน
    # skip = เหตุที่ข้าม: "bite" ช่วงตักอาหาร, "pose" หันเกินกำหนด, "face" หาหน้าไม่เจอ, None = นับ
    def record_frame(self, now, sig, skip):
        if self.frames is None or self.bite_start is None:
            return
        # ไม่ปัดเศษเวลา: ต่างกันแค่ 0.1 มิลลิวินาทีก็ทำให้จำนวนจุดในหน้าต่างกรองต่างไป จนเล่นซ้ำได้ไม่ตรงกับแอป
        self.frames["t"].append(now - self.bite_start)
        self.frames["counted"].append(skip is None)
        self.frames["skip"].append(skip)
        for key in SIGNALS:
            self.frames[key].append(None if sig is None else round(sig[key], 6))

    # โหมดทดลอง: บันทึกสัญญาณรายเฟรมของรอบนี้พร้อมจำนวนจริง ให้ tune.py ใช้
    def save_raw_trial(self, ended_by):
        if self.actual_chews is None or not self.frames or len(self.frames["t"]) < 30:
            return
        trial = {
            "format": 2,  # รุ่น 2: เก็บทุกเฟรม (อ้าซ้ำในคำเดิมไม่ลบข้อมูล) ให้ tune.py จูนเกณฑ์คำใหม่ได้
            "start_state": self.start_state,
            "session": self.session_id,
            "start_time": datetime.fromtimestamp(self.bite_start).strftime("%Y-%m-%d %H:%M:%S"),
            "actual_chews": self.actual_chews,
            "app_count": self.chew_count,
            "ended_by": ended_by,
            "settings": self.settings(),
            "baseline": self.baseline,
            "signal_std": self.signal_std,
            "frames": self.frames,
        }
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{self.session_id}_{datetime.fromtimestamp(self.bite_start):%H%M%S}.json"
        (RAW_DIR / name).write_text(json.dumps(trial), encoding="utf-8")

    # บันทึกคำที่เพิ่งจบลง MealLog (ต้องถือ self.lock อยู่ก่อนเรียก)
    def finish_bite(self, ended_by):
        if self.bite_start is None:
            return
        self.save_raw_trial(ended_by)
        # รอบนั่งนิ่ง (จริง = 0) ต้องบันทึกด้วย จะได้รู้ว่าแอปนับเกินตอนไม่ได้เคี้ยวไหม
        if self.chew_count == 0 and not self.actual_chews == 0:
            self.bite_start = None
            return
        # นับเวลาถึงการเคี้ยวครั้งสุดท้าย ไม่รวมช่วงรอตักคำถัดไป
        duration = (self.last_chew_time or self.bite_start) - self.bite_start
        self.meal_log.add({
            "start_time": datetime.fromtimestamp(self.bite_start).strftime("%Y-%m-%d %H:%M:%S"),
            "chews": self.chew_count,
            "target": self.target_chews,
            "reached_target": self.chew_count >= self.target_chews,
            "duration_sec": round(duration, 1),
            "chews_per_min": round(self.chew_count / duration * 60, 1) if duration > 0 else None,
            "paused_sec": round(self.paused_seconds, 1),
            "ended_by": ended_by,
            **({"actual_chews": self.actual_chews, "error": self.chew_count - self.actual_chews}
               if self.actual_chews is not None else {}),
        })
        self.bite_start = None

    def get_distance(self, p1, p2, image_shape):
        x1, y1 = p1.x * image_shape[1], p1.y * image_shape[0]
        x2, y2 = p2.x * image_shape[1], p2.y * image_shape[0]
        return math.dist((x1, y1), (x2, y2))

    def point_3d(self, p, image_shape):
        h, w = image_shape[:2]
        # z ของ MediaPipe อยู่ในสเกลเดียวกับ x จึงคูณด้วยความกว้างภาพ
        return np.array([p.x * w, p.y * h, p.z * w])

    # วัดทุกสัญญาณในกรอบพิกัดที่ติดกับหัว (แกนจากตาสองข้างกับหน้าผาก ซึ่งไม่ขยับตามขากรรไกร)
    # แล้วหารด้วยระยะหางตา ค่าจึงไม่เปลี่ยนเมื่อพยักหน้า หันหน้า หรือเข้าใกล้/ถอยห่างกล้อง
    # เหลือแค่การขยับของริมฝีปาก ขากรรไกร และแก้ม
    # แกนของหัว: x = ตาซ้ายไปขวา, y = จากหน้าผากลงมา (บนระนาบหน้า), z = ตั้งฉากกับหน้า
    # คืนค่า (จุดกำเนิด, เมทริกซ์แกน 3x3, ระยะหางตา) หรือ None ถ้าคำนวณไม่ได้
    def head_axes(self, lm, image_shape):
        def mean_point(indices):
            return np.mean([self.point_3d(lm[i], image_shape) for i in indices], axis=0)

        left, right = mean_point(EYE_LEFT_POINTS), mean_point(EYE_RIGHT_POINTS)
        origin = (left + right) / 2
        x_axis = right - left
        if np.linalg.norm(x_axis) == 0:
            return None
        x_axis = x_axis / np.linalg.norm(x_axis)
        # ขนาดหัวใช้ระยะหางตาสองข้างเหมือนเดิม (หน่วยของเกณฑ์ทุกตัวอิงค่านี้)
        scale = np.linalg.norm(self.point_3d(lm[EYE_RIGHT_OUTER], image_shape)
                               - self.point_3d(lm[EYE_LEFT_OUTER], image_shape))
        if scale == 0:
            return None
        down = origin - mean_point(FOREHEAD_POINTS)
        y_axis = down - x_axis * down.dot(x_axis)
        if np.linalg.norm(y_axis) == 0:
            return None
        y_axis = y_axis / np.linalg.norm(y_axis)
        return origin, np.array([x_axis, y_axis, np.cross(x_axis, y_axis)]), scale

    def measure_signals(self, lm, image_shape):
        def P(i):
            return self.point_3d(lm[i], image_shape)

        axes = self.head_axes(lm, image_shape)
        mouth_width = np.linalg.norm(P(MOUTH_RIGHT) - P(MOUTH_LEFT))
        if axes is None or mouth_width == 0:
            return None
        origin, R, scale = axes
        # จุดสั่นของระยะหางตาจะไปหารทุกสัญญาณ จึงเฉลี่ยก่อน (ไม่เฉลี่ยแกน เพราะจะตามการพยักหน้าไม่ทัน)
        if self.scale_smooth is None:
            self.scale_smooth = scale
        else:
            self.scale_smooth += SCALE_SMOOTHING * (scale - self.scale_smooth)
        scale = self.scale_smooth
        chin = R @ (P(CHIN) - origin) / scale
        return {
            # ระยะริมฝีปากบน–ล่างเทียบความกว้างปาก
            "lips": float(np.linalg.norm(P(LOWER_LIP_BOTTOM) - P(UPPER_LIP_TOP)) / mouth_width),
            "chin_v": float(chin[1]),  # + = คางลงต่ำ
            "chin_h": float(chin[0]),  # + = คางไปทางขวาของภาพ
            # ความกว้างแก้มเฉลี่ยจากหลายคู่ จุดเดี่ยวบนผิวเรียบสั่นง่าย เฉลี่ยแล้วนิ่งขึ้น
            "cheeks": float(np.mean([np.linalg.norm(P(r) - P(l)) for l, r in CHEEK_PAIRS]) / scale),
        }

    def to_pixel(self, p, image_shape):
        return int(p.x * image_shape[1]), int(p.y * image_shape[0])

    # ประมาณมุมหันซ้าย/ขวา (yaw) และก้ม/เงย (pitch) จากความลึก z ของ Landmark
    def get_head_angles(self, lm, image_shape):
        h, w = image_shape[:2]
        # MediaPipe ให้ค่า z ในสเกลเดียวกับ x (ความกว้างภาพ) จึงคูณด้วย w ทั้งคู่
        left, right = lm[FACE_LEFT], lm[FACE_RIGHT]
        yaw = math.degrees(math.atan2((right.z - left.z) * w, abs(right.x - left.x) * w))
        top, chin = lm[FOREHEAD], lm[CHIN]
        pitch = math.degrees(math.atan2((chin.z - top.z) * w, abs(chin.y - top.y) * h))
        return yaw, pitch

    def smooth_angles(self, yaw, pitch):
        if self.yaw is None:
            self.yaw, self.pitch = yaw, pitch
        else:
            self.yaw += POSE_SMOOTHING * (yaw - self.yaw)
            self.pitch += POSE_SMOOTHING * (pitch - self.pitch)

    # เก็บค่าตอนหุบปากปกติ 3 วินาทีแรก เพื่อใช้เป็น Baseline ส่วนตัว
    def calibrate(self, sig, yaw, pitch):
        now = time.time()
        if self.calib_start is None:
            self.calib_start = now
        for key in SIGNALS:
            self.calib_signals[key].append(sig[key])
        self.calib_yaw.append(yaw)
        self.calib_pitch.append(pitch)
        if now - self.calib_start >= CALIBRATION_SECONDS:
            # ใช้ค่ามัธยฐานเพื่อตัดเฟรมที่เผลอขยับปากหรือขยับหน้าออก
            self.baseline = {key: statistics.median(v) for key, v in self.calib_signals.items()}
            # ความสั่นตอนนิ่ง ต้องน้อยกว่าเกณฑ์นับเคี้ยวมากๆ (ริมฝีปากเก็บเป็นสัดส่วนของค่าปกติ)
            if len(self.calib_signals["lips"]) > 1:
                self.signal_std = {key: statistics.stdev(v) for key, v in self.calib_signals.items()}
                self.signal_std["lips"] /= self.baseline["lips"]
            self.baseline_yaw = statistics.median(self.calib_yaw)
            self.baseline_pitch = statistics.median(self.calib_pitch)
            self.start_bite(now, "calibrated")

    # หน้าต้องอยู่ใกล้ท่าตอนคาลิเบรต ไม่อย่างนั้นระยะที่วัดได้จะเพี้ยนจนนับผิด
    def check_pose(self):
        self.yaw_delta = self.yaw - self.baseline_yaw
        self.pitch_delta = self.pitch - self.baseline_pitch
        self.pose_ok = (abs(self.yaw_delta) <= self.max_yaw_deg
                        and abs(self.pitch_delta) <= self.max_pitch_deg)
        if not self.pose_ok:
            # ทิ้งจังหวะอ้าปากค้างไว้ กันไม่ให้นับผิดตอนหันหน้ากลับมา
            self.pause_count()
        return self.pose_ok

    # จับคำใหม่และนับการเคี้ยวด้วย ChewCounter.step (โค้ดเดียวกับที่ tune.py ใช้เล่นซ้ำ)
    # sig = สัญญาณที่เฉลี่ยแล้ว (ใช้จับคำใหม่และวิธีเกณฑ์), raw = สัญญาณดิบ (ให้ฟิลเตอร์จัดการเอง)
    def update_chew(self, sig, raw):
        now = time.time()
        if self.step(sig, raw, now) == "new_bite":
            # เฟรมที่ยืนยันคำใหม่เป็นเฟรมสุดท้ายของคำเก่า (step ถอนการนับตอนเริ่มอ้าปากที่เฟรมนี้)
            # tune.py จะได้เล่นซ้ำเจอเฟรมนี้และได้จำนวนตรงกับแอป
            self.record_frame(now, raw, "bite")
            self.frame_recorded = True
            # บันทึกคำเก่าแล้วเริ่มนับคำใหม่ (อ้าซ้ำในคำเดิม step จัดการล้างตัวนับเอง)
            self.finish_bite("new_bite")
            self.start_bite(now, "new_bite")

    # หามุมหน้าเทียบกับกล้องจริง (ไม่ใช่เทียบกับตอนคาลิเบรต) ใช้บอกว่ากล้องเยื้องหรือสูงต่ำไป
    def solve_head_pose(self, lm, image_shape):
        h, w = image_shape[:2]
        image_points = np.array([(lm[i].x * w, lm[i].y * h) for i in PNP_LANDMARKS], dtype=np.float64)
        camera = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(FACE_MODEL, image_points, camera, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        # yaw: + = หน้าหันไปทางซ้ายของภาพ, pitch: + = ก้มหน้า (หรือกล้องอยู่สูงกว่าหน้า)
        yaw = math.degrees(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])))
        pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        return yaw, pitch

    # ตรวจมุมกล้องแต่ละหัวข้อ คืนค่า {หัวข้อ: (สถานะ, รหัสคำแนะนำ, ค่าที่วัดได้)}
    def check_setup(self, lm, image_shape, gray):
        h, w = image_shape[:2]
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        report = {}

        # ระยะห่าง: ริมฝีปากในภาพต้องใหญ่พอให้การขยับ 5% ชนะจุดที่สั่น
        face_frac = x1 - x0
        mouth_px = self.get_distance(lm[MOUTH_LEFT], lm[MOUTH_RIGHT], image_shape)
        value = f"{face_frac * 100:.0f}%"
        if face_frac < 0.22 or mouth_px < 40:
            report["distance"] = ("bad", "far", value)
        elif face_frac > 0.55:
            report["distance"] = ("warn", "close", value)
        else:
            report["distance"] = ("ok", "ok", value)

        # ตำแหน่งในภาพ: เหลือที่ใต้คางไว้เผื่อก้มตักอาหาร
        center_x = (x0 + x1) / 2
        if y1 > 0.85:
            report["framing"] = ("warn", "low", "")
        elif y0 < 0.03:
            report["framing"] = ("warn", "high", "")
        elif not 0.3 <= center_x <= 0.7:
            report["framing"] = ("warn", "side", "")
        else:
            report["framing"] = ("ok", "ok", "")

        pose = self.solve_head_pose(lm, image_shape)
        if pose is not None:
            yaw, pitch = pose
            if abs(yaw) <= 10:
                report["facing"] = ("ok", "ok", f"{yaw:+.0f} deg")
            else:
                report["facing"] = ("warn" if abs(yaw) <= 20 else "bad", "side", f"{yaw:+.0f} deg")

            # กล้องต่ำกว่าหน้าเล็กน้อยดีที่สุด เพราะตอนกินเราก้มมองจานอยู่แล้ว
            if pitch > 10:
                report["height"] = ("warn" if pitch <= 20 else "bad", "high", f"{pitch:+.0f} deg")
            elif pitch < -20:
                report["height"] = ("warn" if pitch >= -30 else "bad", "low", f"{pitch:+.0f} deg")
            else:
                report["height"] = ("ok", "ok", f"{pitch:+.0f} deg")

        # แสง: วัดความสว่างเฉพาะบริเวณหน้า เทียบกับทั้งภาพเพื่อจับแสงย้อน
        bx0, bx1 = max(int(x0 * w), 0), min(int(x1 * w), w)
        by0, by1 = max(int(y0 * h), 0), min(int(y1 * h), h)
        if bx1 > bx0 and by1 > by0:
            face_mean = float(gray[by0:by1, bx0:bx1].mean())
            frame_mean = float(gray.mean())
            value = f"{face_mean:.0f}/255"
            if face_mean < 70:
                report["light"] = ("bad", "dark", value)
            elif face_mean > 210:
                report["light"] = ("warn", "bright", value)
            elif frame_mean - face_mean > 30:
                report["light"] = ("warn", "backlit", value)
            else:
                report["light"] = ("ok", "ok", value)

        # กล้องเห็นการขยับของคางไหม: แกนขึ้นลง/ซ้ายขวาของหัวต้องขนานกับภาพ ไม่ใช่พุ่งเข้าหากล้อง
        # ถ้ามองเฉียงมาก การขยับจะเหลือแต่ในแนวความลึก ซึ่ง MediaPipe เดาจากภาพเดียวได้ไม่แม่น
        axes = self.head_axes(lm, image_shape)
        if axes is not None:
            origin, R, scale = axes
            vertical = R[1]
            view_v = math.hypot(vertical[0], vertical[1])  # 1.0 = ขยับขึ้นลงเห็นเต็มที่ในภาพ
            view_h = math.hypot(R[0, 0], R[0, 1])
            worst, view = ("vertical", view_v) if view_v <= view_h else ("side", view_h)
            value = f"{view * 100:.0f}% visible"
            if view >= 0.9:
                report["chin_view"] = ("ok", "ok", value)
            else:
                report["chin_view"] = ("warn" if view >= 0.75 else "bad", worst, value)
            # ลูกศรแกนหัวสำหรับวาดบนหน้า (ยาวครึ่งระยะหางตา)
            self.head_axes_px = [(int(p[0]), int(p[1]))
                                 for p in (origin, origin + R[0] * scale * 0.5, origin + vertical * scale * 0.5)]

        # ความนิ่งของสัญญาณที่ใช้นับ วัดตอนคาลิเบรต ต้องน้อยกว่าเกณฑ์นับเคี้ยวหลายเท่า
        if self.signal_std is None:
            report["stability"] = ("pending", "pending", "")
        else:
            limits = dict(self.thresholds, lips=min(1 - self.chew_low_pct, self.chew_high_pct - 1))
            noise = {key: self.signal_std[key] / limits[key] for key in SIGNALS if key in self.enabled}
            noise = noise or {"lips": self.signal_std["lips"] / limits["lips"]}
            worst = max(noise, key=noise.get)
            # แสดงเป็น % ของเกณฑ์ เช่น 20% = จุดสั่นขนาด 1 ใน 5 ของเกณฑ์นับ
            value = f"{SIGNAL_LABELS[worst]} {noise[worst] * 100:.0f}% of threshold"
            if noise[worst] <= 1 / 3:
                report["stability"] = ("ok", "ok", value)
            elif noise[worst] <= 1 / 1.5:
                report["stability"] = ("warn", "noisy", value)
            else:
                report["stability"] = ("bad", "noisy", value)

        return report

    # เงาคนนั่ง (หัว คอ ไหล่) ตำแหน่งและขนาดที่ควรเห็นในภาพ ขยับกล้อง/ที่นั่งให้ตัวจริงทับเงานี้
    # หัวสูงกว่ากลางภาพเล็กน้อย เหลือที่ใต้คางไว้เผื่อก้มตักอาหาร สีขอบบอกผลตรวจมุมกล้องรวม
    def silhouette_geometry(self, h, w):
        center = (w // 2, int(h * 0.45))
        ax, ay = int(w * 0.19), int(h * 0.33)
        return center, ax, ay, center[1] + int(ay * 0.88), int(h * 0.88)

    def draw_silhouette(self, image):
        h, w = image.shape[:2]
        statuses = [v[0] for v in (self.setup_report or {}).values()]
        if "bad" in statuses:
            color = GUIDE_COLORS["bad"]
        elif "warn" in statuses:
            color = GUIDE_COLORS["warn"]
        elif statuses:
            color = GUIDE_COLORS["ok"]
        else:
            color = GUIDE_COLORS["pending"]
        center, ax, ay, neck_top, shoulder_y = self.silhouette_geometry(h, w)
        cx = center[0]
        neck = np.array([(cx - ax * 0.42, neck_top), (cx + ax * 0.42, neck_top),
                         (cx + ax * 0.5, shoulder_y), (cx - ax * 0.5, shoulder_y)], np.int32)
        shoulders = np.array([(cx - ax * 0.5, shoulder_y), (cx + ax * 0.5, shoulder_y),
                              (cx + ax * 2.0, h * 0.95), (cx + ax * 2.4, h), (cx - ax * 2.4, h),
                              (cx - ax * 2.0, h * 0.95)], np.int32)
        shadow = image.copy()
        cv2.ellipse(shadow, center, (ax, ay), 0, 0, 360, (18, 14, 10), -1, cv2.LINE_AA)
        cv2.fillPoly(shadow, [neck, shoulders], (18, 14, 10), cv2.LINE_AA)
        cv2.addWeighted(shadow, 0.28, image, 0.72, 0, dst=image)
        lines = image.copy()
        cv2.ellipse(lines, center, (ax, ay), 0, 0, 360, color, 2, cv2.LINE_AA)
        for side in (-1, 1):  # เส้นขอบคอและไหล่ซ้าย/ขวา
            outline = np.array([(cx + side * ax * 0.42, neck_top), (cx + side * ax * 0.5, shoulder_y),
                                (cx + side * ax * 2.0, h * 0.95), (cx + side * ax * 2.4, h)], np.int32)
            cv2.polylines(lines, [outline], False, color, 2, cv2.LINE_AA)
        cv2.addWeighted(lines, 0.75, image, 0.25, 0, dst=image)

    # ข้อมูลวิเคราะห์เพิ่ม (โหมดปรับค่า): เส้นตาราง 3x3 จุดกึ่งกลาง และลูกศรแกนหัว
    def draw_debug_guide(self, image):
        h, w = image.shape[:2]
        for i in (1, 2):
            cv2.line(image, (w * i // 3, 0), (w * i // 3, h), GRAY, 1)
            cv2.line(image, (0, h * i // 3), (w, h * i // 3), GRAY, 1)
        center, ax, ay, _, _ = self.silhouette_geometry(h, w)
        cv2.drawMarker(image, center, WHITE, cv2.MARKER_CROSS, 16, 1)
        cv2.putText(image, "Fit your head & shoulders here", (center[0] - 130, max(20, center[1] - ay - 8)),
                    FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        # ลูกศรแกนหัว: ยิ่งสั้นยิ่งแปลว่ากล้องมองเฉียง เห็นการขยับในแนวนั้นน้อย
        if self.head_axes_px:
            origin, x_end, y_end = self.head_axes_px
            view = (self.setup_report or {}).get("chin_view", ("ok",))[0]
            arrow_color = SETUP_COLORS.get(view, WHITE)
            cv2.arrowedLine(image, origin, x_end, arrow_color, 2, cv2.LINE_AA, tipLength=0.25)
            cv2.arrowedLine(image, origin, y_end, arrow_color, 2, cv2.LINE_AA, tipLength=0.25)

    def draw_setup(self, image):
        if not self.setup_report:
            return
        y = 160
        cv2.putText(image, "Camera check", (30, y), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
        for key in SETUP_ORDER:
            if key not in self.setup_report:
                continue
            y += 22
            status, hint, value = self.setup_report[key]
            color = SETUP_COLORS[status]
            cv2.circle(image, (38, y - 5), 6, color, -1, cv2.LINE_AA)
            text = SETUP_TEXT[(key, hint)] + (f" ({value})" if value and status != "ok" else "")
            cv2.putText(image, text, (52, y), FONT, 0.5, WHITE if status == "ok" else color, 1, cv2.LINE_AA)

    # แถบสัญญาณแต่ละส่วนใต้วงแหวน แถบเกินเส้นส้ม = ขยับเกินเกณฑ์
    def draw_meters(self, image):
        if not self.last_dev:
            return
        h, w = image.shape[:2]
        x = w - 200
        y = self.ring_geometry(h, w)[1] * 2 + 80
        bar_w = 80
        for key in SIGNALS:
            enabled = key in self.enabled
            dev = self.last_dev.get(key, 0.0)
            color = (BRIGHT_GREEN if dev > 1 else WHITE) if enabled else GRAY
            cv2.putText(image, SIGNAL_LABELS[key], (x, y), FONT, 0.45, color, 1, cv2.LINE_AA)
            bx = x + 105
            cv2.rectangle(image, (bx, y - 10), (bx + bar_w, y), GRAY, 1)
            fill = int(bar_w * min(dev, 1.5) / 1.5)
            if fill > 0:
                cv2.rectangle(image, (bx, y - 10), (bx + fill, y), color, -1)
            threshold_x = bx + int(bar_w / 1.5)
            cv2.line(image, (threshold_x, y - 13), (threshold_x, y + 3), ORANGE, 1)
            y += 20
        if self.method == "peaks":
            source = SIGNAL_LABELS.get(self.peak_source, "waiting")
            cv2.putText(image, f"Peaks from: {source}", (x, y), FONT, 0.45, WHITE, 1, cv2.LINE_AA)

    # วงแหวนความคืบหน้า มองผ่านหางตาก็รู้ว่าเคี้ยวไปถึงไหนแล้ว
    # จานโปร่งสีเข้ม + เส้นราง + เส้นความคืบหน้าปลายมน ความคืบหน้าค่อยๆ ไหลไปหาค่าจริงให้ดูลื่น
    def ring_geometry(self, h, w):
        radius = max(34, min(h, w) // 9)
        return (w - radius - 26, radius + 26), radius, max(5, radius // 7)

    def draw_ring(self, image, target, label, color, sublabel="", dim=False, done=False):
        h, w = image.shape[:2]
        center, radius, thickness = self.ring_geometry(h, w)
        self.ring_display += 0.25 * (min(max(target, 0.0), 1.0) - self.ring_display)
        progress = self.ring_display

        layer = image.copy()
        cv2.circle(layer, center, radius + thickness + 6, (16, 12, 8), -1, cv2.LINE_AA)
        cv2.circle(layer, center, radius, (70, 62, 56), thickness, cv2.LINE_AA)
        if progress > 0.005:
            # เริ่มจากด้านบน (-90 องศา) วนตามเข็มนาฬิกา ปลายทั้งสองข้างมน
            end = -90 + 360 * progress
            cv2.ellipse(layer, center, (radius, radius), -90, 0, 360 * progress, color, thickness, cv2.LINE_AA)
            for angle in (-90, end):
                p = (int(center[0] + radius * math.cos(math.radians(angle))),
                     int(center[1] + radius * math.sin(math.radians(angle))))
                cv2.circle(layer, p, thickness // 2, color, -1, cv2.LINE_AA)
        if done:  # เคี้ยวครบ: แสงเขียวนุ่มๆ รอบวง
            cv2.circle(layer, center, radius + thickness + 3, color, 2, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.45 if dim else 0.88, image, 0.55 if dim else 0.12, 0, dst=image)

        # วงคลื่นกระจายออกทุกครั้งที่นับได้ (แทนตัวอักษร +1) ตอนทดลองจะรู้ทันทีว่าครั้งไหนแอปนับพลาด
        now = time.time()
        if self.chew_count > self.flash_count:
            self.pulse_start = now
        self.flash_count = self.chew_count
        if self.pulse_start is not None and now - self.pulse_start < 0.5:
            t = (now - self.pulse_start) / 0.5
            pulse = image.copy()
            cv2.circle(pulse, center, int(radius + thickness + 4 + 22 * t), ACCENT_SOFT, 2, cv2.LINE_AA)
            cv2.addWeighted(pulse, 1 - t, image, t, 0, dst=image)

        put_text_center(image, label, (center[0], center[1] - (radius // 8 if sublabel else 0)),
                        int(radius * 0.78), MUTED if dim else INK)
        if sublabel:
            put_text_center(image, sublabel, (center[0], center[1] + int(radius * 0.5)),
                            max(11, radius // 4), MUTED)

    def draw_overlay(self, image, face_found):
        self.draw_silhouette(image)
        if self.show_debug:
            self.draw_debug_guide(image)
            self.draw_setup(image)

        if self.baseline is None:
            # คาลิเบรต: วงแหวนนับถอยหลัง 3-2-1
            if face_found and self.calib_start is not None:
                elapsed = time.time() - self.calib_start
                self.draw_ring(image, elapsed / CALIBRATION_SECONDS,
                               str(max(1, math.ceil(CALIBRATION_SECONDS - elapsed))), ACCENT_SOFT, "sec")
            else:
                self.draw_ring(image, 0.0, "-", ACCENT_SOFT, dim=True)
            return

        done = self.chew_count >= self.target_chews
        self.draw_ring(image, self.chew_count / max(self.target_chews, 1), str(self.chew_count),
                       DONE_GREEN if done else ACCENT_SOFT, f"/ {self.target_chews}",
                       dim=not face_found or not self.pose_ok, done=done)
        if not self.show_debug:
            return

        self.draw_meters(image)
        bottom = image.shape[0] - 30
        if not face_found:
            cv2.putText(image, "No face detected", (30, bottom), FONT, 0.8, ORANGE, 2, cv2.LINE_AA)
            return
        if not self.pose_ok:
            cv2.putText(image, "PAUSED: face the camera", (30, 120), FONT, 1.0, ORANGE, 2, cv2.LINE_AA)
        elif self.is_biting:
            cv2.putText(image, "New bite", (30, 120), FONT, 1.0, WHITE, 2, cv2.LINE_AA)
        # แสดงมุมที่ขยับไปจากท่าตอนคาลิเบรต ช่วยใช้ปรับค่าในแถบตั้งค่า
        pose_color = WHITE if self.pose_ok else ORANGE
        cv2.putText(image, f"Turn {self.yaw_delta:+.0f}  Tilt {self.pitch_delta:+.0f} deg",
                    (30, bottom), FONT, 0.6, pose_color, 1, cv2.LINE_AA)

    # ฟังก์ชัน recv จะถูกเรียกใช้ทุกครั้งที่มีเฟรมภาพใหม่ส่งมาจากเบราว์เซอร์
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # แปลงข้อมูล WebRTC Frame เป็น Numpy Array (OpenCV format)
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # ใช้วัดแสง ต้องทำก่อนวาดอะไรทับภาพ

        # ประมวลผลด้วย MediaPipe
        results = self.face_mesh.process(image_rgb)
        face_found = results.multi_face_landmarks is not None

        with self.lock:
            now = time.time()
            frame_gap = min(now - self.last_frame_time, MAX_FRAME_GAP) if self.last_frame_time else 0.0
            self.last_frame_time = now

            self.skip_reason = "face"  # เหตุที่เฟรมนี้ไม่ได้นับ (None = นับ) update_chew จะเปลี่ยนให้
            self.frame_recorded = False
            sig = None
            if face_found:
                lm = results.multi_face_landmarks[0].landmark
                self.setup_report = self.check_setup(lm, image.shape, gray)
                sig = self.measure_signals(lm, image.shape)

                if sig is not None:
                    self.smooth_signals(sig)
                    self.smooth_angles(*self.get_head_angles(lm, image.shape))

                    if self.baseline is None:
                        self.calibrate(sig, self.yaw, self.pitch)
                    elif self.check_pose():
                        self.update_chew(self.sig_smooth, sig)
                    else:
                        self.skip_reason = "pose"

                # จุดมาร์คเกอร์ที่ริมฝีปาก คาง และแก้ม (เฉพาะโหมดแสดงข้อมูลวิเคราะห์)
                if self.show_debug:
                    upper = self.to_pixel(lm[UPPER_LIP_TOP], image.shape)
                    lower = self.to_pixel(lm[LOWER_LIP_BOTTOM], image.shape)
                    cv2.line(image, upper, lower, (255, 0, 0), 2)
                    for i in (UPPER_LIP_TOP, LOWER_LIP_BOTTOM, CHIN):
                        cv2.circle(image, self.to_pixel(lm[i], image.shape), 4, (0, 255, 0), -1)
                    for pair in CHEEK_PAIRS:
                        for i in pair:
                            cv2.circle(image, self.to_pixel(lm[i], image.shape), 2, (0, 255, 0), -1)
            else:
                # หน้าหลุดจากกล้อง เริ่มเฉลี่ยมุมและสัญญาณใหม่เมื่อเจอหน้าอีกครั้ง
                self.setup_report = None
                self.head_axes_px = None
                self.scale_smooth = None
                self.yaw = None
                self.pitch = None
                self.sig_smooth = None
                self.last_dev = {}
                self.peak_buffer.clear()
                self.is_jaw_open = False
                self.bite_frames = 0
                if self.baseline is None:
                    # หน้าหลุดระหว่างคาลิเบรต ให้เริ่มจับเวลาใหม่
                    self.calib_start = None
                    self.calib_signals = {key: [] for key in SIGNALS}
                    self.calib_yaw = []
                    self.calib_pitch = []

            if self.baseline is not None and not self.frame_recorded:
                self.record_frame(now, sig, self.skip_reason)

            # รวมเวลาที่ไม่ได้นับ (หน้าหลุดหรือหันเกินกำหนด) ไว้ใช้คัดคำที่ข้อมูลไม่น่าเชื่อถือ
            if self.bite_start is not None and (not face_found or not self.pose_ok):
                self.paused_seconds += frame_gap

            self.draw_overlay(image, face_found)

        # ส่งภาพที่วาด UI แล้วกลับไปแสดงผลที่เบราว์เซอร์
        return av.VideoFrame.from_ndarray(image, format="bgr24")


# เส้นทางเชื่อมวิดีโอระหว่างเบราว์เซอร์กับเซิร์ฟเวอร์ บนเว็บถ้ามีแค่ STUN การเชื่อมต่ออาจช้ามากหรือไม่ติด
# (ต้องลองหลายเส้นทางจนหมดเวลา) ถ้ามี TURN จะต่อได้เร็วและแน่นอนกว่า
# - ค่าตั้งต้น (คืน None): ให้ streamlit-webrtc หา TURN ฟรีเองจาก Secrets/ตัวแปรระบบ เช่น HF_TOKEN (Hugging Face)
#   หรือ CLOUDFLARE_TURN_KEY_ID + CLOUDFLARE_TURN_KEY_API_TOKEN ถ้าไม่มีจะใช้ STUN ของ Google
# - หรือใส่ TURN ของตัวเองใน Secrets: TURN_URLS (คั่นด้วยจุลภาค), TURN_USERNAME, TURN_CREDENTIAL
def rtc_configuration():
    try:
        turn = st.secrets.get("TURN_URLS")
    except Exception:  # ไม่มีไฟล์ secrets ตอนรันในเครื่อง
        turn = None
    if not turn:
        return None
    return {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": [u.strip() for u in turn.split(",")],
         "username": st.secrets.get("TURN_USERNAME", ""),
         "credential": st.secrets.get("TURN_CREDENTIAL", "")},
    ]}


# สร้าง WebRTC Streamer บนหน้าเว็บ
ctx = webrtc_streamer(
    key="chew-counter",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: ChewProcessor(meal_log),
    # ขอภาพ 640x480 ราว 24 เฟรม/วินาที พอสำหรับจับหน้า กล้องมือถือบางรุ่นส่ง 720p ขึ้นไป ทำให้เข้ารหัส
    # ส่งขึ้นเซิร์ฟเวอร์ และประมวลผลช้า (บนเว็บ CPU ของเซิร์ฟเวอร์ฟรีมีน้อย)
    media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                                        "frameRate": {"ideal": 24, "max": 30}},
                              "audio": False},
    rtc_configuration=rtc_configuration(),
    video_html_attrs={"style": {"width": "100%", "borderRadius": "24px"}, "autoPlay": True,
                      "controls": False, "muted": True},
    translations={"start": "เริ่มใช้งาน", "stop": "หยุด", "select_device": "เลือกกล้อง",
                  "media_api_not_available": "เบราว์เซอร์นี้ใช้กล้องไม่ได้",
                  "device_ask_permission": "กรุณาอนุญาตให้ใช้กล้อง",
                  "device_not_available": "ไม่พบกล้อง", "device_access_denied": "ไม่ได้รับอนุญาตให้ใช้กล้อง"},
    async_processing=True
)

if ctx.state.signalling and not ctx.state.playing:
    st.caption("กำลังเชื่อมต่อวิดีโอกับเซิร์ฟเวอร์... ครั้งแรกอาจใช้เวลาสักครู่")

# ส่งค่าตั้งค่าและคำสั่งปุ่มข้าม Thread ไปยัง VideoProcessor
if ctx.video_processor:
    processor = ctx.video_processor
    with processor.lock:
        processor.target_chews = target_chews
        processor.chew_low_pct = chew_low_pct / 100
        processor.chew_high_pct = chew_high_pct / 100
        processor.thresholds = {"chin_v": chin_v_pct / 100, "chin_h": chin_h_pct / 100, "cheeks": cheeks_pct / 100}
        processor.enabled = {key for key, on in [("lips", use_lips), ("chin_v", use_chin_v),
                                                 ("chin_h", use_chin_h), ("cheeks", use_cheeks)] if on}
        processor.show_debug = show_debug
        if processor.method != method:
            processor.peak_buffer.clear()
            processor.is_jaw_open = False
        processor.method = method
        processor.peak_band = tuple(peak_band)
        processor.peak_prominence = peak_prominence
        processor.peak_noise_k = peak_noise_k
        processor.actual_chews = int(actual_chews) if record_actual else None
        # เกณฑ์คำใหม่ต้องสูงกว่าขอบบนของการเคี้ยวชัดเจน ไม่อย่างนั้นเคี้ยวแรงๆ จะกลายเป็นรีเซ็ต
        processor.bite_pct = max(bite_pct, chew_high_pct + 10) / 100
        processor.max_yaw_deg = max_yaw_deg
        processor.max_pitch_deg = max_pitch_deg
        setup_report = dict(processor.setup_report or {})

    col1, col2 = st.columns(2)
    with col1:
        if st.button("เริ่มคำใหม่", use_container_width=True, type="primary"):
            processor.reset_bite()
    with col2:
        if st.button("คาลิเบรตใหม่", use_container_width=True):
            processor.start_calibration()
else:
    setup_report = None

st.markdown('<div class="section-label">Details</div>', unsafe_allow_html=True)

# ผลตรวจมุมกล้องเป็นภาษาไทย (ภาพนิ่ง ณ ตอนกดปุ่ม) พร้อมวิธีตั้งกล้อง
with st.expander("📷  มุมกล้อง"):
    st.markdown(
        """
- **วางกล้องตรงหน้า หลังจานอาหาร** สูงประมาณระดับคาง ตอนก้มมองจาน หน้าจะหันเข้าหากล้องพอดี
- **ระยะ 40–60 ซม.** ขยับจนหัวและไหล่ทับ **เงาคนนั่ง** บนวิดีโอ (สีขอบเงา: เขียว = ดี, เหลือง = พอใช้, แดง = ควรแก้)
- **ให้แสงอยู่ด้านหน้า** อย่านั่งหันหลังให้หน้าต่าง และ**ล็อกกล้องให้นิ่ง** ถ้ากล้องขยับต้องคาลิเบรตใหม่
"""
    )
    if setup_report is not None:
        st.button("ตรวจมุมกล้องตอนนี้", use_container_width=True)  # กดแล้วหน้าเว็บรันใหม่และอ่านผลล่าสุด
        if not setup_report:
            st.info("ยังไม่เห็นหน้า มองกล้องแล้วกด 'ตรวจมุมกล้องตอนนี้'")
        else:
            show = {"ok": st.success, "warn": st.warning, "bad": st.error, "pending": st.info}
            for key in SETUP_ORDER:
                if key in setup_report:
                    status, hint, value = setup_report[key]
                    detail = f" ({value})" if value else ""
                    show[status](f"**{SETUP_LABEL_TH[key]}**{detail}: {SETUP_ADVICE_TH[(key, hint)]}")
    else:
        st.caption("กด START ก่อน แล้วจะตรวจมุมกล้องได้")

# สรุปข้อมูลมื้อนี้และดาวน์โหลดเป็น CSV
meal_panel = st.expander("📊  ข้อมูลมื้อนี้")
meal_panel.caption("คำที่กำลังเคี้ยวจะถูกบันทึกเมื่อเริ่มคำใหม่ กดเริ่มคำใหม่ หรือกด STOP "
                   "ตารางไม่อัปเดตเอง กด 'อัปเดตข้อมูล' เพื่อดูข้อมูลล่าสุด")
col1, col2 = meal_panel.columns(2)
with col1:
    st.button("อัปเดตข้อมูล", use_container_width=True)  # กดแล้วหน้าเว็บจะรันใหม่และดึงข้อมูลล่าสุด
with col2:
    if st.button("เริ่มมื้อใหม่ (ล้างข้อมูล)", use_container_width=True):
        meal_log.clear()

# ชื่อไฟล์บอกสัญญาณและเกณฑ์ที่เปิดใช้ตอนบันทึก เช่น lips96-104_chinV3_chinH2_cheeks1.5
feature_parts = []
if use_lips or not (use_chin_v or use_chin_h or use_cheeks):
    feature_parts.append(f"lips{chew_low_pct}-{chew_high_pct}")
if use_chin_v:
    feature_parts.append(f"chinV{chin_v_pct:g}")
if use_chin_h:
    feature_parts.append(f"chinH{chin_h_pct:g}")
if use_cheeks:
    feature_parts.append(f"cheeks{cheeks_pct:g}")
if method == "peaks":
    method_tag = f"peak{peak_band[0]:g}-{peak_band[1]:g}Hz-p{peak_prominence:g}-n{peak_noise_k:g}"
else:
    method_tag = "thr"
file_name = f"chewing_{datetime.now():%Y%m%d_%H%M%S}_{method_tag}_{'_'.join(feature_parts)}.csv"

rows = meal_log.snapshot()
with meal_panel:
    if rows:
        df = pd.DataFrame(rows)
        m1, m2, m3 = st.columns(3)
        m1.metric("จำนวนคำ", len(df))
        m2.metric("เคี้ยวเฉลี่ยต่อคำ", f"{df['chews'].mean():.1f} ครั้ง")
        m3.metric("คำที่เคี้ยวครบเป้า", f"{df['reached_target'].mean() * 100:.0f}%")
        if "error" in df and df["error"].notna().any():
            trials = df[df["error"].notna()]
            a1, a2, a3 = st.columns(3)
            a1.metric("คลาดเฉลี่ยต่อรอบ", f"{trials['error'].abs().mean():.1f} ครั้ง")
            a2.metric("นับตรงเป๊ะ", f"{(trials['error'] == 0).mean() * 100:.0f}%")
            a3.metric("คลาดไม่เกิน ±1", f"{(trials['error'].abs() <= 1).mean() * 100:.0f}%")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"ชื่อไฟล์: `{file_name}` ถ้าเปลี่ยนการตั้งค่า ให้บันทึกแล้วกดเริ่มมื้อใหม่ก่อน "
                   "ไฟล์หนึ่งจะได้มีแต่ผลของการตั้งค่าเดียว")
        # บนเว็บไฟล์จะไปอยู่ที่เซิร์ฟเวอร์ ไม่ใช่เครื่องผู้ใช้ จึงให้บันทึกลงโฟลเดอร์ได้เฉพาะตอนรันในเครื่อง
        if not IS_CLOUD and st.button("💾 บันทึกลงโฟลเดอร์ data", use_container_width=True):
            DATA_DIR.mkdir(exist_ok=True)
            # utf-8-sig ทำให้ Excel อ่านภาษาไทยได้ถูกต้อง
            df.to_csv(DATA_DIR / file_name, index=False, encoding="utf-8-sig")
            st.success(f"บันทึกแล้ว: {DATA_DIR / file_name}")
        st.download_button(
            "ดาวน์โหลด CSV (เปิดด้วย Excel ได้)",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name=file_name,
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("ยังไม่มีข้อมูล เริ่มกินแล้วกด 'อัปเดตข้อมูล'")
