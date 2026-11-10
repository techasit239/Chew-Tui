"""ตัวนับการเคี้ยวจากสัญญาณใบหน้า

ใช้ร่วมกันระหว่างแอป (นับแบบเรียลไทม์ทีละเฟรม) และ tune.py (เล่นข้อมูลที่บันทึกไว้ซ้ำเพื่อหาค่าที่ดีที่สุด)
ทั้งสองที่จึงนับด้วยโค้ดชุดเดียวกัน ค่าที่ tune.py หาได้จะใช้ในแอปได้ผลเหมือนกัน
"""
import math
from collections import deque

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

SIGNALS = ["lips", "chin_v", "chin_h", "cheeks"]

SIGNAL_SMOOTHING = 0.5   # ค่าเฉลี่ยถ่วงน้ำหนักของสัญญาณเคี้ยว กันจุดสั่นจนนับเกิน
MIN_CHEW_INTERVAL = 0.25 # วินาที คนเคี้ยวไม่เร็วกว่านี้ กันนับซ้ำเมื่อหลายส่วนขยับไม่พร้อมกัน

# วิธีกรองความถี่ + นับยอดคลื่น
PEAK_FS = 30.0           # Hz สุ่มสัญญาณใหม่ให้ห่างเท่ากันก่อนกรอง (เฟรมจากเบราว์เซอร์มาไม่สม่ำเสมอ)
PEAK_WINDOW = 4.0        # วินาที ข้อมูลย้อนหลังที่ใช้กรองแต่ละครั้ง
PEAK_MIN_DATA = 1.5      # วินาที ต้องมีข้อมูลอย่างน้อยเท่านี้ถึงเริ่มหายอดคลื่น
PEAK_EDGE = 0.25         # วินาที ไม่นับยอดคลื่นช่วงต้นหน้าต่าง ฟิลเตอร์ยังไม่นิ่ง
PEAK_DELAY = 0.35        # วินาที รอให้ยอดคลื่นผ่านไปก่อนนับ จะได้รู้แน่ว่าเป็นยอดจริง
MIN_PEAK_GAP = 0.4       # วินาที ยอดคลื่นต้องห่างกันอย่างน้อยเท่านี้ (เคี้ยวเร็วสุดราว 150 ครั้ง/นาที)
# ยอดคลื่นต้องสูงกว่าจุดสั่น (วัดตอนคาลิเบรต) กี่เท่า กันนับตอนนั่งนิ่ง
# จำลองแล้ว: จุดสั่นล้วนๆ ขึ้นไปได้ถึง 6-9 เท่า (วัดยอดถึงท้องคลื่น และเลือกสัญญาณที่เด่นสุดทุกเฟรม)
PEAK_NOISE_K = 8.0

# จับคำใหม่ (อ้าปากกว้างตักอาหาร)
BITE_CONFIRM_FRAMES = 3  # ต้องอ้าปากกว้างต่อเนื่องกี่เฟรมถึงนับเป็นคำใหม่ (กันสัญญาณรบกวน)
BITE_COOLDOWN = 3.0      # วินาที อ้าปากซ้ำภายในช่วงนี้ = ยังเป็นคำเดิม (ตักอาหารแล้วอ้าซ้ำ ไม่ใช่คำใหม่)
BITE_MAX_SECONDS = 3.0   # วินาที ช่วงตักอาหารนานสุด เกินนี้เริ่มนับเลย แม้ปากยังหุบไม่สนิท
# วินาที ตอนเริ่มอ้าปากตักคำใหม่ ปากที่กำลังอ้าขึ้นถูกนับเป็นการเคี้ยวก่อนที่จะรู้ว่าเป็นคำใหม่
# พอยืนยันคำใหม่แล้ว จึงถอนการนับที่เกิดในช่วงนี้ออกจากคำเก่า (จำลองแล้ว: ไม่ถอน = เกินคำละ 1 ครั้ง)
BITE_RETRACT_SECONDS = 0.5

# ค่าตั้งต้น และชื่อที่ใช้ในไฟล์ค่าที่จูนแล้ว (data/best_settings.json)
DEFAULT_SETTINGS = {
    "method": "peaks",          # "peaks" = กรองความถี่ + ยอดคลื่น, "threshold" = เกณฑ์แบบเดิม
    # ไม่ใช้แก้ม: ข้อมูลทดลองจริง แก้มแยกการเคี้ยวออกจากการนั่งนิ่งได้แย่ที่สุดใน 4 สัญญาณ
    "enabled": ["lips", "chin_v", "chin_h"],
    "chew_low_pct": 0.96,       # ช่วงที่ริมฝีปากขยับได้โดยยังไม่นับ (สัดส่วนของค่าปกติ)
    "chew_high_pct": 1.04,
    # เกณฑ์ของคางและแก้ม เป็นสัดส่วนของระยะหางตาซ้าย–ขวา
    "thresholds": {"chin_v": 0.03, "chin_h": 0.02, "cheeks": 0.015},
    "peak_band": [0.5, 3.0],    # Hz ช่วงความถี่ที่เก็บไว้
    "peak_prominence": 0.5,     # ยอดคลื่นต้องสูงกว่าพื้นรอบข้างกี่เท่าของเกณฑ์แต่ละสัญญาณ
    "peak_noise_k": PEAK_NOISE_K,
    # อ้าปากกว้างกว่ากี่เท่าของค่าปกติถึงเป็นคำใหม่ ข้อมูลทดลองจริง: เคี้ยวอ้าถึงราว 1.25–1.5 เท่า
    # ตักอาหารราว 1.55–2.95 เท่า (เดิม 1.5 ชนกับการเคี้ยวแรงๆ จนตัวนับถูกล้างกลางรอบ)
    "bite_pct": 1.80,
}


class ChewCounter:
    """นับการเคี้ยวจากสัญญาณทีละเฟรม ต้องตั้ง baseline (และ signal_std ถ้ามี) จากการคาลิเบรตก่อนใช้"""

    def __init__(self, settings=None):
        self.baseline = None        # ค่าปกติของแต่ละสัญญาณ {ชื่อ: ค่า}
        self.signal_std = None      # ความสั่นตอนนิ่งของแต่ละสัญญาณ (ริมฝีปากเป็นสัดส่วนของค่าปกติ)
        self.apply_settings(settings or DEFAULT_SETTINGS)
        self.analyze_every = 1      # tune.py ตั้งให้มากกว่า 1 เพื่อเล่นข้อมูลซ้ำได้เร็วขึ้น
        self._sos_cache = {}
        self.peak_buffer = deque()  # [(เวลา, สัญญาณดิบ)] ย้อนหลัง PEAK_WINDOW วินาที
        self.peak_source = None     # สัญญาณที่ใช้นับในหน้าต่างล่าสุด
        self.sig_smooth = None
        self.last_dev = {}
        self.skip_reason = None     # เหตุที่เฟรมล่าสุดไม่ได้นับ ("bite" = ช่วงตักอาหาร) หรือ None = นับ
        self.reset_bite_state()
        self.reset_count()

    def reset_bite_state(self, biting=False, last_bite_time=None):
        self.is_biting = biting
        self.bite_frames = 0
        self.close_frames = 0
        self.last_bite_time = last_bite_time

    def apply_settings(self, settings):
        s = {**DEFAULT_SETTINGS, **settings}
        self.method = s["method"]
        self.enabled = set(s["enabled"])
        self.chew_low_pct = s["chew_low_pct"]
        self.chew_high_pct = s["chew_high_pct"]
        self.thresholds = dict(s["thresholds"])
        self.peak_band = tuple(s["peak_band"])
        self.peak_prominence = s["peak_prominence"]
        self.peak_noise_k = s["peak_noise_k"]
        self.bite_pct = s["bite_pct"]

    def settings(self):
        return {
            "method": self.method,
            "enabled": [k for k in SIGNALS if k in self.enabled],
            "chew_low_pct": self.chew_low_pct,
            "chew_high_pct": self.chew_high_pct,
            "thresholds": dict(self.thresholds),
            "peak_band": list(self.peak_band),
            "peak_prominence": self.peak_prominence,
            "peak_noise_k": self.peak_noise_k,
            "bite_pct": self.bite_pct,
        }

    # เริ่มนับคำใหม่ ยอดคลื่นของคำก่อนจะได้ไม่ถูกนับซ้ำในคำนี้
    def reset_count(self):
        self.chew_count = 0
        self.count_times = []       # เวลาที่นับได้แต่ละครั้ง (ใช้ถอนการนับตอนเริ่มอ้าปากตักคำใหม่)
        self.last_chew_time = None
        self.last_peak_time = None
        self.is_jaw_open = False
        self.peak_buffer.clear()
        self._frame_i = 0

    # หยุดนับชั่วคราว (หน้าหลุดหรือหันเกินกำหนด) ทิ้งจังหวะที่ค้างไว้ กันนับผิดตอนกลับมา
    def pause_count(self):
        self.is_jaw_open = False
        self.peak_buffer.clear()
        self.bite_frames = 0
        self.last_dev = {}

    # หนึ่งเฟรม: จับคำใหม่ก่อน แล้วค่อยนับการเคี้ยว
    # คืน "new_bite" เมื่อเริ่มคำใหม่ (ผู้เรียกต้องบันทึกคำเก่าแล้ว reset_count เอง) ไม่อย่างนั้นคืน None
    def step(self, sig, raw, now):
        lip_rel = sig["lips"] / self.baseline["lips"]
        self.last_dev = self.deviations(sig)
        self.skip_reason = "bite"

        # อ้าปากกว้างพิเศษ = กำลังตักอาหารคำใหม่เข้าปาก
        if lip_rel > self.bite_pct:
            # คลื่นช่วงอ้าปากกว้างใหญ่มาก จะทำให้ฟิลเตอร์แกว่งจนเกิดยอดปลอม จึงไม่เก็บ
            self.peak_buffer.clear()
            self.bite_frames += 1
            if self.bite_frames >= BITE_CONFIRM_FRAMES and not self.is_biting:
                new_bite = self.last_bite_time is None or now - self.last_bite_time >= BITE_COOLDOWN
                self.is_biting = True
                self.close_frames = 0
                self.last_bite_time = now
                if new_bite:
                    recent = [c for c in self.count_times if c > now - BITE_RETRACT_SECONDS]
                    self.chew_count -= len(recent)
                    self.count_times = self.count_times[:len(self.count_times) - len(recent)]
                    return "new_bite"
                # อ้าซ้ำในคำเดิม (ตักแล้วอ้าซ้ำ): ล้างจำนวนที่นับระหว่างตักอาหาร
                self.reset_count()
            return None
        self.bite_frames = 0

        if self.is_biting:
            # หุบปากกลับมาใกล้ปกติ (ไม่เกิน 1/4 ทางไปเกณฑ์คำใหม่) ติดกัน 3 เฟรม = ตักเสร็จ เริ่มนับการเคี้ยวได้
            # ถ้าเริ่มนับเร็วกว่านี้ (เช่นครึ่งทาง) ช่วงปากกำลังหุบลงจะถูกนับเป็นการเคี้ยว 1 ครั้งเกินมาทุกคำ
            # ดูแค่ริมฝีปาก: ถ้ารอให้ทุกสัญญาณนิ่ง จุดสั่นของริมฝีปากจะทำให้ค้างอยู่ตรงนี้ไม่ยอมนับ
            # (ข้อมูลทดลองจริง: ค้างถึง 13 วินาทีทั้งที่ปากกลับมาปกติแล้ว)
            # คางต้องกลับมาด้วย: ตอนตักอาหารคางลงต่ำมาก ถ้าเริ่มนับตอนคางยังขยับกลับ จะนับเกินอีก 1 ครั้ง
            # (ใช้เกณฑ์คางทั้งเกณฑ์ ไม่ใช่ครึ่งเดียว จุดสั่นของคางจึงไม่ทำให้ค้าง)
            closing = lip_rel < 1 + (self.bite_pct - 1) / 4 and self.last_dev["chin_v"] < 1.0
            self.close_frames = self.close_frames + 1 if closing else 0
            if self.close_frames >= BITE_CONFIRM_FRAMES or now - self.last_bite_time > BITE_MAX_SECONDS:
                self.is_biting = False
                self.close_frames = 0
                self.peak_buffer.clear()
            return None

        self.skip_reason = None
        self.count_step(sig, raw, now)
        return None

    def smooth_signals(self, sig):
        if self.sig_smooth is None:
            self.sig_smooth = dict(sig)
        else:
            for key, value in sig.items():
                self.sig_smooth[key] += SIGNAL_SMOOTHING * (value - self.sig_smooth[key])

    # แปลงแต่ละสัญญาณเป็น "สัดส่วนของเกณฑ์": 1.0 = ขยับถึงเกณฑ์นับพอดี
    def deviations(self, sig):
        rel = sig["lips"] / self.baseline["lips"]
        # ริมฝีปากนับทั้งสองทิศ: บางคนเคี้ยวแล้วริมฝีปากอ้าขึ้น บางคนหุบปากสนิทแล้วริมฝีปากเม้มลง
        if rel >= 1:
            dev = {"lips": (rel - 1) / (self.chew_high_pct - 1)}
        else:
            dev = {"lips": (1 - rel) / (1 - self.chew_low_pct)}
        for key, threshold in self.thresholds.items():
            dev[key] = abs(sig[key] - self.baseline[key]) / threshold
        return dev

    def activity(self, dev):
        return max([dev[key] for key in SIGNALS if key in self.enabled] or [dev["lips"]])

    # นับหนึ่งเฟรม: sig = สัญญาณที่เฉลี่ยแล้ว (วิธีเกณฑ์), raw = สัญญาณดิบ (ให้ฟิลเตอร์จัดการเอง)
    def count_step(self, sig, raw, now):
        if self.method == "peaks":
            self.count_peaks(raw, now)
            return
        # ส่วนใดส่วนหนึ่งขยับเกินเกณฑ์ แล้วทุกส่วนกลับมาต่ำกว่าครึ่งเกณฑ์ = เคี้ยว 1 ครั้ง
        activity = self.activity(self.deviations(sig))
        if activity > 1:
            self.is_jaw_open = True
        elif self.is_jaw_open and activity < 0.5:
            self.is_jaw_open = False
            if self.last_chew_time is None or now - self.last_chew_time >= MIN_CHEW_INTERVAL:
                self.chew_count += 1
                self.count_times.append(now)
                self.last_chew_time = now

    def bandpass(self):
        key = (self.peak_band, PEAK_FS)
        if key not in self._sos_cache:
            self._sos_cache[key] = butter(2, self.peak_band, btype="bandpass", fs=PEAK_FS, output="sos")
        return self._sos_cache[key]

    # วิธีกรองความถี่: เก็บสัญญาณย้อนหลัง -> กรองเหลือช่วงความถี่การเคี้ยว -> นับยอดคลื่น
    # การขยับช้าๆ (หน้าเลื่อน) และจุดสั่นเร็วๆ ถูกตัดทิ้ง ยอดคลื่นหนึ่งยอด = เคี้ยวหนึ่งครั้ง
    # ไม่ว่าการเคี้ยวนั้นริมฝีปากจะอ้าแล้วเม้มกี่จังหวะ
    def count_peaks(self, raw, now):
        self.peak_buffer.append((now, raw))
        while self.peak_buffer[0][0] < now - PEAK_WINDOW:
            self.peak_buffer.popleft()
        self._frame_i += 1
        if self._frame_i % self.analyze_every:
            return
        t0 = self.peak_buffer[0][0]
        if now - t0 < PEAK_MIN_DATA:
            return
        times = np.array([t for t, _ in self.peak_buffer])
        grid = np.arange(t0, now, 1 / PEAK_FS)
        if len(grid) < int(PEAK_MIN_DATA * PEAK_FS):
            return

        # แปลงทุกสัญญาณเป็นหน่วย "เท่าของเกณฑ์" แล้วเลือกสัญญาณที่คลื่นเด่นกว่าจุดสั่นของตัวเองมากที่สุด
        # (ไม่บวกรวมกัน เพราะริมฝีปากกับคางอาจขยับสวนทางกันจนหักล้างกัน)
        sos = self.bandpass()
        # จุดสั่นแบบสุ่มเหลือหลังกรองตามสัดส่วนความกว้างของช่วงความถี่ที่เก็บไว้
        band_fraction = math.sqrt((self.peak_band[1] - self.peak_band[0]) / (PEAK_FS / 2))
        best = None
        for key in [k for k in SIGNALS if k in self.enabled] or ["lips"]:
            values = np.array([s[key] for _, s in self.peak_buffer])
            if key == "lips":
                values = values / self.baseline["lips"]
                scale = min(1 - self.chew_low_pct, self.chew_high_pct - 1)
            else:
                scale = self.thresholds[key]
            filtered = sosfiltfilt(sos, np.interp(grid, times, values) / scale)
            noise = (self.signal_std or {}).get(key, 0.0) / scale * band_fraction
            snr = float(filtered.std()) / max(noise, 0.02)
            if best is None or snr > best[0]:
                best = (snr, key, filtered, noise)
        _, self.peak_source, filtered, noise = best
        min_prominence = max(self.peak_prominence, self.peak_noise_k * noise)

        peaks, _ = find_peaks(filtered, prominence=min_prominence,
                              distance=max(1, int(MIN_PEAK_GAP * PEAK_FS)))
        for i in peaks:
            t = grid[i]
            # ยอดคลื่นที่ขอบหน้าต่างยังไม่น่าเชื่อ รอให้เข้ามาอยู่ช่วงกลางก่อน (เฟรมถัดๆ ไปจะเจออีก)
            if t < t0 + PEAK_EDGE or t > now - PEAK_DELAY:
                continue
            # ยอดเดิมถูกเจอซ้ำทุกเฟรม จึงนับเฉพาะยอดที่อยู่หลังยอดล่าสุดที่นับไปแล้ว
            if self.last_peak_time is None or t > self.last_peak_time + MIN_PEAK_GAP:
                self.chew_count += 1
                self.count_times.append(now)
                self.last_chew_time = now
                self.last_peak_time = t


def replay_trial(trial, settings, analyze_every=1):
    """เล่นรอบทดลองที่บันทึกไว้ (จาก data/raw) ซ้ำด้วยค่าที่กำหนด แล้วคืนจำนวนที่นับได้

    ไฟล์รุ่น 2 (format = 2) เก็บทุกเฟรม จึงเล่นการจับคำใหม่ซ้ำได้ด้วย เกณฑ์คำใหม่ (bite_pct) จึงจูนได้
    ไฟล์รุ่นเก่าไม่มีข้อมูลพอ จะข้ามเฟรมที่แอปไม่ได้นับตามที่บันทึกไว้แทน
    """
    counter = ChewCounter(settings)
    counter.baseline = trial["baseline"]
    counter.signal_std = trial.get("signal_std")
    counter.analyze_every = analyze_every
    frames = trial["frames"]
    full = trial.get("format", 1) >= 2
    if full:
        # เริ่มจากสถานะการจับคำใหม่เดียวกับแอปตอนเริ่มรอบ (เวลาในไฟล์นับจาก 0 = ตอนเริ่มรอบ)
        st = trial["start_state"]
        age = st["last_bite_age"]
        counter.reset_bite_state(st["is_biting"], None if age is None else -age)
        counter.bite_frames = st["bite_frames"]
        counter.close_frames = st["close_frames"]
    for i, t in enumerate(frames["t"]):
        if frames["lips"][i] is None:  # หน้าหลุด แอปเริ่มเฉลี่ยสัญญาณใหม่
            counter.sig_smooth = None
            counter.pause_count()
            continue
        raw = {key: frames[key][i] for key in SIGNALS}
        counter.smooth_signals(raw)
        if full:
            if frames["skip"][i] == "pose":  # หันเกินกำหนด แอปหยุดนับ
                counter.pause_count()
                continue
            # จับคำใหม่ได้ = แอปจบรอบตรงนี้และบันทึกจำนวนที่นับได้ถึงตอนนั้น (เหมือนกันทั้งตอนจบรอบจริง
            # และตอนจับผิดกลางรอบ ซึ่งจะได้จำนวนน้อยกว่าจริง ทำให้ tune.py รู้ว่าเกณฑ์คำใหม่ต่ำเกินไป)
            if counter.step(counter.sig_smooth, raw, t) == "new_bite":
                return counter.chew_count
            continue
        if not frames["counted"][i]:
            counter.pause_count()
            continue
        counter.count_step(counter.sig_smooth, raw, t)
    # รอบนี้จบเพราะอ้าปากตักคำใหม่ แต่ค่าที่ลองอยู่ยังไม่ทันจับได้ (เช่นเกณฑ์คำใหม่สูงกว่าตอนบันทึก)
    # ถอนการนับช่วงปากกำลังอ้าเหมือนที่แอปทำตอนจับคำใหม่ได้ ไม่อย่างนั้นจะเกินรอบละ 1 ครั้งเพราะค่าที่ใช้บันทึก
    if full and trial.get("ended_by") == "new_bite" and frames["t"]:
        end = frames["t"][-1]
        counter.chew_count -= sum(c > end - BITE_RETRACT_SECONDS for c in counter.count_times)
    return counter.chew_count
