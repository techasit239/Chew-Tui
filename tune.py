"""หาค่าตั้งที่นับการเคี้ยวได้แม่นที่สุดจากรอบทดลองที่บันทึกไว้ (data/raw/*.json)

วิธีใช้: ดับเบิลคลิก tune.bat หรือรัน  python tune.py
ผลลัพธ์: data/best_settings.json  (ในแอปกด "⚡ ใช้ค่าที่จูนอัตโนมัติ" ใน sidebar)

ขั้นตอน
1. แบ่งรอบทดลองเป็นชุดจูน 70% และชุดทดสอบ 30% (ชุดทดสอบไม่ถูกใช้หาค่าเลย ใช้วัดว่าค่าที่ได้ใช้กับรอบใหม่ได้จริง)
2. คัดกรองเร็ว: ลองค่าหลายพันชุดด้วยตัวนับแบบประมาณ (กรองทั้งรอบทีเดียว) แล้วไต่หาค่าที่ดีขึ้นรอบๆ ชุดที่ดีที่สุด
3. ตรวจจริง: เล่นรอบทดลองซ้ำด้วยโค้ดเดียวกับในแอป (chew_counter.replay_trial) กับผู้เข้ารอบ แล้วเลือกชุดที่ดีที่สุด
"""
import argparse
import itertools
import json
import math
import random
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

from chew_counter import (BITE_RETRACT_SECONDS, DEFAULT_SETTINGS, MIN_PEAK_GAP, PEAK_DELAY, PEAK_EDGE,
                          PEAK_FS, PEAK_MIN_DATA, SIGNAL_SMOOTHING, SIGNALS, ChewCounter, replay_trial)

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_FILE = DATA_DIR / "best_settings.json"

IDLE_WEIGHT = 2.0  # รอบนั่งนิ่ง (จริง = 0) นับเกินแล้วน่ารำคาญกว่านับขาด จึงให้น้ำหนักมากกว่า

# ค่าที่ลอง (ตรงกับขั้นของแถบเลื่อนในแอป ค่าที่ได้จะโหลดเข้าแถบเลื่อนได้พอดี)
SPACE = {
    "method": ["peaks", "threshold"],
    "enabled": [list(c) for r in range(1, 5) for c in itertools.combinations(SIGNALS, r)],
    "lips_low": [90, 93, 95, 96, 97, 98],
    "lips_high": [102, 103, 104, 105, 107, 110, 119],
    "chin_v": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "chin_h": [1.0, 1.5, 2.0, 3.0, 4.0],
    "cheeks": [0.5, 1.0, 1.5, 2.0, 3.0],
    "band_low": [0.3, 0.4, 0.5, 0.7],
    "band_high": [2.0, 2.5, 3.0, 3.5],
    "prominence": [0.2, 0.3, 0.5, 0.8, 1.0],
    "noise_k": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "bite": [150, 160, 170, 180, 190, 200, 220, 250],  # เกณฑ์คำใหม่ (% ของค่าปกติ) จูนได้เฉพาะไฟล์รุ่น 2
}


def to_settings(p):
    return {
        "method": p["method"],
        "enabled": p["enabled"],
        "chew_low_pct": p["lips_low"] / 100,
        "chew_high_pct": p["lips_high"] / 100,
        "thresholds": {"chin_v": p["chin_v"] / 100, "chin_h": p["chin_h"] / 100, "cheeks": p["cheeks"] / 100},
        "peak_band": [p["band_low"], p["band_high"]],
        "peak_prominence": p["prominence"],
        "peak_noise_k": p["noise_k"],
        "bite_pct": p["bite"] / 100,
    }


def from_settings(s):
    s = {**DEFAULT_SETTINGS, **s}
    return {
        "method": s["method"], "enabled": [k for k in SIGNALS if k in s["enabled"]],
        "lips_low": round(s["chew_low_pct"] * 100), "lips_high": round(s["chew_high_pct"] * 100),
        "chin_v": s["thresholds"]["chin_v"] * 100, "chin_h": s["thresholds"]["chin_h"] * 100,
        "cheeks": s["thresholds"]["cheeks"] * 100,
        "band_low": s["peak_band"][0], "band_high": s["peak_band"][1],
        "prominence": s["peak_prominence"], "noise_k": s["peak_noise_k"],
        "bite": round(s["bite_pct"] * 100),
    }


def key_of(p):
    return json.dumps(p, sort_keys=True)


# รอบที่ข้อมูลไม่น่าเชื่อ จะทำให้หาค่าผิดทาง จึงไม่ใช้ (คืนเหตุผล หรือ None ถ้าใช้ได้)
def quality_problem(t):
    fr = t["frames"]
    if len(fr["t"]) < 30:
        return "ข้อมูลน้อยเกินไป"
    if t.get("format", 1) >= 2:
        # ไฟล์รุ่น 2: ช่วงตักอาหารเล่นซ้ำได้ ไม่นับเป็นข้อเสีย ดูแค่ช่วงที่หันหน้าเกินหรือหน้าหลุด
        usable = np.mean([s not in ("pose", "face") and v is not None for s, v in zip(fr["skip"], fr["lips"])])
        reason = "หันหน้าเกินกำหนด/หน้าหลุด"
    else:
        usable = np.mean([c and v is not None for c, v in zip(fr["counted"], fr["lips"])])
        reason = "หันหน้า/หน้าหลุด/อ้าปากค้าง"
    usable_secs = fr["t"][-1] * usable
    if usable < 0.6:
        return f"แอปหยุดนับ {100 - usable * 100:.0f}% ของเวลา ({reason})"
    # เคี้ยวจริงต้องใช้เวลาพอสมควร ถ้าสั้นมากน่าจะถูกตัดเป็นคำใหม่กลางรอบ จำนวนจริงจึงไม่ใช่ค่าที่ใส่ไว้
    if usable_secs < max(4.0, t["actual_chews"] * 0.45):
        return f"สั้นเกินไป ({usable_secs:.1f} วินาที) น่าจะถูกตัดเป็นคำใหม่กลางรอบ"
    return None


def load_trials(all_formats=False):
    trials, old = [], 0
    for f in sorted(RAW_DIR.glob("*.json")):
        try:
            t = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"ข้ามไฟล์ {f.name}: {e}")
            continue
        if not (t.get("baseline") and t.get("frames") and t.get("actual_chews") is not None):
            continue
        # ไฟล์รุ่นเก่าถูกลบข้อมูลช่วงที่แอปจับคำใหม่ผิดไป เล่นซ้ำการจับคำใหม่ไม่ได้ จึงไม่ใช้เป็นค่าตั้งต้น
        if t.get("format", 1) < 2 and not all_formats:
            old += 1
            continue
        problem = quality_problem(t)
        if problem:
            print(f"ไม่ใช้ {f.name} (จริง {t['actual_chews']} ครั้ง): {problem}")
            continue
        t["file"] = f.name
        trials.append(t)
    if old:
        print(f"ไม่ใช้ไฟล์รุ่นเก่า {old} ไฟล์ (บันทึกก่อนแก้การจับคำใหม่ ข้อมูลบางช่วงหายไป) "
              f"ใส่ --all-formats ถ้าต้องการใช้ด้วย")
    return trials


# ---------- ตัวนับแบบประมาณ (เร็ว) ใช้คัดกรอง ----------

class _BiteTracer(ChewCounter):
    """เล่นการจับคำใหม่ด้วยโค้ดจริง (ChewCounter.step) แต่ไม่นับ แค่จดว่าเฟรมไหนแอปจะนับ และตัวนับถูกล้างตรงไหน"""

    def __init__(self, settings):
        self.countable, self.resets, self._i = set(), [], -1
        super().__init__(settings)

    def count_step(self, sig, raw, now):
        self.countable.add(self._i)

    def reset_count(self):
        super().reset_count()
        if hasattr(self, "resets"):
            self.resets.append(self._i)


def countable_mask(trial, bite_pct):
    """เฟรมที่แอปจะนับ (หลังการล้างตัวนับครั้งสุดท้าย) เมื่อใช้เกณฑ์คำใหม่ค่านี้ (เหมือน replay_trial)"""
    fr = trial["frames"]
    n = len(fr["t"])
    if trial.get("format", 1) < 2:
        return np.array([fr["counted"][i] and fr["lips"][i] is not None for i in range(n)])
    tracer = _BiteTracer({**DEFAULT_SETTINGS, "bite_pct": bite_pct})
    tracer.baseline = trial["baseline"]
    st = trial["start_state"]
    age = st["last_bite_age"]
    tracer.reset_bite_state(st["is_biting"], None if age is None else -age)
    tracer.bite_frames, tracer.close_frames = st["bite_frames"], st["close_frames"]
    tracer.resets.clear()
    for i, t in enumerate(fr["t"]):
        tracer._i = i
        if fr["lips"][i] is None:
            tracer.sig_smooth = None
            tracer.pause_count()
            continue
        raw = {key: fr[key][i] for key in SIGNALS}
        tracer.smooth_signals(raw)
        if fr["skip"][i] == "pose":
            tracer.pause_count()
            continue
        if tracer.step(tracer.sig_smooth, raw, t) == "new_bite":
            break  # แอปจบรอบตรงนี้ (เหมือน replay_trial)
    else:
        # รอบจบเพราะคำใหม่ที่ค่านี้ยังไม่ทันจับได้: ตัดช่วงปากกำลังอ้าท้ายรอบออก (เหมือน replay_trial)
        if trial.get("ended_by") == "new_bite" and n:
            end = fr["t"][-1]
            tracer.countable = {i for i in tracer.countable if fr["t"][i] <= end - BITE_RETRACT_SECONDS}
    mask = np.zeros(n, bool)
    last_reset = max(tracer.resets, default=-1)
    for i in tracer.countable:
        if i > last_reset:
            mask[i] = True
    return mask


class FastTrial:
    """เตรียมข้อมูลหนึ่งรอบไว้ล่วงหน้า: แบ่งเป็นช่วงที่แอปนับต่อเนื่อง สุ่มใหม่ที่ 30 Hz และเก็บผลกรองไว้ใช้ซ้ำ
    ช่วงที่นับได้ขึ้นกับเกณฑ์คำใหม่ จึงเตรียมแยกตามค่าเกณฑ์คำใหม่แต่ละค่า"""

    def __init__(self, trial):
        self.actual = trial["actual_chews"]
        self.baseline = trial["baseline"]
        self.std = trial.get("signal_std") or {}
        self.trial = trial
        self._segments = {}
        self._filtered = {}

    def segments(self, bite_pct):
        if bite_pct in self._segments:
            return self._segments[bite_pct]
        fr = self.trial["frames"]
        mask = countable_mask(self.trial, bite_pct)
        segments = []
        start = None
        n = len(mask)
        for i in range(n + 1):
            ok = i < n and mask[i]
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                t = np.array(fr["t"][start:i])
                if len(t) > 1 and t[-1] - t[0] >= PEAK_MIN_DATA:
                    grid = np.arange(t[0], t[-1], 1 / PEAK_FS)
                    x = {}
                    for key in SIGNALS:
                        v = np.array(fr[key][start:i], dtype=float)
                        if key == "lips":
                            v = v / self.baseline["lips"]
                        x[key] = np.interp(grid, t, v)
                    segments.append((grid, x))
                start = None
        self._segments[bite_pct] = segments
        return segments

    def filtered(self, bite_pct, seg_i, key, band):
        k = (bite_pct, seg_i, key, band)
        if k not in self._filtered:
            sos = butter(2, band, btype="bandpass", fs=PEAK_FS, output="sos")
            self._filtered[k] = sosfiltfilt(sos, self.segments(bite_pct)[seg_i][1][key])
        return self._filtered[k]

    def count_peaks(self, s):
        band = tuple(s["peak_band"])
        band_fraction = math.sqrt((band[1] - band[0]) / (PEAK_FS / 2))
        total = 0
        for i, (grid, _) in enumerate(self.segments(s["bite_pct"])):
            best = None
            for key in [k for k in SIGNALS if k in s["enabled"]] or ["lips"]:
                scale = (min(1 - s["chew_low_pct"], s["chew_high_pct"] - 1) if key == "lips"
                         else s["thresholds"][key])
                f = self.filtered(s["bite_pct"], i, key, band) / scale
                noise = self.std.get(key, 0.0) / scale * band_fraction
                snr = float(f.std()) / max(noise, 0.02)
                if best is None or snr > best[0]:
                    best = (snr, f, noise)
            _, f, noise = best
            prom = max(s["peak_prominence"], s["peak_noise_k"] * noise)
            peaks, _ = find_peaks(f, prominence=prom, distance=max(1, int(MIN_PEAK_GAP * PEAK_FS)))
            t = grid[peaks]
            total += int(((t >= grid[0] + PEAK_EDGE) & (t <= grid[-1] - PEAK_DELAY)).sum())
        return total

    def count(self, s):
        if s["method"] == "peaks":
            return self.count_peaks(s)
        return replay_trial(self.trial, s)  # วิธีเกณฑ์เร็วอยู่แล้ว ใช้ตัวจริงได้เลย


def real_bite_openings(trials):
    """ปากอ้ากว้างสุด (เฉลี่ยแบบเดียวกับแอป) ช่วง 2 วินาทีแรกของรอบที่เริ่มจากการตักอาหารจริง (% ของค่าปกติ)

    ไฟล์แต่ละรอบจบตรงคำใหม่พอดี การเล่นซ้ำจึงไม่เคยถูกลงโทษเมื่อเกณฑ์คำใหม่สูงจนจับคำใหม่จริงไม่ได้
    (เคยเลือก 250% ทั้งที่ข้อมูลจริงตักอาหารอ้าปากแค่ 199–262% = จับคำใหม่ได้แค่ 3%) จึงต้องวัดจากข้อมูลตรงๆ
    """
    peaks = []
    for t in trials:
        if t.get("format", 1) < 2 or t["start_state"]["reason"] != "new_bite":
            continue
        fr, base, sm, best = t["frames"], t["baseline"]["lips"], None, None
        for ts, v in zip(fr["t"], fr["lips"]):
            if ts >= 2.0:
                break
            if v is None:
                sm = None
                continue
            sm = v if sm is None else sm + SIGNAL_SMOOTHING * (v - sm)
            best = sm if best is None else max(best, sm)
        if best is not None:
            peaks.append(best / base * 100)
    return peaks


def chewing_openings(trials):
    """ปากอ้ากว้างสุด (เฉลี่ยแบบเดียวกับแอป) ระหว่างเคี้ยวในแต่ละรอบ ไม่รวม 1 วินาทีสุดท้ายที่เริ่มตักคำถัดไป"""
    peaks = []
    for t in trials:
        if t.get("format", 1) < 2 or t["actual_chews"] == 0:
            continue
        fr, base, sm, best = t["frames"], t["baseline"]["lips"], None, None
        end = fr["t"][-1] - 1.0
        for ts, v, skip in zip(fr["t"], fr["lips"], fr["skip"]):
            if v is None:
                sm = None
                continue
            sm = v if sm is None else sm + SIGNAL_SMOOTHING * (v - sm)
            if skip is None and ts < end:
                best = sm if best is None else max(best, sm)
        if best is not None:
            peaks.append(best / base * 100)
    return peaks


def score(errors, actuals):
    w = [IDLE_WEIGHT if a == 0 else 1.0 for a in actuals]
    return sum(abs(e) * wi for e, wi in zip(errors, w)) / sum(w)


def fast_score(fast_trials, p):
    s = to_settings(p)
    errors = [ft.count(s) - ft.actual for ft in fast_trials]
    return score(errors, [ft.actual for ft in fast_trials])


# ---------- ตรวจจริงด้วยโค้ดเดียวกับในแอป ----------

_WORKER_TRIALS = None


def _init_worker(trials):
    global _WORKER_TRIALS
    _WORKER_TRIALS = trials


def _exact_counts(p):
    s = to_settings(p)
    return [replay_trial(t, s, analyze_every=2) for t in _WORKER_TRIALS]


def summarize(counts, trials):
    errors = [c - t["actual_chews"] for c, t in zip(counts, trials)]
    return {
        "score": round(score(errors, [t["actual_chews"] for t in trials]), 3),
        "mae": round(float(np.mean(np.abs(errors))), 2),
        "exact_pct": round(100 * float(np.mean([e == 0 for e in errors])), 1),
        "within1_pct": round(100 * float(np.mean([abs(e) <= 1 for e in errors])), 1),
        "n": len(trials),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random", type=int, default=3000, help="จำนวนค่าที่สุ่มลองในขั้นคัดกรอง")
    ap.add_argument("--finalists", type=int, default=20, help="จำนวนผู้เข้ารอบที่ตรวจด้วยโค้ดจริง")
    ap.add_argument("--seed", type=int, default=0)
    # ค่าตั้งต้นไม่ใช้แก้ม: ข้อมูลทดลองจริง แก้มแยกการเคี้ยวออกจากการนั่งนิ่งได้แย่ที่สุด
    ap.add_argument("--exclude", nargs="*", default=["cheeks"], choices=SIGNALS,
                    help="สัญญาณที่ไม่ต้องการให้ใช้ (ตั้งต้น: cheeks) ใส่ --exclude เปล่าๆ เพื่อลองทุกสัญญาณ")
    ap.add_argument("--all-formats", action="store_true",
                    help="ใช้ไฟล์รุ่นเก่าด้วย (บันทึกก่อนแก้การจับคำใหม่ ข้อมูลบางช่วงหายไป)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    if args.exclude:
        SPACE["enabled"] = [e for e in SPACE["enabled"] if not set(e) & set(args.exclude)]
        print(f"ไม่ใช้สัญญาณ: {', '.join(args.exclude)}")

    trials = load_trials(args.all_formats)
    print(f"พบรอบทดลอง {len(trials)} รอบใน {RAW_DIR}")
    if len(trials) < 8:
        print("ต้องมีอย่างน้อย 8 รอบ: เปิดโหมดทดลองในแอป แล้วเคี้ยวรอบละ 10 ครั้งให้ได้ 20-30 รอบ "
              "(แนะนำให้มีรอบนั่งนิ่งที่ใส่จำนวนจริง = 0 ด้วยสัก 3-5 รอบ)")
        sys.exit(1)

    # เกณฑ์คำใหม่ต้องจับการตักอาหารจริงได้อย่างน้อย 95% (ต่ำกว่าเปอร์เซ็นไทล์ที่ 5 ของการอ้าปากตอนตักอาหาร)
    openings = real_bite_openings(trials)
    if len(openings) >= 5:
        limit = float(np.percentile(openings, 5))
        allowed_bites = [b for b in SPACE["bite"] if b < limit] or [min(SPACE["bite"])]
        print(f"ตอนตักอาหารจริงอ้าปาก {min(openings):.0f}–{max(openings):.0f}% ({len(openings)} ครั้ง) "
              f"ลองเกณฑ์คำใหม่ได้ไม่เกิน {limit:.0f}%: {allowed_bites}")
    else:
        allowed_bites = [round(DEFAULT_SETTINGS["bite_pct"] * 100)]
        limit = None
        print(f"รอบที่เริ่มจากการตักอาหารมีน้อยกว่า 5 รอบ วัดการอ้าปากไม่ได้ ใช้เกณฑ์คำใหม่ {allowed_bites[0]}% ตามค่าตั้งต้น")
    SPACE["bite"] = allowed_bites
    # ถ้าหลายเกณฑ์คำใหม่ได้คะแนนเท่ากัน เลือกค่ากลางระหว่างการเคี้ยวแรงสุดกับการตักอาหารเบาสุด จะเผื่อได้ทั้งสองฝั่ง
    chewing = chewing_openings(trials)
    preferred = ((np.percentile(chewing, 90) + limit) / 2 if limit and len(chewing) >= 5
                 else DEFAULT_SETTINGS["bite_pct"] * 100)

    def rank_key(score_value, p):
        return score_value + 1e-3 * abs(p["bite"] - preferred)

    order = list(range(len(trials)))
    rng.shuffle(order)
    n_test = max(2, round(len(trials) * 0.3)) if len(trials) >= 10 else 0
    test = [trials[i] for i in order[:n_test]]
    train = [trials[i] for i in order[n_test:]]
    print(f"ชุดจูน {len(train)} รอบ, ชุดทดสอบ {len(test)} รอบ"
          + ("" if test else " (ข้อมูลน้อยกว่า 10 รอบ จึงไม่มีชุดทดสอบ ผลอาจดูดีเกินจริง)"))

    fast = [FastTrial(t) for t in train]
    seen = {}

    def evaluate(p):
        k = key_of(p)
        if k not in seen:
            seen[k] = (fast_score(fast, p), p)
        return seen[k][0]

    # ค่าที่ใช้อยู่ล่าสุดกับค่าตั้งต้นเป็นจุดเริ่มด้วย
    current = from_settings(trials[-1].get("settings") or DEFAULT_SETTINGS)
    evaluate(current)
    evaluate(from_settings(DEFAULT_SETTINGS))

    print(f"ขั้นที่ 1/3 สุ่มลอง {args.random} ชุด...")
    for i in range(args.random):
        evaluate({k: rng.choice(v) for k, v in SPACE.items()})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{args.random}  ดีที่สุดตอนนี้คลาดเฉลี่ย {min(v[0] for v in seen.values()):.2f}")

    print("ขั้นที่ 2/3 ไต่หาค่าที่ดีขึ้นรอบๆ ชุดที่ดีที่สุด...")
    starts = [p for _, p in sorted(seen.values(), key=lambda v: v[0])[:5]]
    for p in starts:
        p = dict(p)
        best = evaluate(p)
        improved = True
        while improved:
            improved = False
            for name, values in SPACE.items():
                for v in values:
                    if v == p[name]:
                        continue
                    q = {**p, name: v}
                    sc = evaluate(q)
                    if sc < best - 1e-9:
                        best, p, improved = sc, q, True

    def allowed(p):
        return not set(p["enabled"]) & set(args.exclude) and p["bite"] in SPACE["bite"]

    finalists = []
    for _, p in sorted(seen.values(), key=lambda v: rank_key(*v)):
        if allowed(p) and all(key_of(p) != key_of(f) for f in finalists):
            finalists.append(p)
        if len(finalists) >= args.finalists:
            break

    print(f"ขั้นที่ 3/3 ตรวจผู้เข้ารอบ {len(finalists)} ชุดด้วยโค้ดเดียวกับในแอป...")
    with Pool(initializer=_init_worker, initargs=(train,)) as pool:
        train_counts = pool.map(_exact_counts, finalists)
    ranked = sorted(zip(finalists, train_counts), key=lambda fc: rank_key(summarize(fc[1], train)["score"], fc[0]))
    best_p, best_train_counts = ranked[0]

    def evaluate_on(p, ts):
        s = to_settings(p)
        return summarize([replay_trial(t, s, analyze_every=2) for t in ts], ts)

    report = {"best": {"train": summarize(best_train_counts, train)},
              "current": {"train": evaluate_on(current, train)},
              "default": {"train": evaluate_on(from_settings(DEFAULT_SETTINGS), train)}}
    if test:
        for name, p in (("best", best_p), ("current", current), ("default", from_settings(DEFAULT_SETTINGS))):
            report[name]["test"] = evaluate_on(p, test)

    best_settings = to_settings(best_p)
    out = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_trials": len(trials),
        "settings": best_settings,
        "metrics": report,
    }
    DATA_DIR.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nผลลัพธ์ (คลาดเฉลี่ยต่อรอบ | นับตรงเป๊ะ | คลาดไม่เกิน ±1)")
    for name, label in (("best", "ค่าที่จูนได้"), ("current", "ค่าที่ใช้อยู่"), ("default", "ค่าตั้งต้น")):
        for part in ("train", "test"):
            if part in report[name]:
                m = report[name][part]
                print(f"  {label:12s} {'ชุดจูน' if part == 'train' else 'ชุดทดสอบ':8s} "
                      f"{m['mae']:5.2f} ครั้ง | {m['exact_pct']:5.1f}% | {m['within1_pct']:5.1f}%  ({m['n']} รอบ)")
    print("\nค่าที่จูนได้:")
    print(f"  วิธีนับ: {best_settings['method']}  สัญญาณ: {', '.join(best_settings['enabled'])}")
    print(f"  ริมฝีปาก {best_p['lips_low']}-{best_p['lips_high']}%  คางขึ้นลง {best_p['chin_v']}%  "
          f"คางซ้ายขวา {best_p['chin_h']}%  แก้ม {best_p['cheeks']}%")
    if best_settings["method"] == "peaks":
        print(f"  ช่วงความถี่ {best_p['band_low']}-{best_p['band_high']} Hz  ความสูงยอดคลื่น {best_p['prominence']}  "
              f"สูงกว่าจุดสั่น {best_p['noise_k']} เท่า")
    print(f"  เกณฑ์คำใหม่ {best_p['bite']}% ของค่าปกติ")
    print(f"\nบันทึกแล้ว: {OUT_FILE}")
    print('ในแอปกด "⚡ ใช้ค่าที่จูนอัตโนมัติ" ใน sidebar เพื่อใช้ค่านี้')


if __name__ == "__main__":
    main()
