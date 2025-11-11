from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder="templates")

# Load model (pipeline หรือ model เดี่ยว)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "favorite_app_model.joblib")
model = joblib.load(MODEL_PATH)

JOBS = [
    "นักเรียน / นักศึกษา", "ข้าราชการ / พนักงานรัฐวิสาหกิจ", "พนักงานบริษัทเอกชน",
    "เจ้าของธุรกิจ / ผู้ประกอบการ / ฟรีแลนซ์", "ค้าขาย / พ่อค้าแม่ค้า",
    "วิศวกร / ช่างเทคนิค / IT / นักวิทยาศาสตร์", "เกษียณอายุ", "แม่บ้าน / พ่อบ้าน",
    "ไม่มีอาชีพ / ว่างงาน", "อื่น ๆ",
]
HOURS = ["1 - 2 ชั่วโมง", "3 - 4 ชั่วโมง", "มากกว่า 4 ชั่วโมง"]
ACTIVE = [
    "06.00 - 12.00 น. (ช่วงเช้า)", "12.00 - 18.00 น. (ช่วงบ่าย)",
    "18.00 - 24.00 น. (ช่วงกลางคืน)", "00.00 - 06.00 น. (ช่วงดึก)",
    "อื่น ๆ / ไม่ระบุเวลา",
]
YEARS = ["1 - 2 ปี", "2 - 3 ปี", "3 - 4 ปี", "มากกว่า 4 ปี"]
SMR_CHOICES = [
    "การสื่อสาร – คุยกับคนอื่น", "การรับข้อมูล – เสพข่าว ความรู้",
    "ความบันเทิง – ผ่อนคลาย สนุก", "การแสดงตัวตน – โพสต์ แชร์ สร้างภาพลักษณ์",
    "ประโยชน์เฉพาะทาง – ใช้ทำงาน ทำธุรกิจ หรือการศึกษา",
]
APPR_CHOICES = [
    "ใช้งานง่าย / สะดวก / ฟังก์ชันไม่ซับซ้อน", "มีเนื้อหาที่ตรงกับความสนใจ",
    "มีเพื่อน / คนรู้จักใช้เยอะ", "สามารถสื่อสาร พูดคุย หรือเชื่อมต่อกับผู้อื่นได้สะดวก",
    "มีความบันเทิงสูง (วิดีโอ รูปภาพ เพลง ฯลฯ)", "ใช้เพื่อติดตามข่าวสารหรือเหตุการณ์ปัจจุบันได้รวดเร็ว",
    "ใช้เพื่อติดตามบุคคลหรือเพจที่ชื่นชอบ (ดารา, อินฟลูเอนเซอร์, แบรนด์ ฯลฯ)",
    "เหมาะกับการแสดงออก / โพสต์รูป / วิดีโอ / ความคิดเห็นของตัวเอง",
    "สามารถใช้เพื่อขายของ / โปรโมตสินค้า / ทำธุรกิจได้ดี",
    "มีระบบแนะนำคอนเทนต์ (Algorithm) ที่ตรงใจ",
    "แอปพลิเคชันมีความน่าเชื่อถือ / ปลอดภัย / เสถียร",
]

@app.route("/", methods=["GET"])
def index():
    return render_template(
        "form.html",
        jobs=JOBS, hours=HOURS, active=ACTIVE, years=YEARS,
        smr_choices=SMR_CHOICES, appr_choices=APPR_CHOICES,
        result=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    # 1) รับค่าและทำ default กันพัง
    age = request.form.get("Age", type=int)
    if age is None:
        age = 0  # หรือกำหนดค่าเฉลี่ยตาม dataset ของคุณ

    gender = request.form.get("Gender", default="")
    jobs   = request.form.get("Jobs", default="")
    hours  = request.form.get("DailyUsageHours", default="")
    active = request.form.get("ActiveTimeClean", default="")
    years  = request.form.get("UsageYears", default="")
    smr_selected  = request.form.getlist("SocialMediaReason") or []
    appr_selected = request.form.getlist("AppReason") or []

    # 2) สร้าง DataFrame ให้คอลัมน์ตรงกับตอนเทรน
    df_in = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Jobs": jobs,
        "DailyUsageHours": hours,
        "ActiveTimeClean": active,
        "UsageYears": years,
        "SocialMediaReason": ", ".join(smr_selected),
        "AppReason": ", ".join(appr_selected),
    }])

    # 3) ทำนายคลาส
    yhat = model.predict(df_in)[0]

    # 4) Top-3 probability: รองรับทั้งกรณีเป็น Pipeline หรือโมเดลเดี่ยว
    proba_top3 = None
    classes = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_in)[0]
            classes = getattr(model, "classes_", None)
        elif hasattr(model, "named_steps"):  # เป็น Pipeline
            clf = model.named_steps.get("clf", None)
            if clf is not None and hasattr(clf, "predict_proba"):
                proba = model.predict_proba(df_in)[0]
                classes = clf.classes_
            else:
                proba = None
        else:
            proba = None

        if proba is not None and classes is not None:
            pairs = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:3]
            proba_top3 = [(c, float(round(p, 4))) for c, p in pairs]
    except Exception:
        proba_top3 = None  # ไม่ให้ล้มทั้งเพจ

    result = {
        "yhat": yhat,
        "top3": proba_top3,
        "echo": df_in.to_dict(orient="records")[0]
    }

    return render_template(
        "form.html",
        jobs=JOBS, hours=HOURS, active=ACTIVE, years=YEARS,
        smr_choices=SMR_CHOICES, appr_choices=APPR_CHOICES,
        result=result
    )

if __name__ == "__main__":
    # Local dev; สำหรับ production ใช้ gunicorn หรือ vercel-wsgi ตามแพลตฟอร์ม
    app.run(host="0.0.0.0", port=5000, debug=True)
