# ====== medicineBE.py ======
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io, base64
import cv2, os, json, re
import pandas as pd
import easyocr
from rapidfuzz import process, fuzz
import joblib
import types
from functools import wraps


# ===== โหลดคอนฟิก =====
from config import (
    TEMPLATE_FOLDER, STATIC_FOLDER,
    MODEL_PATH, LEAFLET_PATH,
    CORS_ORIGINS, EASYOCR_LANGS, EASYOCR_GPU
)

print("===== CONFIGURATION CHECK =====")
print(f"TEMPLATE_FOLDER   = {TEMPLATE_FOLDER}")
print(f"STATIC_FOLDER     = {STATIC_FOLDER}")
print(f"MODEL_PATH        = {MODEL_PATH}")
print(f"LEAFLET_PATH      = {LEAFLET_PATH}")
print(f"CORS_ORIGINS      = {CORS_ORIGINS}")
print(f"EASYOCR_LANGS     = {EASYOCR_LANGS}")
print(f"EASYOCR_GPU       = {EASYOCR_GPU}")
print("==========================================\n")

# -------------------------------------------------------------------------------------------------
# ตั้งค่าแอป
# -------------------------------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

CORS(app,
     resources={r"/(predict|health)": {"origins": CORS_ORIGINS}},
     supports_credentials=True
)

app.config.update(
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_SECURE=True,
    SECRET_KEY=os.environ.get("SECRET_KEY", "change-this")
)

@app.route("/", methods=["GET"])
def index_page():
    return render_template("first.page.html")

@app.route("/app", methods=["GET"])
def app_page():
    return render_template("second.page_full.html")


os.environ['YOLO_FUSE'] = '0'

# -------------------------------------------------------------------------------------------------
# 1) โหลดโมเดล YOLO
# -------------------------------------------------------------------------------------------------
model = YOLO(MODEL_PATH)
if hasattr(model, "overrides"):
    model.overrides.pop("fuse", None)
if hasattr(model, "model") and hasattr(model.model, "fuse"):
    model.model.fuse = types.MethodType(lambda self, *a, **k: self, model.model)

# -------------------------------------------------------------------------------------------------
# 2) โหลด Leaflet DB (CSV)
# -------------------------------------------------------------------------------------------------
def _safe_load_list(x):
    if pd.isna(x) or x == "":
        return []
    if isinstance(x, list):
        return x
    try:
        return json.loads(x)
    except Exception:
        return [str(x)]

cols = ["generics", "strengths", "indications", "warnings"]
if os.path.exists(LEAFLET_PATH):
    df = pd.read_csv(LEAFLET_PATH, encoding="utf-8-sig")
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(_safe_load_list)
        else:
            df[col] = [[] for _ in range(len(df))]
    LEAFLET_ROWS = df.to_dict(orient="records")
else:
    LEAFLET_ROWS = []

CANONICAL = sorted({
    (g or "").lower().strip()
    for row in LEAFLET_ROWS
    for g in row.get("generics", [])
    if isinstance(g, str) and g.strip() != ""
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EASY_DIR = os.path.join(BASE_DIR, "easyocr_models")
os.makedirs(EASY_DIR, exist_ok=True)

# -------------------------------------------------------------------------------------------------
# 3) โหลดโมเดลข้อความ
# -------------------------------------------------------------------------------------------------
TEXT_CLF = None
TEXT_CLF_LABELS = []
TEXT_VECT = None
try:
    from config import TEXT_MODEL_PATH, TEXT_SCORE_THRESHOLD
except Exception:
    TEXT_MODEL_PATH = "/mnt/data/medicine_text_model.pkl"
    TEXT_SCORE_THRESHOLD = 0.45

if os.path.exists(TEXT_MODEL_PATH):
    try:
        pack = joblib.load(TEXT_MODEL_PATH)
        TEXT_CLF = pack.get("model") if isinstance(pack, dict) else getattr(pack, "model", None)
        TEXT_VECT = pack.get("vectorizer") if isinstance(pack, dict) else getattr(pack, "vectorizer", None)
        TEXT_CLF_LABELS = (pack.get("labels") if isinstance(pack, dict) else getattr(pack, "labels", [])) or []
        print(f"[TEXT MODEL] Loaded: {TEXT_MODEL_PATH} (labels={len(TEXT_CLF_LABELS)})")
    except Exception as e:
        print(f"[TEXT MODEL] Failed to load: {e}")
else:
    print(f"[TEXT MODEL] Not found at {TEXT_MODEL_PATH}")

# -------------------------------------------------------------------------------------------------
# 4) EasyOCR
# -------------------------------------------------------------------------------------------------
ocr_reader = easyocr.Reader(['en', 'th'], gpu=EASYOCR_GPU,
                            model_storage_directory=EASY_DIR,
                            user_network_directory=EASY_DIR)

# -------------------------------------------------------------------------------------------------
# 5) Utilities
# -------------------------------------------------------------------------------------------------
SPLIT_RX = re.compile(r"\s*(?:\+|\/|,| with )\s*", re.I)
STRENGTH_RX = re.compile(r'(\d+(?:\.\d+)?)\s?(mg|g|mcg|µg|mL|ml|IU|%)', re.I)

def normalize_text(t: str):
    t = (t or "").strip().lower()
    t = re.sub(rf'[^a-z0-9\u0E00-\u0E7F\s\+\-\/\,\.µgml%]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def split_generics(t: str):
    t = normalize_text(t)
    return [p.strip() for p in SPLIT_RX.split(t) if p.strip()]

def parse_strengths(texts):
    strengths = set()
    for t in texts:
        for m in STRENGTH_RX.finditer(t):
            val, unit = m.group(1), m.group(2)
            unit = unit.replace('ml', 'mL')
            strengths.add(f"{val} {unit}")
    return sorted(strengths)

def canon_generic_one(word, score_cut=90):
    if not CANONICAL:
        return None, 0.0
    m, score, _ = process.extractOne(word, CANONICAL, scorer=fuzz.token_set_ratio)
    return (m, score/100.0) if score >= score_cut else (None, 0.0)

# -------------------------------------------------------------------------------------------------
# 6) Health check
# -------------------------------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"message": "API is running"}), 200

# -------------------------------------------------------------------------------------------------
# 7) /predict (no auth)
# -------------------------------------------------------------------------------------------------
@app.route("/predict", methods=["POST", "GET"])
def predict():
    try:
        had_image = False
        had_text = False
        user_text = ""
        pil_img = None

        if "image" in request.files:
            file = request.files["image"]
            img_bytes = file.read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            had_image = True
            user_text = request.form.get("text", "").strip()
            had_text = bool(user_text)
        else:
            data = request.get_json(force=True, silent=True) or {}
            b64 = (data.get("image_base64") or "").strip()
            if b64:
                if b64.startswith("data:"):
                    b64 = b64.split(",", 1)[1]
                img_bytes = base64.b64decode(b64)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                had_image = True
            user_text = (data.get("text") or data.get("query") or "").strip()
            had_text = bool(user_text)

        if not had_image and not had_text:
            return jsonify({"error": "No input. Send an image or text."}), 400

        # YOLO + OCR (if image)
        if had_image:
            results = model.predict(source=np.array(pil_img), imgsz=640, conf=0.25, verbose=False)
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else np.array([])
            print("Detected boxes:", len(boxes))

        return jsonify({"ok": True, "had_image": had_image, "had_text": had_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)