import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Web
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")
STATIC_FOLDER   = os.path.join(BASE_DIR, "static")
CORS_ORIGINS    = os.getenv("CORS_ORIGINS", "https://tafern.consolutechcloud.com")  # แก้ตามโดเมนจริง



# Models & Data
MODEL_PATH      = os.path.join(BASE_DIR, "model", "retinanet_drug_best.onnx")
LEAFLET_PATH    = os.path.join(BASE_DIR, "data", "leaflet_train.csv")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "data", "classes.txt") 

TEXT_MODEL_PATH = os.path.join(BASE_DIR, "model", "leaflet_realdata.joblib")
TEXT_VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "leaflet_realdata1.joblib")

def _get_int(env, default):
    try:
        return int(os.getenv(env, str(default)))
    except Exception:
        return default

TEXT_TOPK = _get_int("TEXT_TOPK", 5)

# ===== EasyOCR (ถ้าไม่ได้ใช้ก็ไม่เป็นไร) =====
# OCR
EASYOCR_LANGS   = os.getenv("EASYOCR_LANGS", "th,en").split(",")
EASYOCR_GPU     = os.getenv("EASYOCR_GPU", "false").lower() == "true"
