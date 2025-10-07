import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Web
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")
STATIC_FOLDER   = os.path.join(BASE_DIR, "static")
CORS_ORIGINS    = os.getenv("CORS_ORIGINS", "https://g5weds.consolutechcloud.com")  # แก้ตามโดเมนจริง



# Models & Data
MODEL_PATH      = os.path.join(BASE_DIR, "model", "best.pt")
LEAFLET_PATH    = os.path.join(BASE_DIR, "data", "leaflet.csv")

# OCR
EASYOCR_LANGS   = os.getenv("EASYOCR_LANGS", "th,en").split(",")
EASYOCR_GPU     = os.getenv("EASYOCR_GPU", "false").lower() == "true"
