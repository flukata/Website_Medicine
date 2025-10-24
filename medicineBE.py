# ===== detection_api.py =====
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort
import io, base64, os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re

from config import (
    TEMPLATE_FOLDER, STATIC_FOLDER,
    MODEL_PATH, LEAFLET_PATH,
    CORS_ORIGINS, EASYOCR_LANGS, EASYOCR_GPU,
    CLASS_NAMES_PATH, TEXT_MODEL_PATH, TEXT_VECTORIZER_PATH, TEXT_TOPK
)

# ------------------- CLASS NAMES -------------------
CLASS_NAMES = None
try:
    # inline list (optional)
    from config import CLASS_NAMES as CFG_CLASS_NAMES
    if isinstance(CFG_CLASS_NAMES, (list, tuple)) and CFG_CLASS_NAMES:
        CLASS_NAMES = list(CFG_CLASS_NAMES)
        print(f"[CLASSNAMES] Using inline list from config ({len(CLASS_NAMES)})")
except Exception:
    pass

if CLASS_NAMES is None:
    try:
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8-sig") as f:
            CLASS_NAMES = [ln.strip() for ln in f if ln.strip()]
        print(f"[CLASSNAMES] Loaded {len(CLASS_NAMES)} from {CLASS_NAMES_PATH}")
    except Exception as e:
        print(f"[CLASSNAMES] Failed to load from {CLASS_NAMES_PATH}: {e}")
        CLASS_NAMES = None

# ------------------- APP & ONNX --------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

CORS(app,
     resources={r"/(predict|health)": {"origins": CORS_ORIGINS}},
     supports_credentials=True
)

sess = ort.InferenceSession(MODEL_PATH)
input_name = sess.get_inputs()[0].name
in_shape = sess.get_inputs()[0].shape  # e.g. [1,3,320,320] or [1,320,320,3]

def model_input_size_and_order(shape):
    """Return (W,H,is_chw) from model input shape."""
    if shape and len(shape) == 4:
        dims = [d if isinstance(d, int) else None for d in shape]
        # NCHW
        if dims[1] in (1, 3):
            H = dims[2] or 640
            W = dims[3] or 640
            return W, H, True
        # NHWC
        if dims[3] in (1, 3):
            H = dims[1] or 640
            W = dims[2] or 640
            return W, H, False
    return 640, 640, True

IN_W, IN_H, IS_CHW = model_input_size_and_order(in_shape)

def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    return out

def stable_softmax(x: np.ndarray, axis=1) -> np.ndarray:
    x = x.astype(np.float32)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.maximum(1e-12, np.sum(e, axis=axis, keepdims=True))

HEAD_TYPE = os.getenv("HEAD_TYPE", "sigmoid").lower()
CONF_TH   = float(os.getenv("CONF_TH", "0.30"))
IOU_TH    = float(os.getenv("IOU_TH",  "0.50"))

# ----------------- Box helpers ---------------------
def map_boxes_to_size(boxes_xyxy: np.ndarray, from_w: int, from_h: int, to_w: int, to_h: int) -> np.ndarray:
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    b = boxes_xyxy.astype(np.float32).copy()
    mn = float(np.min(b)); mx = float(np.max(b))
    if 0.0 <= mn and mx <= 1.0001:
        b[:, [0, 2]] *= to_w
        b[:, [1, 3]] *= to_h
    else:
        sx = float(to_w) / max(1.0, float(from_w))
        sy = float(to_h) / max(1.0, float(from_h))
        b[:, [0, 2]] *= sx
        b[:, [1, 3]] *= sy
    b[:, [0, 2]] = np.sort(b[:, [0, 2]], axis=1)
    b[:, [1, 3]] = np.sort(b[:, [1, 3]], axis=1)
    b[:, 0] = np.clip(b[:, 0], 0, to_w - 1)
    b[:, 2] = np.clip(b[:, 2], 0, to_w - 1)
    b[:, 1] = np.clip(b[:, 1], 0, to_h - 1)
    b[:, 3] = np.clip(b[:, 3], 0, to_h - 1)
    return b

def draw_boxes_in_place(img: Image.Image, boxes, color="red", width=2) -> Image.Image:
    out = img.copy()
    if out.mode not in ("RGB", "RGBA"):
        out = out.convert("RGB")
    draw = ImageDraw.Draw(out)
    for x0, y0, x1, y1 in boxes:
        draw.rectangle([int(x0), int(y0), int(x1), int(y1)], outline=color, width=width)
    return out

# =======================
# ==== TEXT MODEL 🧠 ====
# =======================
TEXT_PIPE = None      # pipeline (vectorizer+classifier)
TEXT_VEC  = None      # vectorizer (if separated)
TEXT_CLS  = None      # classifier (if separated)
TEXT_LABELS = None    # class labels if available
TEXT_LAST_ERROR = None

def _load_text_model():
    """Use separated classifier+vectorizer if TEXT_VECTORIZER_PATH exists; else treat as pipeline."""
    global TEXT_PIPE, TEXT_VEC, TEXT_CLS, TEXT_LABELS, TEXT_LAST_ERROR
    TEXT_LAST_ERROR = None
    try:
        # ใช้ joblib โหลดโมเดล Scikit-learn เช่น LogisticRegression, SVM, หรือ Pipeline ที่เทรนไว้
        cls_or_pipe = joblib.load(TEXT_MODEL_PATH)

        vec_path = (TEXT_VECTORIZER_PATH or "").strip()
        if vec_path and os.path.exists(vec_path):
            # แยกโหลด classifier + vectorizer
            TEXT_CLS = cls_or_pipe
            TEXT_VEC = joblib.load(vec_path)
            if hasattr(TEXT_CLS, "classes_") and TEXT_CLS.classes_ is not None:
                TEXT_LABELS = list(TEXT_CLS.classes_)
            print(f"[TEXTMODEL] Loaded classifier={TEXT_MODEL_PATH} + vectorizer={vec_path}")
            return True

        # โหลดเป็น pipeline เดียว
        TEXT_PIPE = cls_or_pipe
        if hasattr(TEXT_PIPE, "classes_") and TEXT_PIPE.classes_ is not None:
            TEXT_LABELS = list(TEXT_PIPE.classes_)
        elif hasattr(TEXT_PIPE, "named_steps"):
            for step in TEXT_PIPE.named_steps.values():
                if hasattr(step, "classes_") and step.classes_ is not None:
                    TEXT_LABELS = list(step.classes_)
                    break
        print(f"[TEXTMODEL] Loaded pipeline from {TEXT_MODEL_PATH}")
        return True

    except Exception as e:
        TEXT_LAST_ERROR = str(e)
        TEXT_PIPE = TEXT_VEC = TEXT_CLS = None
        TEXT_LABELS = None
        print(f"[TEXTMODEL] Failed to load: {e}")
        return False


    except Exception as e:
        TEXT_LAST_ERROR = str(e)
        TEXT_PIPE = TEXT_VEC = TEXT_CLS = None
        TEXT_LABELS = None
        print(f"[TEXTMODEL] Failed to load: {e}")
        return False

def _labels_name(idx_or_label):
    if isinstance(idx_or_label, (int, np.integer)):
        if TEXT_LABELS and 0 <= int(idx_or_label) < len(TEXT_LABELS):
            return str(TEXT_LABELS[int(idx_or_label)])
        return str(int(idx_or_label))
    return str(idx_or_label)

# ===============================
# ==== LEAFLET KNOWLEDGE BASE ===
# ===============================
DF_LEAFLET = None
GEN_COL = None
CHAR_VECT = None     # (vectorizer, matrix)
GEN_LIST = None      # list of entries (generics + synonyms)
GEN_CANON = None     # entry -> canonical generic (lower)

GEN_INDEX_EXACT = {}  # map คีย์ normalize -> canonical generic (lower)

def _key_norm(s: str) -> str:
    # แปลงเป็นตัวพิมพ์เล็ก + ลบทิ้งทุกอักขระที่ไม่ใช่ a-z/0-9 เพื่อกันเคสสะกดต่างกัน
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def _lookup_exact_generic(q: str):
    """คืน row ของ generic แบบ exact หลัง normalize; ถ้าไม่เจอคืน None"""
    k = _key_norm(q)
    g = GEN_INDEX_EXACT.get(k)
    if not g:
        return None
    return _row_for_canonical_generic(g)
    
def _norm(s):
    return str(s).strip().lower() if pd.notna(s) else ""

def _first_col(df, candidates):
    # exact
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive
    lowmap = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lowmap:
            return lowmap[c.lower()]
    return None

def _load_leaflet_df(path):
    encs = ["utf-8-sig", "utf-8", "cp874", "latin-1"]
    last_err = None
    for enc in encs:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    print(f"[LEAFLET] Failed to read {path}: {last_err}")
    return None

def re_split(pat, s):  # tiny helper to keep code tidy
    return re.split(pat, s)

def _get_field(row_or_dict, keys):
    # works for dict-like row
    for k in keys:
        if k in row_or_dict and pd.notna(row_or_dict[k]) and str(row_or_dict[k]).strip():
            return str(row_or_dict[k]).strip()
    return ""

def _build_generic_index():
    """Read CSV -> resolve columns -> build char TF-IDF index with synonyms."""
    global DF_LEAFLET, GEN_COL, CHAR_VECT, GEN_LIST, GEN_CANON
    if not LEAFLET_PATH or not os.path.exists(LEAFLET_PATH):
        print(f"[LEAFLET] File not found: {LEAFLET_PATH}")
        return False

    DF_LEAFLET = _load_leaflet_df(LEAFLET_PATH)
    if DF_LEAFLET is None or DF_LEAFLET.empty:
        print(f"[LEAFLET] Empty or unreadable CSV at {LEAFLET_PATH}")
        return False

    DF_LEAFLET.columns = [c.strip() for c in DF_LEAFLET.columns]
    GEN_COL = _first_col(DF_LEAFLET, ["generics","generic","generic_name","name","drug","drug_name"])
    if not GEN_COL:
        print("[LEAFLET] Missing generic column (expected generics/generic/generic_name/name/drug/drug_name)")
        return False

    SYN_COL = _first_col(DF_LEAFLET, ["synonyms","alias","aka"])  # optional

    # entries = generics + synonyms (lowercased)
    entries = []
    canon = {}

    for _, r in DF_LEAFLET.iterrows():
        g = _norm(r.get(GEN_COL, ""))
        if not g:
            continue
        entries.append(g)
        canon[g] = g
        if SYN_COL and pd.notna(r.get(SYN_COL)):
            syn_raw = str(r[SYN_COL])
            for tok in [t.strip() for t in re_split(r"[|;,/]", syn_raw) if t.strip()]:
                tnorm = _norm(tok)
                if tnorm:
                    entries.append(tnorm)
                    canon[tnorm] = g

    # unique preserve order
    uniq, seen = [], set()
    for e in entries:
        if e not in seen:
            uniq.append(e); seen.add(e)

    # TF-IDF char_wb 2..5
    vect = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5))
    G = vect.fit_transform(uniq)

    GEN_LIST = uniq
    GEN_CANON = canon
    CHAR_VECT = (vect, G)
    print(f"[LEAFLET] Index built with {len(GEN_LIST)} entries from {len(DF_LEAFLET)} rows (GEN_COL='{GEN_COL}')")
    return True

def _row_for_canonical_generic(g_lower):
    if DF_LEAFLET is None or DF_LEAFLET.empty or not GEN_COL:
        return None
    m = DF_LEAFLET[DF_LEAFLET[GEN_COL].astype(str).str.strip().str.lower() == g_lower]
    if m.empty:
        return None
    return m.iloc[0].to_dict()

def _assemble_details(row_dict):
    if not row_dict:
        return None
    ind = _get_field(row_dict, ["indications","indication","uses"])
    ins = _get_field(row_dict, ["instructions","dosage","direction","how_to_use"])
    war = _get_field(row_dict, ["warnings","warning","precautions","caution"])
    stg = _get_field(row_dict, ["strength","dosage_strengths","strengths","dose"])
    gen = _get_field(row_dict, ["generics","generic","generic_name","name","drug","drug_name"])
    return {
        "generics": gen or None,
        "strength": stg or None,
        "indications": ind or None,
        "instructions": ins or None,
        "warnings": war or None
    }

def _tfidf_search(query, topk=5, threshold=0.25):
    """Return [(display_name, sim, canonical_generic_lower), ...]"""
    if not query or not CHAR_VECT or not GEN_LIST:
        return []
    q = _norm(query)
    vect, G = CHAR_VECT
    sims = linear_kernel(vect.transform([q]), G)[0]
    idx = np.argsort(-sims)[:topk]
    out = []
    for i in idx:
        s = float(sims[i])
        if s < threshold:
            continue
        name = GEN_LIST[i]
        canon = GEN_CANON.get(name, name)
        out.append((name, s, canon))
    return out

# build leaflet index at startup (soft-fail)
try:
    _build_generic_index()
except Exception as e:
    print(f"[LEAFLET] Build index error: {e}")

# ===============================
# ==== TEXT INFERENCE + KB   ====
# ===============================
def analyze_text(query: str, topk: int = None):
    global TEXT_PIPE, TEXT_VEC, TEXT_CLS, TEXT_LABELS, TEXT_LAST_ERROR
    if not query or not query.strip():
        return None

    # Ensure model is loaded (but we can still return KB results if model missing)
    model_ready = True
    if TEXT_PIPE is None and TEXT_CLS is None:
        model_ready = _load_text_model()

    topk = topk or TEXT_TOPK
    q = query.strip()

    # 1) Try model inference if available
    top = []
    best = {}
    if model_ready and (TEXT_VEC is not None and TEXT_CLS is not None):
        try:
            X = TEXT_VEC.transform([q])
            if hasattr(TEXT_CLS, "predict_proba"):
                p = TEXT_CLS.predict_proba(X)[0]
            elif hasattr(TEXT_CLS, "decision_function"):
                s = np.array(TEXT_CLS.decision_function(X)[0], dtype=np.float32)
                ex = np.exp(s - np.max(s)); p = ex / np.maximum(1e-12, ex.sum())
            else:
                pred = TEXT_CLS.predict(X)[0]
                best = {"label": str(pred), "score": 1.0}
                top = [best]
                p = None
            if p is not None:
                idx = np.argsort(p)[::-1][:topk]
                for i in idx:
                    lab = TEXT_LABELS[i] if (TEXT_LABELS is not None and i < len(TEXT_LABELS)) else i
                    top.append({"label": str(lab), "score": float(p[i])})
                if top:
                    best = top[0]
        except Exception as e:
            # keep going to KB fallback
            print(f"[TEXTMODEL] Separated inference error: {e}")

    elif model_ready and (TEXT_PIPE is not None):
        try:
            if hasattr(TEXT_PIPE, "predict_proba"):
                p = TEXT_PIPE.predict_proba([q])[0]
            elif hasattr(TEXT_PIPE, "decision_function"):
                s = np.array(TEXT_PIPE.decision_function([q])[0], dtype=np.float32)
                ex = np.exp(s - np.max(s)); p = ex / np.maximum(1e-12, ex.sum())
            else:
                pred = TEXT_PIPE.predict([q])[0]
                best = {"label": str(pred), "score": 1.0}
                top = [best]
                p = None
            if p is not None:
                idx = np.argsort(p)[::-1][:topk]
                for i in idx:
                    lab = TEXT_LABELS[i] if (TEXT_LABELS is not None and i < len(TEXT_LABELS)) else i
                    top.append({"label": str(lab), "score": float(p[i])})
                if top:
                    best = top[0]
        except Exception as e:
            print(f"[TEXTMODEL] Pipeline inference error: {e}")

    # 2) Enrich with KB (TF-IDF) even if model not ready
    search_key = (best.get("label") if best else None) or q
    tfidf_hits = _tfidf_search(search_key, topk=max(3, topk), threshold=0.20)
    details = None
    alts = []
    if tfidf_hits:
        top_name, sim, canon = tfidf_hits[0]
        row = _row_for_canonical_generic(canon)
        details = {
            "source": "tfidf_leaflet",
            "match": top_name,
            "match_score": float(sim),
            "canonical_generic": canon,
            "fields": _assemble_details(row)
        }
        for name, s, _canon in tfidf_hits[1:3]:
            alts.append({"generic": name, "score": float(s)})

    # 3) Build response
    # NOTE: ถ้าไม่มีโมเดลแต่มีรายละเอียดจาก KB เรา "ไม่ส่ง error" เพื่อให้ฝั่งหน้าเว็บโชว์ผล
    resp = {
        "query": q,
        "top": top,
        "best": best,
        "details": details,
        "alternatives": alts
    }
    if not model_ready and not details:
        resp["error"] = f"Text model not available: {TEXT_LAST_ERROR or 'unknown error'}"
    return resp


def analyze_label_like_text(label: str, topk: int = None):
    if not label:
        return None
    topk = topk or TEXT_TOPK
    q = label.strip()

    # 1) Exact lookup (generics or synonyms)
    row = _lookup_exact_generic(q)
    if row:
        canon = _norm(row.get(GEN_COL, ""))
        details = {
            "source": "leaflet_exact",
            "match": q,
            "match_score": 1.0,
            "canonical_generic": canon,
            "fields": _assemble_details(row)
        }
        return {
            "query": q,
            "top": [{"label": canon or q, "score": 1.0}],
            "best": {"label": canon or q, "score": 1.0},
            "details": details,
            "alternatives": []
        }

    # 2) TF-IDF with stricter acceptance
    strict_th = 0.55
    hits = _tfidf_search(q, topk=max(5, topk), threshold=0.20)

    # optional: filter by same first letter
    initial = q[:1].lower()
    hits = [(n,s,c) for (n,s,c) in hits if n[:1].lower()==initial]

    if hits and hits[0][1] >= strict_th:
        name, sim, canon = hits[0]
        row = _row_for_canonical_generic(canon)
        details = {
            "source": "tfidf_leaflet",
            "match": name,
            "match_score": float(sim),
            "canonical_generic": canon,
            "fields": _assemble_details(row)
        }
        alts = [{"generic": n, "score": float(s)} for (n,s,_) in hits[1:3]]
        return {
            "query": q,
            "top": [{"label": name, "score": float(sim)}],
            "best": {"label": name, "score": float(sim)},
            "details": details,
            "alternatives": alts
        }

    # 3) No reliable KB hit → keep detection label; no details
    return {
        "query": q,
        "top": [{"label": q, "score": 1.0}],
        "best": {"label": q, "score": 1.0},
        "details": None,
        "alternatives": []
    }

# ------------------- ROUTES -------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"message": "API is running"}), 200

@app.route("/predict", methods=["POST", "GET"])
def predict():
    try:
        had_image = False
        had_text  = False
        user_text = ""
        pil_img   = None

        # ---- robust input parsing: multipart vs json
        ct = (request.content_type or "").lower()
        is_multipart = "multipart/form-data" in ct

        if is_multipart:
            if "image" in request.files and getattr(request.files["image"], "filename", ""):
                img_bytes = request.files["image"].read()
                pil_img = Image.open(io.BytesIO(img_bytes))
                had_image = True
            user_text = (request.form.get("text") or request.form.get("query") or "").strip()
            had_text = bool(user_text)
        else:
            data = request.get_json(force=True, silent=True) or {}
            b64 = (data.get("image_base64") or "").strip()
            if b64:
                if b64.startswith("data:"):
                    b64 = b64.split(",", 1)[1]
                img_bytes = base64.b64decode(b64)
                pil_img = Image.open(io.BytesIO(img_bytes))
                had_image = True
            user_text = (data.get("text") or data.get("query") or "").strip()
            had_text = bool(user_text)

        if not had_image and not had_text:
            return jsonify({"ok": False, "error": "No input. Send an image or text."}), 400

        detections = []
        obb_boxes  = []
        image_b64  = ""
        text_result = None

        # =============== IMAGE: ONNX detection (enhanced) ===============
        if had_image:
            W0, H0 = pil_img.size

            # --- options from env ---
            LETTERBOX    = os.getenv("LETTERBOX", "1") == "1"
            MODEL_COLOR  = os.getenv("MODEL_COLOR", "rgb").lower()   # 'rgb' or 'bgr'
            ADAPTIVE_TH  = os.getenv("ADAPTIVE_TH", "1") == "1"

            # parse mean/std (0..1 scale; because we divide by 255 below)
            def _parse_vec3(env_key):
                raw = (os.getenv(env_key, "") or "").strip()
                if not raw:
                    return None
                try:
                    vals = [float(x) for x in raw.split(",")]
                    return vals if len(vals) == 3 else None
                except:
                    return None
            NORM_MEAN = _parse_vec3("NORM_MEAN")  # e.g., 0.485,0.456,0.406
            NORM_STD  = _parse_vec3("NORM_STD")   # e.g., 0.229,0.224,0.225

            # --- letterbox helper (keep ratio, pad with gray) ---
            def _letterbox_pil(im, new_w, new_h, color=(114,114,114)):
                w, h = im.size
                r = min(new_w / w, new_h / h)
                rw, rh = int(round(w * r)), int(round(h * r))
                pad_w, pad_h = new_w - rw, new_h - rh
                pad_l, pad_t = pad_w // 2, pad_h // 2
                # resize + paste
                im_res = im.resize((rw, rh), Image.BILINEAR)
                canvas = Image.new("RGB", (new_w, new_h), color)
                canvas.paste(im_res, (pad_l, pad_t))
                return canvas, r, pad_l, pad_t  # ratio, dx, dy

            # --- preprocess to model input ---
            if LETTERBOX:
                img_for_model, ratio, dx, dy = _letterbox_pil(pil_img.convert("RGB"), IN_W, IN_H)
            else:
                img_for_model = pil_img.convert("RGB").resize((IN_W, IN_H), Image.BILINEAR)
                # emulate "letterbox = off"
                ratio = min(IN_W / float(W0), IN_H / float(H0))
                # เมื่อ resize แบบยืด ไม่ได้ใช้ pad จริง ๆ แต่ set dx,dy เป็น 0 ไว้
                dx, dy = 0.0, 0.0

            arr = np.asarray(img_for_model).astype(np.float32) / 255.0   # [0,1], RGB by PIL

            # optional: color order swap
            if MODEL_COLOR == "bgr":
                arr = arr[..., ::-1]  # RGB->BGR

            # optional: mean/std normalization
            if NORM_MEAN and NORM_STD:
                # broadcast [H,W,C] - [C] / [C]
                arr = (arr - np.array(NORM_MEAN, dtype=np.float32)) / np.array(NORM_STD, dtype=np.float32)

            # layout
            if IS_CHW:
                arr = np.transpose(arr, (2, 0, 1))  # HWC->CHW
            img_np = np.expand_dims(arr, 0).astype(np.float32, copy=False)

            # --- run onnx ---
            outputs = sess.run(None, {input_name: img_np})

            # --- parse outputs (ตามที่คุณปรับไว้แล้ว) ---
            # ... (ใช้ parse_onnx_outputs / parsed เหมือนเวอร์ชันล่าสุดของคุณ) ...
            # สมมติหลังบล็อกนี้มี: boxes_raw, scores, labels พร้อมใช้งาน
            def parse_onnx_outputs(outs):
                """
                Support 3 styles:
                A) [boxes(N,4), logits(N,C)]
                B) [joined(N, 4+C)]
                C) [boxes(N,4), scores(N,), labels(N,)]
                """
                arrs = [np.asarray(o) for o in outs]

                # squeeze batch dim=1 if present (e.g., (1,N,4)->(N,4), (1,N)->(N,))
                def _sq(a):
                    if a.ndim >= 3 and a.shape[0] == 1:
                        return a[0]
                    if a.ndim == 2 and a.shape[0] == 1:
                        return a[0]
                    return a

                arrs = [_sq(a) for a in arrs]

                # C) boxes + scores + labels (NMS already applied)
                if len(arrs) == 3:
                    b, s, l = arrs
                    if b.ndim == 2 and b.shape[1] == 4 and s.ndim in (1, 2) and l.ndim in (1, 2):
                        s = s.reshape(-1).astype(np.float32)
                        l = l.reshape(-1).astype(np.int64)
                        if b.shape[0] == s.shape[0] == l.shape[0]:
                            return {"type": "bsl", "boxes": b.astype(np.float32), "scores": s, "labels": l}

                # A) boxes + logits
                if len(arrs) == 2:
                    b, log = arrs
                    if b.ndim == 2 and b.shape[1] == 4 and log.ndim == 2 and log.shape[0] == b.shape[0]:
                        return {"type": "bl", "boxes": b.astype(np.float32), "logits": log.astype(np.float32)}

                # B) joined (N, 4+C)
                if len(arrs) == 1:
                    x = arrs[0]
                    if x.ndim == 2 and x.shape[1] >= 5:
                        return {"type": "joined", "joined": x.astype(np.float32)}

                raise ValueError(f"Unsupported ONNX outputs: {[a.shape for a in arrs]}")

            parsed = parse_onnx_outputs(outputs)

            # ป้องกัน NameError: ตั้งค่า default ว่างไว้ก่อน
            boxes_raw = np.empty((0, 4), dtype=np.float32)
            scores    = np.empty((0,),    dtype=np.float32)
            labels    = np.empty((0,),    dtype=np.int64)

            # เติมค่าจริงตามชนิดเอาต์พุต
            if parsed["type"] == "bsl":          # boxes + scores + labels (NMS แล้ว)
                boxes_raw = parsed["boxes"]
                scores    = parsed["scores"]
                labels    = parsed["labels"]

            elif parsed["type"] == "bl":          # boxes + logits
                boxes_raw = parsed["boxes"]
                cls_logits = parsed["logits"]
                probs  = stable_softmax(cls_logits) if (HEAD_TYPE == "softmax") else stable_sigmoid(cls_logits)
                scores = probs.max(axis=1)
                labels = probs.argmax(axis=1)

            elif parsed["type"] == "joined":      # joined (N, 4+C)
                x = parsed["joined"]
                boxes_raw = x[:, :4]
                cls_logits = x[:, 4:]
                probs  = stable_softmax(cls_logits) if (HEAD_TYPE == "softmax") else stable_sigmoid(cls_logits)
                scores = probs.max(axis=1)
                labels = probs.argmax(axis=1)

            else:
                # กันเคส parser คืน type แปลก จะไม่ไปต่อจนเกิด NameError
                raise ValueError(f"Unexpected parsed type: {parsed.get('type')}")


            def to_xyxy(boxes):
                boxes = boxes.astype(np.float32)
                if boxes.size == 0:
                    return boxes
                okx = np.mean(boxes[:, 2] > boxes[:, 0])
                oky = np.mean(boxes[:, 3] > boxes[:, 1])
                if okx > 0.9 and oky > 0.9:
                    return boxes
                cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x1 = cx - w / 2; y1 = cy - h / 2
                x2 = cx + w / 2; y2 = cy + h / 2
                return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

            boxes_xyxy_in = to_xyxy(boxes_raw)

            # --- mask by confidence ---
            mask = scores >= CONF_TH
            boxes_xyxy_in = boxes_xyxy_in[mask]
            scores_sel    = scores[mask]
            labels_sel    = labels[mask]

            # --- adaptive threshold (optional): ถ้าไม่เจออะไรเลย ลองลดเพดานชั่วคราว ---
            if ADAPTIVE_TH and boxes_xyxy_in.size == 0 and scores.size > 0:
                low_th = max(0.10, CONF_TH * 0.5)
                m2 = scores >= low_th
                if np.any(m2):
                    boxes_xyxy_all = to_xyxy(boxes_raw)
                    # (ถ้า boxes_raw เดิม normalized ให้ใช้บล็อกสเกล 0..1 -> พิกเซล เหมือนด้านบนซ้ำอีกครั้ง)
                    mn = float(np.min(boxes_xyxy_all)) if boxes_xyxy_all.size else 0.0
                    mx = float(np.max(boxes_xyxy_all)) if boxes_xyxy_all.size else 0.0
                    if boxes_xyxy_all.size and 0.0 <= mn and mx <= 1.0001:
                        boxes_xyxy_all[:, [0, 2]] *= IN_W
                        boxes_xyxy_all[:, [1, 3]] *= IN_H

                    boxes_xyxy_in = boxes_xyxy_all[m2]
                    scores_sel    = scores[m2]
                    labels_sel    = labels[m2]
                    print(f"[DETECT] adaptive threshold used: {CONF_TH} -> {low_th} (kept {len(scores_sel)})")


            # --- NMS ---
            def box_iou_xyxy(a, b):
                ax1, ay1, ax2, ay2 = a[:,0], a[:,1], a[:,2], a[:,3]
                bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
                inter_x1 = np.maximum(ax1[:,None], bx1[None,:])
                inter_y1 = np.maximum(ay1[:,None], by1[None,:])
                inter_x2 = np.minimum(ax2[:,None], bx2[None,:])
                inter_y2 = np.minimum(ay2[:,None], by2[None,:])
                inter_w = np.maximum(0.0, inter_x2 - inter_x1)
                inter_h = np.maximum(0.0, inter_y2 - inter_y1)
                inter = inter_w * inter_h
                area_a = np.maximum(0.0, (ax2-ax1)) * np.maximum(0.0, (ay2-ay1))
                area_b = np.maximum(0.0, (bx2-bx1)) * np.maximum(0.0, (by2-by1))
                return inter / np.maximum(1e-9, area_a[:,None] + area_b[None,:] - inter)

            def nms_xyxy(boxes, scores, iou_th=0.5, max_det=300):
                if boxes.size == 0:
                    return []
                order = scores.argsort()[::-1]
                keep = []
                while order.size > 0 and len(keep) < max_det:
                    i = order[0]
                    keep.append(i)
                    if order.size == 1:
                        break
                    ious = box_iou_xyxy(boxes[i:i+1], boxes[order[1:]])[0]
                    remain = np.where(ious <= iou_th)[0]
                    order = order[remain + 1]
                return keep

            keep_idx = nms_xyxy(boxes_xyxy_in, scores_sel, iou_th=IOU_TH, max_det=300)
            boxes_xyxy_in = boxes_xyxy_in[keep_idx]
            scores_sel    = scores_sel[keep_idx]
            labels_sel    = labels_sel[keep_idx]

            # --- map to original image (unletterbox-aware) ---
            if LETTERBOX:
                b = boxes_xyxy_in.copy()
                b[:, [0,2]] -= dx
                b[:, [1,3]] -= dy
                # แปลงกลับสเกลเดิม
                b[:, [0,2]] = b[:, [0,2]] / max(1e-12, ratio)
                b[:, [1,3]] = b[:, [1,3]] / max(1e-12, ratio)
                # clip
                b[:, 0] = np.clip(b[:, 0], 0, W0 - 1)
                b[:, 2] = np.clip(b[:, 2], 0, W0 - 1)
                b[:, 1] = np.clip(b[:, 1], 0, H0 - 1)
                b[:, 3] = np.clip(b[:, 3], 0, H0 - 1)
                boxes_xyxy_orig = b
            else:
                # resize แบบยืด — ใช้ตัวช่วยเดิม
                boxes_xyxy_orig = map_boxes_to_size(boxes_xyxy_in, from_w=IN_W, from_h=IN_H, to_w=W0, to_h=H0)

            obb_boxes = boxes_xyxy_orig.tolist()
            for b, s, c in zip(obb_boxes, scores_sel, labels_sel):
                cid = int(c)
                item = {
                    "xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    "cofi": float(s),
                    "class_id": cid
                }

                label_name = None
                if CLASS_NAMES and 0 <= cid < len(CLASS_NAMES):
                    label_name = CLASS_NAMES[cid]
                    item["class_name"] = label_name

                # >>> เพิ่ม: เรียกโมเดลข้อความด้วยชื่อคลาสที่ตรวจได้ เพื่อแนบรายละเอียดรายกล่อง
                if label_name:
                    try:
                        item["analysis"] = analyze_label_like_text(label_name, topk=TEXT_TOPK)
                    except Exception as _e:
                        # ถ้าโมเดลข้อความผิดพลาด ไม่ให้พังทั้งคำขอ
                        item["analysis"] = {"error": str(_e), "query": label_name}
                else:
                    item["analysis"] = None

                detections.append(item)

            # --- draw boxes ---
            out_img = draw_boxes_in_place(pil_img, boxes_xyxy_orig, color="red", width=2)
            fmt = (pil_img.format or "PNG").upper()
            mime = {"JPEG":"image/jpeg","JPG":"image/jpeg","PNG":"image/png","WEBP":"image/webp","BMP":"image/bmp","TIFF":"image/tiff"}.get(fmt, "image/png")
            buf = io.BytesIO()
            save_kwargs = {}
            if fmt in ("JPEG","JPG"):
                save_kwargs.update(dict(quality=95, subsampling=0, optimize=True))
                out_img = out_img.convert("RGB")
            out_img.save(buf, format=fmt, **save_kwargs)
            image_b64 = f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

            # --- debug (ชั่วคราว) ---
            try:
                print(f"[DETECT] in_shape={in_shape}, IS_CHW={IS_CHW}, LETTERBOX={LETTERBOX}, COLOR={MODEL_COLOR}")
                print(f"[DETECT] scores: n={scores.size}, max={float(scores.max()) if scores.size else 'NA'} min={float(scores.min()) if scores.size else 'NA'} kept={len(detections)}")
            except Exception:
                pass

        # >>> เพิ่ม: ถ้าไม่ได้ส่งข้อความมา แต่มีภาพและมี detection
        #     ให้สรุป text_result จากกล่องที่มีความมั่นใจสูงสุด
        if (not had_text) and len(detections) > 0:
            try:
                best_idx = int(np.argmax([d.get("cofi", 0.0) for d in detections]))
                best_det = detections[best_idx]
                best_label = best_det.get("class_name") or str(best_det.get("class_id"))
                text_like = analyze_label_like_text(best_label, topk=TEXT_TOPK) or {}
                text_like["source"] = "from_image_top1"
                text_result = text_like
            except Exception as _e:
                # fallback เงียบ ๆ ถ้าเกิดปัญหา จะไม่ทำให้ทั้งคำขอล้ม
                text_result = text_result or {"error": str(_e)}

        # =============== TEXT: analysis + leaflet details ===============
        if had_text:
            text_result = analyze_text(user_text, topk=TEXT_TOPK)

        # Fallback: ถ้าผู้ใช้ส่งข้อความมาแต่ไม่พบข้อมูลใด ๆ ให้แจ้งผลชัดเจน
        if had_text:
            if not isinstance(text_result, dict):
                text_result = {"query": user_text, "error": "ไม่มีข้อมูลเกี่ยวกับตัวยานี้"}
            else:
                no_top = not text_result.get("top")
                no_det = not text_result.get("details")
                no_err = not text_result.get("error")
                if no_top and no_det and no_err:
                    text_result["error"] = "ไม่มีข้อมูลเกี่ยวกับตัวยานี้"

        # response
        return jsonify({
            "ok": True,
            "had_image": had_image,
            "had_text": had_text,
            "obb_boxes": obb_boxes,
            "detections": detections,
            "image_base64": image_b64,
            "text_result": text_result
        }), 200

    except Exception as e:
        print("ERROR /predict:", repr(e))
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
