import os
import json
import re
import difflib
import urllib.request
import urllib.parse
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("looksmax-hub")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = FastAPI(title="Looksmax Hub - Workout & Macro Tracker")

# ================= 1. NUTRITION ENGINE (STRICT MULTI-WORD MATCHING) =================
DB_FILE = os.path.join(os.path.dirname(__file__), "foods_db.json")

def load_food_database() -> Dict[str, Any]:
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"foods_db.json okuma hatası: {e}")
    return {}

LOCAL_FOOD_DB = load_food_database()

UNIT_GRAM_MAP = {
    "olcek": 30.0,
    "scoop": 30.0,
    "dilim": 28.0,
    "kase": 200.0,
    "tabak": 250.0,
    "porsiyon": 200.0,
    "kasik": 15.0,
    "yemek kasigi": 15.0,
    "tatli kasigi": 5.0,
    "cay kasigi": 3.0,
    "bardak": 200.0,
    "su bardagi": 200.0,
    "avuc": 30.0
}

TYPO_CORRECTIONS = {
    "psirnc": "pirinc",
    "psirinc": "pirinc",
    "pirinç": "pirinc",
    "pırınc": "pirinc",
    "tavk": "tavuk",
    "kanaat": "kanat",
    "kasarlı": "kasarli",
    "karısık": "karisik",
    "ölçek": "olcek",
    "kaşık": "kasik",
    "ekmeği": "ekmegi",
    "ekmek": "ekmegi",
    "buğday": "bugday",
    "bugday": "bugday"
}

def normalize_turkish(text: str) -> str:
    t = text.lower()
    t = t.replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
    t = re.sub(r'[^a-z0-9\s.,]', ' ', t)
    words = t.split()
    corrected_words = [TYPO_CORRECTIONS.get(w, w) for w in words]
    return " ".join(corrected_words).strip()

def search_local_food(query: str):
    if not query:
        return None
    norm_q = normalize_turkish(query)

    # 1. Uzun Öbek Eşleşmesi (En uzun kelime grubundan başla)
    sorted_keys = sorted(LOCAL_FOOD_DB.keys(), key=lambda x: len(x), reverse=True)
    for key in sorted_keys:
        if key in norm_q:
            return {**LOCAL_FOOD_DB[key], "matched_key": key}

    # 2. Tam Eşleşme
    if norm_q in LOCAL_FOOD_DB:
        return {**LOCAL_FOOD_DB[norm_q], "matched_key": norm_q}

    # 3. Yüksek Benzerlikli Fuzzy Match
    matches = difflib.get_close_matches(norm_q, LOCAL_FOOD_DB.keys(), n=1, cutoff=0.85)
    if matches:
        return {**LOCAL_FOOD_DB[matches[0]], "matched_key": matches[0]}

    return None

def fetch_open_food_facts(query: str):
    try:
        clean_q = urllib.parse.quote(query)
        url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={clean_q}&search_simple=1&action=process&json=1&page_size=1"
        req = urllib.request.Request(url, headers={"User-Agent": "LooksmaxHub-NutritionEngine/2.0"})
        with urllib.request.urlopen(req, timeout=2.0) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data.get("products") and len(data["products"]) > 0:
                p = data["products"][0]
                nutr = p.get("nutriments", {})
                cal = float(nutr.get("energy-kcal_100g") or nutr.get("energy-kcal") or 0)
                pro = float(nutr.get("proteins_100g") or nutr.get("proteins") or 0)
                carb = float(nutr.get("carbohydrates_100g") or nutr.get("carbohydrates") or 0)
                fat = float(nutr.get("fat_100g") or nutr.get("fat") or 0)
                name = p.get("product_name_tr") or p.get("product_name") or query.title()
                if cal > 0:
                    return {"name": name, "cal": cal, "pro": pro, "carb": carb, "fat": fat, "unit": "g"}
    except Exception as e:
        logger.warning(f"OpenFoodFacts API hatası ({query}): {e}")
    return None

def parse_and_calculate_meal(user_text: str) -> Optional[Dict[str, Any]]:
    norm_text = normalize_turkish(user_text)
    
    unit_pattern = r'(?:kg|kilo|kilogram|g|gr|gram|adet|tane|dilim|olcek|scoop|kase|tabak|porsiyon|kasik|bardak)'
    norm_text = re.sub(rf'(\d+(?:[.,]\d+)?\s*{unit_pattern}?)', r',\1', norm_text)
    raw_tokens = [t.strip() for t in re.split(r'[,+\n]|(?:\s+ve\s+)', norm_text) if t.strip()]

    total_cal = 0.0
    total_pro = 0.0
    total_carb = 0.0
    total_fat = 0.0
    items_summary_list = []

    for token in raw_tokens:
        if not token:
            continue

        qty = None
        unit = None

        m_num = re.search(rf'^(\d+(?:[.,]\d+)?)\s*({unit_pattern})?', token)
        if m_num:
            qty = float(m_num.group(1).replace(",", "."))
            unit = m_num.group(2)
            food_query = token[m_num.end():].strip()
        else:
            food_query = token.strip()

        if not food_query:
            continue

        food_data = search_local_food(food_query)
        
        if food_data:
            multiplier_gram = 0.0
            display_title = ""

            if unit in UNIT_GRAM_MAP:
                multiplier_gram = (qty or 1.0) * UNIT_GRAM_MAP[unit]
                display_title = f"{int(qty or 1)} {unit.title()} {food_data['name']} ({int(multiplier_gram)}g)"
            elif unit in ["kg", "kilo", "kilogram"]:
                multiplier_gram = (qty or 1.0) * 1000.0
                display_title = f"{int(multiplier_gram)}g {food_data['name']}"
            elif unit in ["g", "gr", "gram"]:
                multiplier_gram = qty or 100.0
                display_title = f"{int(multiplier_gram)}g {food_data['name']}"
            elif unit in ["adet", "tane"] or (qty is not None and qty < 15 and food_data.get("unit") == "item"):
                count = qty or 1.0
                c = food_data["cal"] * count
                p = food_data["pro"] * count
                cb = food_data["carb"] * count
                f = food_data["fat"] * count
                total_cal += c
                total_pro += p
                total_carb += cb
                total_fat += f
                items_summary_list.append(f"{int(count)} Adet {food_data['name']}")
                continue
            elif "serving_weight" in food_data:
                count = qty or 1.0
                multiplier_gram = count * food_data["serving_weight"]
                s_unit = food_data.get("serving_unit", "porsiyon")
                display_title = f"{int(count)} {s_unit.title()} {food_data['name']} ({int(multiplier_gram)}g)"
            elif qty and qty >= 15:
                multiplier_gram = qty
                display_title = f"{int(multiplier_gram)}g {food_data['name']}"
            else:
                multiplier_gram = 100.0
                display_title = f"100g {food_data['name']}"

            ratio = multiplier_gram / 100.0
            total_cal += food_data["cal"] * ratio
            total_pro += food_data["pro"] * ratio
            total_carb += food_data["carb"] * ratio
            total_fat += food_data["fat"] * ratio
            items_summary_list.append(display_title)
        else:
            api_data = fetch_open_food_facts(food_query)
            if api_data:
                grams = 100.0
                if unit in UNIT_GRAM_MAP:
                    grams = (qty or 1.0) * UNIT_GRAM_MAP[unit]
                elif unit in ["kg", "kilo", "kilogram"]:
                    grams = (qty or 1.0) * 1000.0
                elif qty and qty >= 15:
                    grams = qty
                elif qty:
                    grams = qty * 100.0

                ratio = grams / 100.0
                total_cal += api_data["cal"] * ratio
                total_pro += api_data["pro"] * ratio
                total_carb += api_data["carb"] * ratio
                total_fat += api_data["fat"] * ratio
                items_summary_list.append(f"{int(grams)}g {api_data['name']}")
            else:
                grams = (qty or 1.0) * 1000.0 if unit in ["kg", "kilo", "kilogram"] else (qty if (qty and qty >= 15) else 100.0)
                total_cal += grams * 1.5
                total_pro += grams * 0.10
                total_carb += grams * 0.18
                total_fat += grams * 0.05
                items_summary_list.append(f"{int(grams)}g {food_query.title()}")

    if total_cal > 0:
        return {
            "food_name": " + ".join(items_summary_list),
            "items_summary": ", ".join(items_summary_list),
            "calories": round(total_cal),
            "protein": round(total_pro),
            "carbs": round(total_carb),
            "fat": round(total_fat)
        }
    return None

# ================= 2. FASTAPI SCHEMAS & ROUTES =================
class ChatInput(BaseModel):
    user_message: str
    image_base64: Optional[str] = None
    workout_summary: Optional[str] = ""
    history: List[dict] = []

class NutritionChatInput(BaseModel):
    user_message: str
    image_base64: Optional[str] = None
    daily_summary: Optional[str] = ""
    history: List[dict] = []

HTML_INTERFACE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Looksmax Hub - Antrenman & Beslenme</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background-color: #0b0d12; color: #e5e7eb; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

        .auth-overlay { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.92); display: flex; justify-content: center; align-items: center; z-index: 9999; backdrop-filter: blur(8px); }
        .auth-box { background: #131722; border: 1px solid #1f293d; padding: 36px; border-radius: 18px; width: 360px; display: flex; flex-direction: column; gap: 14px; box-shadow: 0 12px 40px rgba(0,242,254,0.18); }
        .auth-box h2 { font-size: 1.35rem; font-weight: 800; color: #00f2fe; text-align: center; }
        .auth-box input { background: #0a0c10; border: 1px solid #2b354d; color: #fff; padding: 12px 14px; border-radius: 9px; font-size: 0.9rem; outline: none; }
        .auth-box input:focus { border-color: #00f2fe; }
        .auth-box button { background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); color: #000; border: none; font-weight: 800; padding: 13px; border-radius: 9px; cursor: pointer; }
        .auth-toggle { font-size: 0.8rem; color: #9ca3af; text-align: center; cursor: pointer; }
        .auth-toggle b { color: #00f2fe; }

        .header-bar { height: 60px; background: #0f121a; border-bottom: 1px solid #1c2230; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; flex-shrink: 0; }
        .brand { font-size: 1.15rem; font-weight: 800; color: #00f2fe; display: flex; align-items: center; gap: 8px; cursor: pointer; }
        .back-hub-btn { background: #1a202c; color: #00f2fe; border: 1px solid #28334a; padding: 6px 14px; border-radius: 8px; font-size: 0.8rem; font-weight: 700; cursor: pointer; display: none; }
        .back-hub-btn:hover { background: #232b3b; }
        .user-section { display: flex; align-items: center; gap: 12px; }
        .user-tag { font-size: 0.8rem; background: #161c26; padding: 6px 12px; border-radius: 8px; color: #10b981; border: 1px solid #263245; }
        .logout-btn { background: none; border: none; color: #ef4444; cursor: pointer; font-size: 0.8rem; font-weight: 600; }

        .content-container { flex: 1; display: flex; justify-content: center; align-items: center; overflow: hidden; position: relative; }
        .view-panel { display: none; width: 100%; height: 100%; padding: 20px; }
        .view-panel.active { display: flex; }

        #hubView { justify-content: center; align-items: center; flex-direction: column; gap: 28px; }
        .hub-title { text-align: center; }
        .hub-title h1 { font-size: 2.2rem; font-weight: 800; color: #fff; margin-bottom: 6px; }
        .hub-title p { font-size: 0.95rem; color: #9ca3af; }
        
        .hub-grid { display: flex; gap: 20px; max-width: 1050px; width: 100%; justify-content: center; }
        .hub-card { flex: 1; background: #131722; border: 1px solid #222c3f; border-radius: 20px; padding: 32px 24px; display: flex; flex-direction: column; justify-content: space-between; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 8px 24px rgba(0,0,0,0.4); text-align: left; }
        .hub-card:hover { transform: translateY(-6px); border-color: #00f2fe; box-shadow: 0 12px 35px rgba(0,242,254,0.22); }
        .card-icon { font-size: 2.3rem; margin-bottom: 14px; }
        .card-heading { font-size: 1.3rem; font-weight: 800; color: #fff; margin-bottom: 8px; }
        .card-desc { font-size: 0.82rem; color: #9ca3af; line-height: 1.5; margin-bottom: 20px; }
        .card-action { align-self: flex-start; background: #1a2232; color: #00f2fe; border: 1px solid #2d3b54; padding: 9px 16px; border-radius: 10px; font-weight: 700; font-size: 0.82rem; transition: 0.2s; }
        .hub-card:hover .card-action { background: #00f2fe; color: #000; }

        #coachView { flex-direction: column; max-width: 950px; }
        .chat-container { flex: 1; display: flex; flex-direction: column; background: #131722; border-radius: 16px; border: 1px solid #1f2738; overflow: hidden; }
        .messages { flex: 1; overflow-y: auto; padding: 22px; display: flex; flex-direction: column; gap: 14px; }
        .msg { max-width: 82%; padding: 13px 17px; border-radius: 14px; font-size: 0.92rem; line-height: 1.5; word-wrap: break-word; }
        .msg.user { align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 3px; }
        .msg.coach { align-self: flex-start; background: #1a2130; border: 1px solid #283449; border-bottom-left-radius: 3px; }
        .msg img.preview-img { max-width: 240px; border-radius: 8px; margin-bottom: 8px; display: block; }
        
        .preview-box { display: none; padding: 8px 16px; background: #0d1017; align-items: center; gap: 10px; border-top: 1px solid #1c2230; }
        .preview-box img { height: 45px; border-radius: 6px; border: 1px solid #00f2fe; }
        .preview-box button { background: #ef4444; color: white; border: none; border-radius: 50%; width: 22px; height: 22px; cursor: pointer; }

        .chat-input-area { padding: 14px 18px; border-top: 1px solid #1c2230; background: #0f121a; display: flex; gap: 10px; align-items: center; }
        .chat-input { flex: 1; background: #181f2c; border: 1px solid #29364d; color: #fff; padding: 12px 16px; border-radius: 10px; font-size: 0.9rem; outline: none; }
        .chat-input:focus { border-color: #00f2fe; }
        .file-btn { background: #181f2c; border: 1px solid #29364d; color: #00f2fe; padding: 10px 14px; border-radius: 10px; cursor: pointer; font-size: 1.1rem; }
        input[type="file"] { display: none; }
        .send-btn { background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); color: #000; border: none; font-weight: 800; padding: 12px 24px; border-radius: 10px; cursor: pointer; }

        #overloadView { gap: 20px; max-width: 1350px; }
        .overload-col-left { width: 46%; display: flex; flex-direction: column; gap: 16px; height: 100%; }
        .overload-col-right { width: 54%; display: flex; flex-direction: column; gap: 16px; height: 100%; }

        .panel-card { background: #131722; border: 1px solid #1f2738; border-radius: 16px; padding: 18px; display: flex; flex-direction: column; gap: 12px; }
        .panel-header { font-size: 0.95rem; font-weight: 800; color: #00f2fe; display: flex; justify-content: space-between; align-items: center; }
        .badge-cyan { font-size: 0.75rem; background: rgba(0, 242, 254, 0.1); color: #00f2fe; border: 1px solid rgba(0, 242, 254, 0.3); padding: 4px 8px; border-radius: 6px; font-weight: 600; }

        .days-tab-bar { display: flex; gap: 6px; overflow-x: auto; padding-bottom: 4px; }
        .day-tab-btn { flex: 1; min-width: 48px; background: #0a0c10; border: 1px solid #1f2738; border-radius: 10px; padding: 8px 4px; color: #9ca3af; font-size: 0.75rem; font-weight: 700; cursor: pointer; text-align: center; transition: 0.2s; }
        .day-tab-btn .tab-sub { font-size: 0.65rem; color: #6b7280; display: block; margin-top: 2px; }
        .day-tab-btn:hover { border-color: #2b3a52; color: #fff; }
        .day-tab-btn.active { background: #172133; border-color: #00f2fe; color: #00f2fe; }
        .day-tab-btn.active .tab-sub { color: #38bdf8; }

        .empty-day-box { background: #0a0c10; border: 1px dashed #242f44; border-radius: 12px; padding: 36px 16px; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; }
        .empty-day-icon { font-size: 2rem; }
        .empty-day-title { font-size: 1.05rem; font-weight: 800; color: #e5e7eb; }
        .empty-day-desc { font-size: 0.78rem; color: #6b7280; max-width: 250px; line-height: 1.4; }

        .input-form { display: flex; flex-direction: column; gap: 10px; }
        .input-form input { background: #0a0c10; border: 1px solid #2b354d; color: #fff; padding: 11px 12px; border-radius: 8px; font-size: 0.85rem; outline: none; width: 100%; }
        .input-form input:focus { border-color: #00f2fe; }
        
        .form-grid-2x2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; width: 100%; }
        .btn-log { background: #00f2fe; color: #000; border: none; font-weight: 800; padding: 12px; border-radius: 8px; cursor: pointer; margin-top: 4px; }

        .history-list { flex: 1; overflow-y: auto; max-height: 380px; display: flex; flex-direction: column; gap: 8px; padding-right: 4px; }
        .log-item { display: flex; justify-content: space-between; align-items: center; background: #0a0c10; padding: 10px 14px; border-radius: 9px; font-size: 0.85rem; border: 1px solid #1c2230; }
        .log-item .set-badge { background: #1e293b; color: #00f2fe; padding: 2px 7px; border-radius: 5px; font-weight: 700; font-size: 0.75rem; margin-right: 6px; }
        .log-item .ex-title { font-weight: 700; color: #fff; }
        .log-item .ex-val { color: #38bdf8; font-weight: 700; }
        .log-item button { background: none; border: none; color: #ef4444; cursor: pointer; font-size: 0.85rem; padding: 2px 4px; }

        .chart-box { flex: 1; min-height: 320px; position: relative; }

        #nutritionView { gap: 20px; max-width: 1350px; }
        .macro-stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
        .macro-card { background: #0a0c10; border: 1px solid #1c2230; padding: 14px; border-radius: 12px; text-align: center; }
        .macro-val { font-size: 1.35rem; font-weight: 800; margin-top: 4px; }
        .macro-label { font-size: 0.72rem; font-weight: 700; color: #9ca3af; text-transform: uppercase; }

        .macro-c-cal { color: #f59e0b; }
        .macro-c-pro { color: #10b981; }
        .macro-c-carb { color: #00f2fe; }
        .macro-c-fat { color: #ec4899; }

        .meal-items-subtext { font-size: 0.75rem; color: #38bdf8; margin-top: 4px; font-weight: 500; }

        @media (max-width: 850px) {
            body { overflow: auto; height: auto; }
            .hub-grid { flex-direction: column; }
            .content-container { height: auto; overflow: visible; }
            .view-panel { height: auto; flex-direction: column !important; }
            .overload-col-left, .overload-col-right { width: 100%; }
            #coachView { height: 80vh; }
            .macro-stat-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>

    <div class="auth-overlay" id="authOverlay">
        <div class="auth-box">
            <h2 id="authTitle">⚡ LOOKSMAX PRO</h2>
            <input type="text" id="authUsername" placeholder="Kullanıcı Adı" />
            <input type="password" id="authPassword" placeholder="Şifre" />
            <button id="authSubmitBtn" onclick="handleAuthSubmit()">Giriş Yap</button>
            <div class="auth-toggle" id="authToggle" onclick="toggleAuthMode()">Hesabın yok mu? <b>Kayıt Ol</b></div>
        </div>
    </div>

    <div class="header-bar">
        <div style="display:flex; align-items:center; gap:12px;">
            <div class="brand" onclick="openView('hub')">⚡ LOOKSMAX HUB</div>
            <button class="back-hub-btn" id="backHubBtn" onclick="openView('hub')">← Ana Menü</button>
        </div>
        <div class="user-section">
            <div class="user-tag" id="activeUserName">Giriş Yapılmadı</div>
            <button class="logout-btn" onclick="logout()">Çıkış</button>
        </div>
    </div>

    <div class="content-container">

        <div class="view-panel active" id="hubView">
            <div class="hub-title">
                <h1>Modülünü Seç Kral 🦍</h1>
                <p>Neyi yönetmek veya geliştirmek istiyorsan tıkla ve başla.</p>
            </div>
            <div class="hub-grid">
                <div class="hub-card" onclick="openView('coach')">
                    <div>
                        <div class="card-icon">🤖</div>
                        <div class="card-heading">AI Koç & Vision</div>
                        <div class="card-desc">Hipertrofi taktikleri, form kontrolü ve detaylı fizik değerlendirmesi yap.</div>
                    </div>
                    <div class="card-action">Koçla Konuş →</div>
                </div>

                <div class="hub-card" onclick="openView('overload')">
                    <div>
                        <div class="card-icon">📈</div>
                        <div class="card-heading">Progressive Overload</div>
                        <div class="card-desc">Set ve ağırlıklarını kaydet. Gün gün sekme sekme antrenman ve dinlenme günlerini takip et.</div>
                    </div>
                    <div class="card-action">Overload Takip →</div>
                </div>

                <div class="hub-card" onclick="openView('nutrition')">
                    <div>
                        <div class="card-icon">🥗</div>
                        <div class="card-heading">Haftalık Beslenme & Makro</div>
                        <div class="card-desc">Yediklerini yaz veya fotoğrafını at; gün gün tüm haftalık makro ve kalorilerini takip et.</div>
                    </div>
                    <div class="card-action">Makro Takip →</div>
                </div>
            </div>
        </div>

        <div class="view-panel" id="coachView">
            <div class="chat-container">
                <div class="messages" id="chatBox">
                    <div class="msg coach">Selam kral! Ben senin Looksmax & Overload koçunum. Antrenman taktikleri sorabilir, formunu veya fiziğini değerlendirebilirim.</div>
                </div>

                <div class="preview-box" id="previewBox">
                    <img id="imagePreview" src="" alt="Görsel" />
                    <button onclick="clearImage()">✕</button>
                    <span style="font-size:0.75rem; color:#9ca3af;">Görsel seçildi</span>
                </div>

                <div class="chat-input-area">
                    <label class="file-btn" for="imageInput" title="Fotoğraf Yükle">📷</label>
                    <input type="file" id="imageInput" accept="image/*" onchange="handleImageSelect(event, 'coach')" />
                    <input type="text" class="chat-input" id="userInput" placeholder="Koça danış..." onkeypress="handleKey(event, 'coach')" />
                    <button class="send-btn" id="sendBtn" onclick="sendMessage()">Gönder</button>
                </div>
            </div>
        </div>

        <div class="view-panel" id="overloadView">
            <div class="overload-col-left">
                <div class="panel-card">
                    <div class="panel-header">
                        <span>➕ Set Kaydet</span>
                        <span class="badge-cyan" id="currentWeekDisplay">Haftalık Döngü</span>
                    </div>
                    <div class="input-form">
                        <input type="text" id="exerciseName" placeholder="Hareket Adı (Örn: Incline Dumbbell Press)" list="defaultExercises" />
                        <datalist id="defaultExercises">
                            <option value="Bench Press">
                            <option value="Incline Dumbbell Press">
                            <option value="Squat">
                            <option value="Deadlift">
                            <option value="Chest Supported Row">
                            <option value="Overhead Press">
                            <option value="Lateral Raise">
                            <option value="Pull-up">
                            <option value="Face Pull">
                        </datalist>

                        <div class="form-grid-2x2">
                            <input type="number" id="exerciseSet" placeholder="🔢 Set No (1, 2...)" min="1" value="1" />
                            <input type="number" id="exerciseWeight" placeholder="⚖️ Kilo (kg)" step="0.5" />
                            <input type="number" id="exerciseReps" placeholder="🔁 Tekrar Sayısı" min="1" />
                            <input type="text" id="exerciseDate" placeholder="📅 Tarih" />
                        </div>
                        <button class="btn-log" onclick="addWorkoutLog()">Seti Kaydet</button>
                    </div>
                </div>

                <div class="panel-card" style="flex:1;">
                    <div class="panel-header">
                        <span>🗓️ Antrenman Takvimi</span>
                        <span style="font-size:0.75rem; color:#9ca3af;" id="daySetsBadge">0 Set</span>
                    </div>

                    <div class="days-tab-bar" id="workoutDaysTabBar"></div>
                    <div class="history-list" id="dayHistoryList"></div>
                </div>
            </div>

            <div class="overload-col-right">
                <div class="panel-card" style="height: 100%;">
                    <div class="panel-header">
                        <span>📊 Hareket Gelişim Grafiği</span>
                        <select id="chartExerciseSelect" onchange="updateChart()" style="background:#0a0c10; border:1px solid #2b354d; color:#00f2fe; padding:6px 12px; border-radius:7px; font-weight:700; outline:none;"></select>
                    </div>
                    <div class="chart-box">
                        <canvas id="progressionChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <div class="view-panel" id="nutritionView">
            <div class="overload-col-left">
                <div class="chat-container">
                    <div class="messages" id="nutriChatBox">
                        <div class="msg coach">Afiyet olsun kral! Ne yediysen yaz (örn: <i>"4 tam buğday ekmeği"</i>, <i>"1 ölçek protein tozu"</i>, <i>"300g tavuk 150g pirinc"</i>); tüm makrolarını tam hesaplayıp eklerim.</div>
                    </div>

                    <div class="preview-box" id="nutriPreviewBox">
                        <img id="nutriImagePreview" src="" alt="Görsel" />
                        <button onclick="clearNutriImage()">✕</button>
                        <span style="font-size:0.75rem; color:#9ca3af;">Yemek görseli seçildi</span>
                    </div>

                    <div class="chat-input-area">
                        <label class="file-btn" for="nutriImageInput" title="Yemek Fotoğrafı">📷</label>
                        <input type="file" id="nutriImageInput" accept="image/*" onchange="handleImageSelect(event, 'nutri')" />
                        <input type="text" class="chat-input" id="nutriUserInput" placeholder="Yediklerini yaz (örn: 4 dilim tam bugday ekmegi, 1 olcek protein tozu...)" onkeypress="handleKey(event, 'nutri')" />
                        <button class="send-btn" id="nutriSendBtn" onclick="sendNutriMessage()">Ekle</button>
                    </div>
                </div>
            </div>

            <div class="overload-col-right">
                <div class="panel-card">
                    <div class="panel-header">
                        <span>🔥 Makro Durumu</span>
                        <span class="badge-cyan" id="nutriSelectedDateDisplay">Seçili Gün</span>
                    </div>

                    <div class="days-tab-bar" id="nutriDaysTabBar"></div>

                    <div class="macro-stat-grid" style="margin-top: 6px;">
                        <div class="macro-card">
                            <div class="macro-label">Kalori</div>
                            <div class="macro-val macro-c-cal" id="totCalories">0 kcal</div>
                        </div>
                        <div class="macro-card">
                            <div class="macro-label">Protein</div>
                            <div class="macro-val macro-c-pro" id="totProtein">0g</div>
                        </div>
                        <div class="macro-card">
                            <div class="macro-label">Karbonhidrat</div>
                            <div class="macro-val macro-c-carb" id="totCarbs">0g</div>
                        </div>
                        <div class="macro-card">
                            <div class="macro-label">Yağ</div>
                            <div class="macro-val macro-c-fat" id="totFat">0g</div>
                        </div>
                    </div>
                </div>

                <div class="panel-card" style="flex:1;">
                    <div class="panel-header">
                        <span>🍽️ Seçili Günün Öğünleri</span>
                        <button onclick="clearSelectedDayMeals()" style="background:none; border:none; color:#ef4444; font-size:0.75rem; cursor:pointer; font-weight:700;">Bu Günü Sıfırla</button>
                    </div>
                    <div class="history-list" id="mealsList"></div>
                </div>
            </div>
        </div>

    </div>

    <script>
        function getMondayOfWeek(d) {
            d = new Date(d);
            var day = d.getDay(),
                diff = d.getDate() - day + (day === 0 ? -6 : 1);
            var mon = new Date(d.setDate(diff));
            mon.setHours(0, 0, 0, 0);
            return mon;
        }

        const mondayObj = getMondayOfWeek(new Date());
        const currentWeekKey = mondayObj.toISOString().split('T')[0];
        const todayKey = new Date().toLocaleDateString('tr-TR', { day: '2-digit', month: '2-digit', year: 'numeric' });

        const dayNames = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"];
        const weekDaysData = [];

        for (let i = 0; i < 7; i++) {
            const d = new Date(mondayObj);
            d.setDate(mondayObj.getDate() + i);
            const fullDate = d.toLocaleDateString('tr-TR', { day: '2-digit', month: '2-digit', year: 'numeric' });
            const shortDate = d.toLocaleDateString('tr-TR', { day: '2-digit', month: '2-digit' });
            weekDaysData.push({
                index: i,
                dayName: dayNames[i],
                fullDate: fullDate,
                shortDate: shortDate
            });
        }

        let selectedWorkoutDayIdx = weekDaysData.findIndex(item => item.fullDate === todayKey);
        if (selectedWorkoutDayIdx === -1) selectedWorkoutDayIdx = 0;

        let selectedNutriDayIdx = weekDaysData.findIndex(item => item.fullDate === todayKey);
        if (selectedNutriDayIdx === -1) selectedNutriDayIdx = 0;

        function getStorageUsers() {
            return JSON.parse(localStorage.getItem("app_registered_users") || "{}");
        }
        function saveStorageUsers(users) {
            localStorage.setItem("app_registered_users", JSON.stringify(users));
        }

        function getUserWeeklyLogs(username) {
            const allWeeks = JSON.parse(localStorage.getItem("user_weeks_" + username) || "{}");
            return allWeeks[currentWeekKey] || [];
        }
        function saveUserWeeklyLogs(username, logs) {
            const allWeeks = JSON.parse(localStorage.getItem("user_weeks_" + username) || "{}");
            allWeeks[currentWeekKey] = logs;
            localStorage.setItem("user_weeks_" + username, JSON.stringify(allWeeks));
        }

        function getUserWeeklyNutrition(username) {
            const allWeeks = JSON.parse(localStorage.getItem("user_nutri_weeks_" + username) || "{}");
            return allWeeks[currentWeekKey] || {};
        }
        function saveUserWeeklyNutrition(username, nutriData) {
            const allWeeks = JSON.parse(localStorage.getItem("user_nutri_weeks_" + username) || "{}");
            allWeeks[currentWeekKey] = nutriData;
            localStorage.setItem("user_nutri_weeks_" + username, JSON.stringify(allWeeks));
        }

        let currentUser = JSON.parse(localStorage.getItem("active_user") || "null");
        let isRegisterMode = false;
        let weeklyLogs = [];
        let weeklyNutrition = {};
        let chartInstance = null;

        document.getElementById("exerciseDate").value = weekDaysData[selectedWorkoutDayIdx].fullDate;
        document.getElementById("currentWeekDisplay").innerText = "Hafta: " + currentWeekKey;

        function openView(viewName) {
            document.querySelectorAll(".view-panel").forEach(p => p.classList.remove("active"));
            const target = document.getElementById(viewName + "View");
            if (target) target.classList.add("active");

            document.getElementById("backHubBtn").style.display = (viewName === 'hub') ? 'none' : 'block';

            if (viewName === 'overload') {
                setTimeout(updateChart, 150);
            }
            if (viewName === 'nutrition') {
                renderNutriDayTabs();
                renderSelectedDayNutrition();
            }
        }

        function checkAuth() {
            if (!currentUser) {
                document.getElementById("authOverlay").style.display = "flex";
            } else {
                document.getElementById("authOverlay").style.display = "none";
                document.getElementById("activeUserName").innerText = "👤 " + currentUser.username;
                loadUserWorkouts();
                loadUserNutrition();
            }
        }

        function toggleAuthMode() {
            isRegisterMode = !isRegisterMode;
            document.getElementById("authTitle").innerText = isRegisterMode ? "⚡ KAYIT OL" : "⚡ GİRİŞ YAP";
            document.getElementById("authSubmitBtn").innerText = isRegisterMode ? "Hesap Oluştur" : "Giriş Yap";
            document.getElementById("authToggle").innerHTML = isRegisterMode ? "Zaten hesabın var mı? <b>Giriş Yap</b>" : "Hesabın yok mu? <b>Kayıt Ol</b>";
        }

        function handleAuthSubmit() {
            const u = document.getElementById("authUsername").value.trim();
            const p = document.getElementById("authPassword").value.trim();
            if (!u || !p) return alert("Kullanıcı adı ve şifre gir!");

            const allUsers = getStorageUsers();

            if (isRegisterMode) {
                if (allUsers[u]) return alert("Bu kullanıcı adı zaten var!");
                allUsers[u] = p;
                saveStorageUsers(allUsers);
                currentUser = { username: u };
                localStorage.setItem("active_user", JSON.stringify(currentUser));
                checkAuth();
            } else {
                if (!allUsers[u] || allUsers[u] !== p) {
                    if (Object.keys(allUsers).length === 0 || !allUsers[u]) {
                        allUsers[u] = p;
                        saveStorageUsers(allUsers);
                        currentUser = { username: u };
                        localStorage.setItem("active_user", JSON.stringify(currentUser));
                        checkAuth();
                        return;
                    }
                    return alert("Kullanıcı adı veya şifre hatalı!");
                }
                currentUser = { username: u };
                localStorage.setItem("active_user", JSON.stringify(currentUser));
                checkAuth();
            }
        }

        function logout() {
            localStorage.removeItem("active_user");
            currentUser = null;
            location.reload();
        }

        function loadUserWorkouts() {
            if (!currentUser) return;
            weeklyLogs = getUserWeeklyLogs(currentUser.username);
            populateDropdown();
            renderWorkoutDayTabs();
            renderSelectedWorkoutDayLogs();
            updateChart();
        }

        function renderWorkoutDayTabs() {
            const bar = document.getElementById("workoutDaysTabBar");
            if (!bar) return;
            bar.innerHTML = "";

            weekDaysData.forEach((d, idx) => {
                const isActive = (idx === selectedWorkoutDayIdx) ? "active" : "";
                bar.innerHTML += `
                    <button class="day-tab-btn ${isActive}" onclick="selectWorkoutDayTab(${idx})">
                        ${d.dayName}
                        <span class="tab-sub">${d.shortDate}</span>
                    </button>
                `;
            });
        }

        function selectWorkoutDayTab(idx) {
            selectedWorkoutDayIdx = idx;
            document.getElementById("exerciseDate").value = weekDaysData[idx].fullDate;
            renderWorkoutDayTabs();
            renderSelectedWorkoutDayLogs();
        }

        function renderSelectedWorkoutDayLogs() {
            const currentDay = weekDaysData[selectedWorkoutDayIdx];
            const list = document.getElementById("dayHistoryList");
            if (!list) return;
            list.innerHTML = "";

            const dayLogs = weeklyLogs.filter(item => item.date === currentDay.fullDate);
            document.getElementById("daySetsBadge").innerText = `${dayLogs.length} Set`;

            if (dayLogs.length === 0) {
                list.innerHTML = `
                    <div class="empty-day-box">
                        <div class="empty-day-icon">😴</div>
                        <div class="empty-day-title">Dinlenme Günü (Off Day)</div>
                        <div class="empty-day-desc">${currentDay.fullDate} tarihinde henüz kayıtlı bir setin yok kral.</div>
                    </div>
                `;
                return;
            }

            dayLogs.forEach(item => {
                list.innerHTML += `
                    <div class="log-item">
                        <div>
                            <span class="set-badge">${item.set_num || 1}. SET</span>
                            <span class="ex-title">${item.exercise}</span>: 
                            <span class="ex-val">${item.weight} kg</span> × ${item.reps} tkr
                        </div>
                        <button onclick="deleteWorkout(${item.id})" title="Sil">Sil</button>
                    </div>
                `;
            });
        }

        function addWorkoutLog() {
            if (!currentUser) return;
            const name = document.getElementById("exerciseName").value.trim();
            const setVal = document.getElementById("exerciseSet").value.trim();
            const weightVal = document.getElementById("exerciseWeight").value.trim();
            const repsVal = document.getElementById("exerciseReps").value.trim();
            const dateVal = document.getElementById("exerciseDate").value.trim() || weekDaysData[selectedWorkoutDayIdx].fullDate;

            if (!name) return alert("Lütfen hareket adını gir kral!");
            if (!weightVal || isNaN(Number(weightVal))) return alert("Lütfen ağırlığı (kg) gir kral!");
            if (!repsVal || isNaN(Number(repsVal))) return alert("Lütfen tekrar sayısını gir kral!");

            const setNum = parseInt(setVal, 10) || 1;
            const weight = parseFloat(weightVal);
            const reps = parseInt(repsVal, 10);

            weeklyLogs.push({ id: Date.now(), exercise: name, set_num: setNum, weight: weight, reps: reps, date: dateVal });
            saveUserWeeklyLogs(currentUser.username, weeklyLogs);

            document.getElementById("exerciseSet").value = setNum + 1;
            document.getElementById("exerciseWeight").value = "";
            document.getElementById("exerciseReps").value = "";
            loadUserWorkouts();
        }

        function deleteWorkout(id) {
            weeklyLogs = weeklyLogs.filter(item => item.id !== id);
            saveUserWeeklyLogs(currentUser.username, weeklyLogs);
            loadUserWorkouts();
        }

        function populateDropdown() {
            const select = document.getElementById("chartExerciseSelect");
            if (!select) return;
            const currentSelected = select.value;
            const unique = [...new Set(weeklyLogs.map(item => item.exercise))];

            select.innerHTML = "";
            if (unique.length === 0) {
                select.innerHTML = "<option value=''>Kayıt Yok</option>";
                return;
            }
            unique.forEach(ex => {
                const opt = document.createElement("option");
                opt.value = ex;
                opt.innerText = ex;
                select.appendChild(opt);
            });
            if (unique.includes(currentSelected)) select.value = currentSelected;
            else select.value = unique[0];
        }

        function updateChart() {
            const select = document.getElementById("chartExerciseSelect");
            if (!select) return;
            const selectedEx = select.value;
            const filtered = weeklyLogs.filter(item => item.exercise === selectedEx);

            const labels = filtered.map((item, idx) => `${item.date} (${item.set_num || idx+1}.Set)`);
            const weights = filtered.map(item => item.weight);
            const reps = filtered.map(item => item.reps);

            const canvas = document.getElementById('progressionChart');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (chartInstance) chartInstance.destroy();

            chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Ağırlık (kg)',
                            data: weights,
                            borderColor: '#00f2fe',
                            backgroundColor: 'rgba(0, 242, 254, 0.12)',
                            borderWidth: 3,
                            yAxisID: 'y',
                            tension: 0.35,
                            fill: true,
                            pointBackgroundColor: '#00f2fe',
                            pointRadius: 4
                        },
                        {
                            label: 'Tekrar',
                            data: reps,
                            borderColor: '#f59e0b',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            yAxisID: 'y1',
                            tension: 0.35,
                            pointBackgroundColor: '#f59e0b',
                            pointRadius: 4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: '#1a2230' }, ticks: { color: '#9ca3af' } },
                        y: { type: 'linear', position: 'left', grid: { color: '#1a2230' }, ticks: { color: '#00f2fe' } },
                        y1: { type: 'linear', position: 'right', grid: { display: false }, ticks: { color: '#f59e0b' } }
                    },
                    plugins: { legend: { labels: { color: '#e5e7eb', font: { size: 11, weight: 'bold' } } } }
                }
            });
        }

        function loadUserNutrition() {
            if (!currentUser) return;
            weeklyNutrition = getUserWeeklyNutrition(currentUser.username);
            renderNutriDayTabs();
            renderSelectedDayNutrition();
        }

        function renderNutriDayTabs() {
            const bar = document.getElementById("nutriDaysTabBar");
            if (!bar) return;
            bar.innerHTML = "";

            weekDaysData.forEach((d, idx) => {
                const isActive = (idx === selectedNutriDayIdx) ? "active" : "";
                bar.innerHTML += `
                    <button class="day-tab-btn ${isActive}" onclick="selectNutriDayTab(${idx})">
                        ${d.dayName}
                        <span class="tab-sub">${d.shortDate}</span>
                    </button>
                `;
            });
        }

        function selectNutriDayTab(idx) {
            selectedNutriDayIdx = idx;
            renderNutriDayTabs();
            renderSelectedDayNutrition();
        }

        function renderSelectedDayNutrition() {
            const currentDay = weekDaysData[selectedNutriDayIdx];
            const dateDisp = document.getElementById("nutriSelectedDateDisplay");
            if (dateDisp) {
                dateDisp.innerText = currentDay.fullDate + " (" + currentDay.dayName + ")";
            }

            const dayMeals = weeklyNutrition[currentDay.fullDate] || [];
            let cal = 0, pro = 0, carb = 0, fat = 0;
            const list = document.getElementById("mealsList");
            if (!list) return;
            list.innerHTML = "";

            if (dayMeals.length === 0) {
                list.innerHTML = `
                    <div class="empty-day-box">
                        <div class="empty-day-icon">🍽️</div>
                        <div class="empty-day-title">Öğün Girilmedi</div>
                        <div class="empty-day-desc">${currentDay.fullDate} tarihi için henüz yemek kaydedilmedi.</div>
                    </div>
                `;
            } else {
                dayMeals.forEach(meal => {
                    cal += parseFloat(meal.calories || 0);
                    pro += parseFloat(meal.protein || 0);
                    carb += parseFloat(meal.carbs || 0);
                    fat += parseFloat(meal.fat || 0);

                    const subHtml = meal.items_summary ? `<div class="meal-items-subtext">🍽️ ${meal.items_summary}</div>` : '';

                    list.innerHTML += `
                        <div class="log-item" style="flex-direction:column; align-items:flex-start; gap:6px;">
                            <div style="display:flex; justify-content:space-between; width:100%; align-items:center;">
                                <div>
                                    <span class="ex-title">${meal.food_name}</span>
                                    <div style="font-size:0.75rem; color:#9ca3af; margin-top:2px;">
                                        <span class="macro-c-cal">${Math.round(meal.calories)} kcal</span> | 
                                        <span class="macro-c-pro">P: ${Math.round(meal.protein)}g</span> | 
                                        <span class="macro-c-carb">K: ${Math.round(meal.carbs)}g</span> | 
                                        <span class="macro-c-fat">Y: ${Math.round(meal.fat)}g</span>
                                    </div>
                                </div>
                                <button onclick="deleteMeal(${meal.id})">Sil</button>
                            </div>
                            ${subHtml}
                        </div>
                    `;
                });
            }

            document.getElementById("totCalories").innerText = Math.round(cal) + " kcal";
            document.getElementById("totProtein").innerText = Math.round(pro) + "g";
            document.getElementById("totCarbs").innerText = Math.round(carb) + "g";
            document.getElementById("totFat").innerText = Math.round(fat) + "g";
        }

        function deleteMeal(id) {
            const currentDay = weekDaysData[selectedNutriDayIdx];
            if (!weeklyNutrition[currentDay.fullDate]) return;
            weeklyNutrition[currentDay.fullDate] = weeklyNutrition[currentDay.fullDate].filter(m => m.id !== id);
            saveUserWeeklyNutrition(currentUser.username, weeklyNutrition);
            renderSelectedDayNutrition();
        }

        function clearSelectedDayMeals() {
            const currentDay = weekDaysData[selectedNutriDayIdx];
            if (!confirm(currentDay.fullDate + " tarihindeki tüm öğünleri sıfırlamak istiyor musun?")) return;
            weeklyNutrition[currentDay.fullDate] = [];
            saveUserWeeklyNutrition(currentUser.username, weeklyNutrition);
            renderSelectedDayNutrition();
        }

        let conversationHistory = [];
        let selectedBase64Image = null;
        let nutriSelectedImage = null;

        function handleImageSelect(event, type) {
            const file = event.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = function(e) {
                if (type === 'coach') {
                    selectedBase64Image = e.target.result;
                    document.getElementById("imagePreview").src = selectedBase64Image;
                    document.getElementById("previewBox").style.display = "flex";
                } else {
                    nutriSelectedImage = e.target.result;
                    document.getElementById("nutriImagePreview").src = nutriSelectedImage;
                    document.getElementById("nutriPreviewBox").style.display = "flex";
                }
            };
            reader.readAsDataURL(file);
        }

        function clearImage() {
            selectedBase64Image = null;
            document.getElementById("imageInput").value = "";
            document.getElementById("previewBox").style.display = "none";
        }

        function clearNutriImage() {
            nutriSelectedImage = null;
            document.getElementById("nutriImageInput").value = "";
            document.getElementById("nutriPreviewBox").style.display = "none";
        }

        async function sendMessage() {
            const input = document.getElementById("userInput");
            const btn = document.getElementById("sendBtn");
            const chatBox = document.getElementById("chatBox");
            const text = input.value.trim();
            
            if (!text && !selectedBase64Image) return;

            let userHtml = "";
            if (selectedBase64Image) userHtml += `<img src="${selectedBase64Image}" class="preview-img" />`;
            userHtml += `<span>${text || "Fotoğraf analizi"}</span>`;

            chatBox.innerHTML += `<div class="msg user">${userHtml}</div>`;
            const currentImg = selectedBase64Image;
            const currentText = text || "Bu fotoğrafı analiz et.";

            input.value = "";
            clearImage();
            btn.disabled = true;
            chatBox.scrollTop = chatBox.scrollHeight;

            const loadingId = "load-" + Date.now();
            chatBox.innerHTML += `<div class="msg coach" id="${loadingId}"><i>Analiz ediliyor...</i></div>`;
            chatBox.scrollTop = chatBox.scrollHeight;

            const lastSets = weeklyLogs.slice(-6).map(s => `${s.exercise} (${s.set_num}.Set, ${s.date}): ${s.weight}kg x ${s.reps}`).join(", ");

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        user_message: currentText,
                        workout_summary: lastSets,
                        image_base64: currentImg,
                        history: conversationHistory
                    })
                });
                const data = await response.json();
                let replyFormatted = (data.coach_reply || "").replace(/\\n/g, "<br>").replace(/\\*\\*(.*?)\\*\\*/g, "<b>$1</b>");
                document.getElementById(loadingId).innerHTML = replyFormatted || "Yanıt alındı.";

                conversationHistory.push({ role: "user", content: currentText });
                conversationHistory.push({ role: "assistant", content: data.coach_reply });
                if (conversationHistory.length > 8) conversationHistory = conversationHistory.slice(-8);
            } catch (err) {
                document.getElementById(loadingId).innerText = "Hata oluştu kral.";
            } finally {
                btn.disabled = false;
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        }

        async function sendNutriMessage() {
            const input = document.getElementById("nutriUserInput");
            const btn = document.getElementById("nutriSendBtn");
            const chatBox = document.getElementById("nutriChatBox");
            const text = input.value.trim();

            if (!text && !nutriSelectedImage) return;

            const targetDateStr = weekDaysData[selectedNutriDayIdx].fullDate;

            let userHtml = "";
            if (nutriSelectedImage) userHtml += `<img src="${nutriSelectedImage}" class="preview-img" />`;
            userHtml += `<span>${text || "Yemek analizi"}</span>`;

            chatBox.innerHTML += `<div class="msg user">${userHtml}</div>`;
            const currentImg = nutriSelectedImage;
            const currentText = text || "Bu yemeğin makrolarını hesapla ve ekle.";

            input.value = "";
            clearNutriImage();
            btn.disabled = true;
            chatBox.scrollTop = chatBox.scrollHeight;

            const loadingId = "load-nutri-" + Date.now();
            chatBox.innerHTML += `<div class="msg coach" id="${loadingId}"><i>Makrolar hesaplanıyor...</i></div>`;
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const response = await fetch("/nutrition-chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        user_message: currentText,
                        image_base64: currentImg
                    })
                });
                const data = await response.json();

                let replyFormatted = (data.coach_reply || "").replace(/\\n/g, "<br>").replace(/\\*\\*(.*?)\\*\\*/g, "<b>$1</b>");
                document.getElementById(loadingId).innerHTML = replyFormatted || "Yanıt alındı.";

                if (data.detected_meal && Number(data.detected_meal.calories) > 0) {
                    const newMeal = {
                        id: Date.now() + Math.floor(Math.random() * 1000),
                        food_name: data.detected_meal.food_name || "Öğün",
                        items_summary: data.detected_meal.items_summary || currentText,
                        calories: Math.round(Number(data.detected_meal.calories) || 0),
                        protein: Math.round(Number(data.detected_meal.protein) || 0),
                        carbs: Math.round(Number(data.detected_meal.carbs) || 0),
                        fat: Math.round(Number(data.detected_meal.fat) || 0)
                    };

                    if (!weeklyNutrition[targetDateStr]) {
                        weeklyNutrition[targetDateStr] = [];
                    }
                    weeklyNutrition[targetDateStr] = [...weeklyNutrition[targetDateStr], newMeal];
                    saveUserWeeklyNutrition(currentUser.username, weeklyNutrition);
                    renderSelectedDayNutrition();
                }
            } catch (err) {
                document.getElementById(loadingId).innerText = "Hata oluştu kral, tekrar dener misin?";
            } finally {
                btn.disabled = false;
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        }

        function handleKey(e, type) {
            if (e.key === "Enter") {
                if (type === 'coach') sendMessage();
                else sendNutriMessage();
            }
        }

        checkAuth();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTML_INTERFACE

@app.post("/chat")
def coach_dialogue(data: ChatInput):
    if not client:
        return {"user_message": data.user_message, "coach_reply": "Sunucuda GROQ_API_KEY bulunamadı."}

    user_context = f"Kullanıcının Bu Haftaki Son Setleri: {data.workout_summary}" if data.workout_summary else "Bu hafta henüz set girilmedi."

    system_prompt = f"""
Sen elit seviyede bir 'Looksmaxxing, Hipertrofi & Fizik Koçu'sun.
KULLANICI HAFTALIK ANTRENMAN GEÇMİŞİ:
{user_context}

1. SET / PROGRESSIVE OVERLOAD DEĞERLENDİRMESİ:
- Kullanıcının bu haftaki ağırlık ve setlerine bakarak bir sonraki idmanda hedeflemesi gereken net kg ve tekrarı söyle.
"""
    messages = [{"role": "system", "content": system_prompt}]
    for msg in data.history:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    if data.image_base64:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": data.user_message},
            {"type": "image_url", "image_url": {"url": data.image_base64}}
        ]})
    else:
        messages.append({"role": "user", "content": data.user_message})

    reply_text = None
    for m in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
        try:
            chat_completion = client.chat.completions.create(
                messages=messages, model=m, temperature=0.4, max_tokens=600,
            )
            reply_text = chat_completion.choices[0].message.content
            if reply_text:
                break
        except Exception as e:
            logger.warning(f"[/chat] model {m} failed: {e}")
            continue

    if not reply_text:
        reply_text = "Hedeflerine odaklan kral! Antrenmanda her zaman bir önceki haftadan 1 tekrar veya 1-2.5 kg fazla zorlamaya devam et."

    reply_text = re.sub(r'<think>.*?</think>', '', reply_text, flags=re.DOTALL).strip()
    return {"user_message": data.user_message, "coach_reply": reply_text}

@app.post("/nutrition-chat")
def nutrition_dialogue(data: NutritionChatInput):
    detected_meal = parse_and_calculate_meal(data.user_message)
    
    if detected_meal:
        reply_text = (
            f"Afiyet olsun kral! Girdiğin **{detected_meal['items_summary']}** başarıyla listene eklendi: "
            f"**{detected_meal['calories']} kcal | {detected_meal['protein']}g Protein | "
            f"{detected_meal['carbs']}g Karb | {detected_meal['fat']}g Yağ**"
        )
    else:
        reply_text = "Bu yemeği tanıyamadım kral. Lütfen miktar belirterek yaz (Örn: '4 dilim tam buğday ekmeği', '1 ölçek protein tozu')."
        detected_meal = None

    return {
        "user_message": data.user_message,
        "coach_reply": reply_text,
        "detected_meal": detected_meal
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
