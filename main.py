import os
import json
import re
import urllib.request
import urllib.parse
import logging
import traceback
from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from groq import Groq

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("looksmax-hub")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = None

if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logger.error(f"Groq istemcisi baslatilamadi: {e}")
        traceback.print_exc()
else:
    logger.warning("GROQ_API_KEY ortam degiskeni bulunamadi!")

app = FastAPI(title="Looksmax Hub - Elite Performance & Coaching Engine")

# ================= 0. DINAMIK MODEL SECICI & CIKTI TEMIZLEYICI =================
def get_best_available_model() -> str:
    preferences = [
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "gemma2-9b-it",
        "mixtral-8x7b-32768",
        "openai/gpt-oss-20b"
    ]
    if not client:
        return "llama-3.1-8b-instant"
    try:
        models_response = client.models.list()
        active_models = [m.id for m in models_response.data]
        logger.info(f"Aktif Groq Modelleri: {active_models}")
        
        for pref in preferences:
            if pref in active_models:
                return pref
                
        chat_models = [m for m in active_models if not any(x in m for x in ["whisper", "tts", "guard", "embed"])]
        if chat_models:
            return chat_models[0]
    except Exception as e:
        logger.warning(f"Dinamik model secilemedi, varsayilana donuluyor: {e}")
    
    return "llama-3.1-8b-instant"

def strip_thinking_and_tables(text: str) -> str:
    if not text:
        return ""
    # 1. <think> etiketlerini ve iceriklerini temizle
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # 2. Eger yanitta '### Özet', '## Özet' veya 'Özet:' varsa, oncesindeki analiz tablolarini atip direkt ozeti al
    summary_match = re.search(r'(?:###?\s*Özet|Özet:?)(.*)', clean, flags=re.DOTALL | re.IGNORECASE)
    if summary_match and len(summary_match.group(1).strip()) > 20:
        return summary_match.group(1).strip()
    
    # 3. Model ozet basligi atmadan sadece tablo ve altina aciklama koyduysa, Markdown tablolarini temizle
    lines = clean.split('\n')
    filtered_lines = [l for l in lines if not l.strip().startswith('|') and not l.strip().startswith('+-')]
    result = '\n'.join(filtered_lines).strip()
    
    return result if result else clean

# ================= 1. NUTRITION ENGINE (LLM + DETERMINISTIK) =================
CANONICAL_FOODS = {
    "yumurta": {"unit_type": "piece", "base_cal": 72.0, "base_pro": 6.3, "base_carb": 0.4, "base_fat": 4.8},
    "haslanmis yumurta": {"unit_type": "piece", "base_cal": 72.0, "base_pro": 6.3, "base_carb": 0.4, "base_fat": 4.8},
    "sahanda yumurta": {"unit_type": "piece", "base_cal": 92.0, "base_pro": 6.3, "base_carb": 0.4, "base_fat": 7.2},
    "omlet": {"unit_type": "piece", "base_cal": 95.0, "base_pro": 6.5, "base_carb": 0.6, "base_fat": 7.5},
    "yumurta beyazi": {"unit_type": "piece", "base_cal": 17.0, "base_pro": 3.6, "base_carb": 0.2, "base_fat": 0.1},
    "tam bugday ekmegi": {"unit_type": "piece", "base_cal": 68.0, "base_pro": 3.5, "base_carb": 12.0, "base_fat": 0.9},
    "ekmek": {"unit_type": "piece", "base_cal": 75.0, "base_pro": 2.4, "base_carb": 15.0, "base_fat": 0.8},
    "beyaz ekmek": {"unit_type": "piece", "base_cal": 75.0, "base_pro": 2.4, "base_carb": 15.0, "base_fat": 0.8},
    "lavas": {"unit_type": "piece", "base_cal": 160.0, "base_pro": 5.0, "base_carb": 30.0, "base_fat": 2.5},
    "tavuk gogsu": {"unit_type": "gram_100", "base_cal": 120.0, "base_pro": 22.5, "base_carb": 0.0, "base_fat": 2.5},
    "tavuk": {"unit_type": "gram_100", "base_cal": 135.0, "base_pro": 21.0, "base_carb": 0.0, "base_fat": 5.0},
    "pirinc": {"unit_type": "gram_100", "base_cal": 350.0, "base_pro": 7.0, "base_carb": 78.0, "base_fat": 0.6},
    "yulaf": {"unit_type": "gram_100", "base_cal": 370.0, "base_pro": 12.5, "base_carb": 60.0, "base_fat": 7.0},
    "protein tozu": {"unit_type": "piece", "base_cal": 120.0, "base_pro": 24.0, "base_carb": 2.0, "base_fat": 1.5},
    "whey": {"unit_type": "piece", "base_cal": 120.0, "base_pro": 24.0, "base_carb": 2.0, "base_fat": 1.5},
    "muz": {"unit_type": "piece", "base_cal": 105.0, "base_pro": 1.3, "base_carb": 27.0, "base_fat": 0.3},
    "elma": {"unit_type": "piece", "base_cal": 80.0, "base_pro": 0.4, "base_carb": 21.0, "base_fat": 0.3},
    "fistik ezmesi": {"unit_type": "gram_100", "base_cal": 590.0, "base_pro": 25.0, "base_carb": 20.0, "base_fat": 50.0},
    "zeytinyagi": {"unit_type": "piece", "base_cal": 120.0, "base_pro": 0.0, "base_carb": 0.0, "base_fat": 14.0},
    "kiyma": {"unit_type": "gram_100", "base_cal": 220.0, "base_pro": 20.0, "base_carb": 0.0, "base_fat": 15.0},
    "kofte": {"unit_type": "piece", "base_cal": 85.0, "base_pro": 7.0, "base_carb": 2.0, "base_fat": 5.5},
    "patates": {"unit_type": "gram_100", "base_cal": 77.0, "base_pro": 2.0, "base_carb": 17.5, "base_fat": 0.1},
    "sut": {"unit_type": "gram_100", "base_cal": 60.0, "base_pro": 3.2, "base_carb": 4.8, "base_fat": 3.1},
    "yogurt": {"unit_type": "gram_100", "base_cal": 65.0, "base_pro": 3.5, "base_carb": 4.7, "base_fat": 3.3},
    "lor peyniri": {"unit_type": "gram_100", "base_cal": 90.0, "base_pro": 17.0, "base_carb": 3.0, "base_fat": 1.0}
}

class ParsedFoodItem(BaseModel):
    name: str = Field(description="Besinin turkce yalin adi (orn: sahanda yumurta, pirinc, tam bugday ekmegi, tavuk gogsu)")
    amount: float = Field(description="Miktar sayisal degeri (orn: 1, 2, 150, 0.5)")
    unit: str = Field(description="Birim turu: 'adet', 'gram', 'dilim', 'scoop', 'kasik', 'kase', 'porsiyon'")
    estimated_grams: Optional[float] = Field(default=None, description="Adet veya porsiyonsa toplam tahmini gramaji")

class ParsedMealResponse(BaseModel):
    items: List[ParsedFoodItem]

def normalize_turkish(text: str) -> str:
    t = text.lower()
    t = t.replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return " ".join(t.split()).strip()

def parse_meal_with_llm(user_text: str) -> Optional[Dict[str, Any]]:
    if not client:
        return None

    system_prompt = """
Sen profesyonel bir besin ve diyet parser'isin.
Gorevin: Kullanicinin girdigi serbest metindeki ogunleri tespit edip YALNIZCA JSON formatinda ParsedMealResponse semasina uygun cikti vermek.

KURALLAR:
1. '1 adet sahanda yumurta' dendiginde: name='sahanda yumurta', amount=1, unit='adet', estimated_grams=60 olmalidir. ASLA 100g olarak algilama!
2. '2 yumurta' dendiginde unit='adet' olmalidir.
3. '4 dilim ekmek' dendiginde amount=4, unit='dilim' olmalidir.
4. '1 olcek protein tozu' dendiginde amount=1, unit='scoop' olmalidir.
5. Cikti formati kesinlikle su JSON seklinde olmalidir:
{
  "items": [
    {"name": "sahanda yumurta", "amount": 1, "unit": "adet", "estimated_grams": 60}
  ]
}
"""
    try:
        active_model = get_best_available_model()
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            model=active_model,
            response_format={"type": "json_object"},
            temperature=0.1
        )
        parsed_json = json.loads(completion.choices[0].message.content)
        data = ParsedMealResponse(**parsed_json)
        
        total_cal = 0.0
        total_pro = 0.0
        total_carb = 0.0
        total_fat = 0.0
        summary_items = []

        for item in data.items:
            norm_name = normalize_turkish(item.name)
            matched = None
            
            for key, val in CANONICAL_FOODS.items():
                if key == norm_name or key in norm_name or norm_name in key:
                    matched = val
                    break
            
            if matched:
                multiplier = 1.0
                if matched["unit_type"] == "piece":
                    multiplier = item.amount
                elif matched["unit_type"] == "gram_100":
                    grams = item.amount if item.unit in ["gram", "g", "gr"] else (item.estimated_grams or 100.0)
                    multiplier = grams / 100.0

                total_cal += matched["base_cal"] * multiplier
                total_pro += matched["base_pro"] * multiplier
                total_carb += matched["base_carb"] * multiplier
                total_fat += matched["base_fat"] * multiplier
                summary_items.append(f"{item.amount:g} {item.unit} {item.name.title()}")
            else:
                grams = item.amount if item.unit in ["gram", "g", "gr"] else (item.estimated_grams or 100.0)
                ratio = grams / 100.0
                total_cal += 150.0 * ratio
                total_pro += 10.0 * ratio
                total_carb += 15.0 * ratio
                total_fat += 5.0 * ratio
                summary_items.append(f"{item.amount:g} {item.unit} {item.name.title()}")

        if total_cal > 0:
            return {
                "food_name": " + ".join(summary_items),
                "items_summary": ", ".join(summary_items),
                "calories": round(total_cal),
                "protein": round(total_pro, 1),
                "carbs": round(total_carb, 1),
                "fat": round(total_fat, 1)
            }
    except Exception as e:
        logger.error(f"LLM Nutrition Parse Hatasi: {e}")
        traceback.print_exc()

    return None

# ================= 2. RECOVERY ENGINE =================
def compute_recovery_score(sleep_hours: float, hrv: float, resting_hr: float) -> Dict[str, Any]:
    sleep_score = min(40.0, (sleep_hours / 8.0) * 40.0)
    hrv_score = min(35.0, (hrv / 75.0) * 35.0)

    rhr_score = 25.0
    if resting_hr > 60:
        rhr_score = max(5.0, 25.0 - (resting_hr - 60) * 0.8)

    total_score = round(min(100.0, max(10.0, sleep_score + hrv_score + rhr_score)))

    if total_score >= 80:
        status = "Optimal Toparlanma 🔥"
        cns_advice = "Merkezi sinir sistemin zirvede. Bahanen sıfır; bugün ağırlıkları artırıp tükenişe gitmelisin."
        badge_color = "#10b981"
    elif total_score >= 60:
        status = "Orta / Yeterli Toparlanma ⚡"
        cns_advice = "Vücut antrenmana hazır. Formu bozmadan setlerde 1 tekrar cepte bırak (RIR 1)."
        badge_color = "#00f2fe"
    else:
        status = "Yetersiz Toparlanma / Yüksek Stres ⚠️"
        cns_advice = "Otonom sinir sistemin yorgun. Sakatlanmamak için PR zorlama, form odaklı kal."
        badge_color = "#ef4444"

    return {
        "recovery_score": total_score,
        "status": status,
        "cns_advice": cns_advice,
        "badge_color": badge_color
    }

# ================= 3. SCHEMAS & API ENDPOINTS =================
class ChatInput(BaseModel):
    user_message: str
    image_base64: Optional[str] = None
    workout_summary: Optional[str] = ""
    user_profile_summary: Optional[str] = ""
    health_summary: Optional[str] = ""
    injuries_summary: Optional[str] = ""
    history: List[dict] = []

class NutritionChatInput(BaseModel):
    user_message: str
    image_base64: Optional[str] = None
    daily_summary: Optional[str] = ""
    history: List[dict] = []

class HealthSyncInput(BaseModel):
    username: str
    date: str
    sleep_hours: float
    deep_sleep_hours: Optional[float] = 0.0
    hrv_ms: float
    resting_hr: float
    avg_workout_hr: Optional[float] = None
    max_workout_hr: Optional[float] = None
    steps: Optional[int] = 0

class CoachAuditInput(BaseModel):
    profile_data: Optional[dict] = {}
    recent_workouts: Optional[list] = []
    recent_nutrition: Optional[dict] = {}
    recent_health: Optional[dict] = {}
    active_injuries: Optional[list] = []

HTML_INTERFACE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Looksmax HUB - Elite Performance & Coaching</title>
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
        
        .audit-trigger-btn { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: #000; border: none; padding: 6px 14px; border-radius: 8px; font-size: 0.8rem; font-weight: 800; cursor: pointer; display: flex; align-items: center; gap: 6px; box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3); transition: 0.2s; }
        .audit-trigger-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(245, 158, 11, 0.5); }

        .user-section { display: flex; align-items: center; gap: 10px; }
        .user-tag { font-size: 0.8rem; background: #161c26; padding: 6px 12px; border-radius: 8px; color: #10b981; border: 1px solid #263245; }
        .logout-btn { background: none; border: none; color: #ef4444; cursor: pointer; font-size: 0.8rem; font-weight: 600; }

        .content-container { flex: 1; display: flex; justify-content: center; align-items: center; overflow: hidden; position: relative; }
        .view-panel { display: none; width: 100%; height: 100%; padding: 20px; }
        .view-panel.active { display: flex; }

        #hubView { justify-content: center; align-items: center; flex-direction: column; gap: 24px; }
        .hub-title { text-align: center; }
        .hub-title h1 { font-size: 2.2rem; font-weight: 800; color: #fff; margin-bottom: 4px; letter-spacing: 0.5px; }
        .hub-title p { font-size: 0.95rem; color: #9ca3af; }
        
        .hub-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; max-width: 1350px; width: 100%; justify-content: center; }
        .hub-card { background: #131722; border: 1px solid #222c3f; border-radius: 18px; padding: 22px 18px; display: flex; flex-direction: column; justify-content: space-between; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 8px 24px rgba(0,0,0,0.4); text-align: left; }
        .hub-card:hover { transform: translateY(-6px); border-color: #00f2fe; box-shadow: 0 12px 35px rgba(0,242,254,0.22); }
        .card-icon { font-size: 2rem; margin-bottom: 10px; }
        .card-heading { font-size: 1.1rem; font-weight: 800; color: #fff; margin-bottom: 6px; }
        .card-desc { font-size: 0.75rem; color: #9ca3af; line-height: 1.4; margin-bottom: 14px; }
        .card-action { align-self: flex-start; background: #1a2232; color: #00f2fe; border: 1px solid #2d3b54; padding: 7px 12px; border-radius: 8px; font-weight: 700; font-size: 0.75rem; transition: 0.2s; }
        .hub-card:hover .card-action { background: #00f2fe; color: #000; }

        .panel-card { background: #131722; border: 1px solid #1f2738; border-radius: 16px; padding: 18px; display: flex; flex-direction: column; gap: 12px; }
        .panel-header { font-size: 0.95rem; font-weight: 800; color: #00f2fe; display: flex; justify-content: space-between; align-items: center; }
        .badge-cyan { font-size: 0.75rem; background: rgba(0, 242, 254, 0.1); color: #00f2fe; border: 1px solid rgba(0, 242, 254, 0.3); padding: 4px 8px; border-radius: 6px; font-weight: 600; }
        
        .overload-col-left { width: 45%; display: flex; flex-direction: column; gap: 16px; height: 100%; overflow-y: auto; padding-right: 4px; }
        .overload-col-right { width: 55%; display: flex; flex-direction: column; gap: 16px; height: 100%; }

        .input-form { display: flex; flex-direction: column; gap: 10px; }
        .input-form input, .input-form select { background: #0a0c10; border: 1px solid #2b354d; color: #fff; padding: 11px 12px; border-radius: 8px; font-size: 0.85rem; outline: none; width: 100%; }
        .input-form input:focus, .input-form select:focus { border-color: #00f2fe; }
        .form-grid-2x2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; width: 100%; }
        .form-grid-3x1 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; width: 100%; }
        .btn-log { background: #00f2fe; color: #000; border: none; font-weight: 800; padding: 12px; border-radius: 8px; cursor: pointer; margin-top: 4px; }

        .days-tab-bar { display: flex; gap: 6px; overflow-x: auto; padding-bottom: 4px; }
        .day-tab-btn { flex: 1; min-width: 48px; background: #0a0c10; border: 1px solid #1f2738; border-radius: 10px; padding: 8px 4px; color: #9ca3af; font-size: 0.75rem; font-weight: 700; cursor: pointer; text-align: center; transition: 0.2s; }
        .day-tab-btn .tab-sub { font-size: 0.65rem; color: #6b7280; display: block; margin-top: 2px; }
        .day-tab-btn:hover { border-color: #2b3a52; color: #fff; }
        .day-tab-btn.active { background: #172133; border-color: #00f2fe; color: #00f2fe; }
        .day-tab-btn.active .tab-sub { color: #38bdf8; }

        .history-list { flex: 1; overflow-y: auto; max-height: 380px; display: flex; flex-direction: column; gap: 8px; padding-right: 4px; }
        .log-item { display: flex; justify-content: space-between; align-items: center; background: #0a0c10; padding: 10px 14px; border-radius: 9px; font-size: 0.85rem; border: 1px solid #1c2230; }
        .log-item .set-badge { background: #1e293b; color: #00f2fe; padding: 2px 7px; border-radius: 5px; font-weight: 700; font-size: 0.75rem; margin-right: 6px; }
        .log-item .ex-title { font-weight: 700; color: #fff; }
        .log-item .ex-val { color: #38bdf8; font-weight: 700; }
        .log-item button { background: none; border: none; color: #ef4444; cursor: pointer; font-size: 0.85rem; padding: 2px 4px; }

        #coachView { flex-direction: column; max-width: 950px; }
        .chat-container { flex: 1; display: flex; flex-direction: column; background: #131722; border-radius: 16px; border: 1px solid #1f2738; overflow: hidden; }
        .messages { flex: 1; overflow-y: auto; padding: 22px; display: flex; flex-direction: column; gap: 14px; }
        .msg { max-width: 82%; padding: 13px 17px; border-radius: 14px; font-size: 0.92rem; line-height: 1.5; word-wrap: break-word; }
        .msg.user { align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 3px; }
        .msg.coach { align-self: flex-start; background: #1a2130; border: 1px solid #283449; border-bottom-left-radius: 3px; }
        .msg.error { background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #fca5a5; font-family: monospace; font-size: 0.8rem; }
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

        #profileView { gap: 20px; max-width: 1400px; }
        .phase-header-bar { display: flex; justify-content: space-between; align-items: center; background: #0a0c10; padding: 10px 14px; border-radius: 10px; border: 1px solid #1c2230; margin-bottom: 10px; }
        .phase-selector { background: #141923; border: 1px solid #2b354d; color: #00f2fe; padding: 6px 12px; border-radius: 8px; font-weight: 700; font-size: 0.85rem; outline: none; }
        .photo-matrix-4x { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 12px; height: 500px; }
        .photo-card-slot { background: #0a0c10; border: 1px dashed #28354b; border-radius: 12px; position: relative; display: flex; flex-direction: column; align-items: center; justify-content: center; overflow: hidden; height: 100%; cursor: pointer; transition: 0.2s; }
        .photo-card-slot:hover { border-color: #00f2fe; }
        .photo-card-slot img { width: 100%; height: 100%; object-fit: cover; position: absolute; top: 0; left: 0; }
        .slot-badge { position: absolute; top: 8px; left: 8px; background: rgba(0,0,0,0.8); border: 1px solid #00f2fe; color: #00f2fe; font-size: 0.7rem; font-weight: 800; padding: 3px 8px; border-radius: 5px; z-index: 2; }
        .btn-remove-photo { position: absolute; top: 8px; right: 8px; background: rgba(239, 68, 68, 0.9); border: none; color: white; width: 22px; height: 22px; border-radius: 50%; font-size: 0.75rem; font-weight: 800; cursor: pointer; z-index: 3; display: none; }
        .slot-placeholder { z-index: 1; text-align: center; color: #6b7280; font-size: 0.75rem; }

        #healthView { gap: 20px; max-width: 1400px; }
        .recovery-banner { background: #0a0c10; border: 1px solid #1c2230; border-radius: 14px; padding: 18px; display: flex; align-items: center; gap: 20px; }
        .recovery-circle { width: 84px; height: 84px; border-radius: 50%; border: 4px solid #00f2fe; display: flex; flex-direction: column; align-items: center; justify-content: center; font-size: 1.45rem; font-weight: 800; color: #fff; flex-shrink: 0; }
        .recovery-circle span { font-size: 0.65rem; color: #9ca3af; font-weight: 600; }
        .recovery-info h3 { font-size: 1.15rem; font-weight: 800; color: #00f2fe; margin-bottom: 4px; }
        .recovery-info p { font-size: 0.8rem; color: #9ca3af; line-height: 1.45; }

        .guide-btn-card { background: linear-gradient(135deg, #131b2a 0%, #0d121c 100%); border: 1px solid #1f2e47; padding: 16px; border-radius: 12px; display: flex; justify-content: space-between; align-items: center; cursor: pointer; transition: 0.2s; }
        .guide-btn-card:hover { border-color: #00f2fe; transform: translateY(-2px); }

        .modal-overlay { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.85); display: none; justify-content: center; align-items: center; z-index: 10000; backdrop-filter: blur(6px); }
        .modal-box { background: #131722; border: 1px solid #222d42; border-radius: 18px; width: 90%; max-width: 620px; padding: 26px; display: flex; flex-direction: column; gap: 16px; box-shadow: 0 16px 50px rgba(0,242,254,0.18); position: relative; max-height: 90vh; overflow-y: auto; }
        .modal-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e2638; padding-bottom: 12px; }
        .modal-header h3 { font-size: 1.15rem; font-weight: 800; color: #00f2fe; }
        .modal-close-btn { background: none; border: none; color: #9ca3af; font-size: 1.2rem; cursor: pointer; }
        .modal-step { background: #0a0c10; border: 1px solid #1c2230; padding: 12px 14px; border-radius: 10px; font-size: 0.82rem; line-height: 1.5; color: #e5e7eb; }
        .modal-step b { color: #00f2fe; }
        .url-box { background: #171f2e; border: 1px solid #2a374f; padding: 8px 12px; border-radius: 6px; font-family: monospace; color: #10b981; font-size: 0.8rem; display: flex; justify-content: space-between; align-items: center; margin-top: 6px; word-break: break-all; }

        .audit-content-area { font-size: 0.9rem; line-height: 1.6; color: #d1d5db; white-space: pre-line; background: #0a0c10; border: 1px solid #1f2738; padding: 18px; border-radius: 12px; max-height: 480px; overflow-y: auto; }

        @media (max-width: 1100px) {
            body { overflow: auto; height: auto; }
            .hub-grid { grid-template-columns: repeat(2, 1fr); }
            .content-container { height: auto; overflow: visible; }
            .view-panel { height: auto; flex-direction: column !important; }
            .overload-col-left, .overload-col-right { width: 100%; }
            #coachView { height: 80vh; }
            .macro-stat-grid { grid-template-columns: repeat(2, 1fr); }
            .photo-matrix-4x { height: auto; grid-template-columns: 1fr; grid-template-rows: repeat(4, 220px); }
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

    <div class="modal-overlay" id="appleWatchModal" onclick="closeGuideModal(event)">
        <div class="modal-box" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h3>📲 Apple Watch & iPhone Otomatik Senkronizasyon</h3>
                <button class="modal-close-btn" onclick="toggleGuideModal(false)">✕</button>
            </div>
            
            <div class="modal-step">
                <b>Adım 1:</b> iPhone'unuzda <b>Kestirmeler (Shortcuts)</b> uygulamasını açıp <b>+</b> ile yeni bir kestirme oluşturun.
            </div>

            <div class="modal-step">
                <b>Adım 2:</b> Sırasıyla şu 3 sağlık verisini ekleyin:
                <ul style="margin-left: 18px; margin-top: 4px; color:#9ca3af;">
                    <li><i>Sağlık Örneklerini Bul</i> $\rightarrow$ <b>Uyku Analizi</b> (Süre/Saat)</li>
                    <li><i>Sağlık Örneklerini Bul</i> $\rightarrow$ <b>Kalp Atış Hızı Değişkenliği (HRV)</b></li>
                    <li><i>Sağlık Örneklerini Bul</i> $\rightarrow$ <b>Dinlenme Sırasındaki Kalp Atış Hızı</b></li>
                </ul>
            </div>

            <div class="modal-step">
                <b>Adım 3:</b> <b>URL İçeriğini Al</b> eylemini ekleyip <b>POST</b> yöntemiyle şu adresi girin:
                <div class="url-box" id="webhookUrlBox">
                    <span id="webhookUrlText">https://.../api/health-sync</span>
                    <button onclick="copyWebhookUrl()" style="background:#00f2fe; color:#000; border:none; padding:4px 8px; border-radius:4px; font-weight:800; font-size:0.7rem; cursor:pointer;">Kopyala</button>
                </div>
            </div>

            <div class="modal-step">
                <b>💡 Otomatikleştirme:</b> Kestirmeler $\rightarrow$ Otomasyon sekmesinden <i>"Sabah Alarmı Durdurulduğunda"</i> bu kestirmeyi seçerseniz verileriniz her sabah otomatik panele düşer.
            </div>

            <button class="btn-log" onclick="toggleGuideModal(false)" style="margin-top:0;">Anladım Kral 🦍</button>
        </div>
    </div>

    <div class="modal-overlay" id="coachAuditModal" onclick="closeAuditModal(event)">
        <div class="modal-box" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h3 style="color:#f59e0b;">🧠 Koçun Raporu & Haftalık Analiz</h3>
                <button class="modal-close-btn" onclick="toggleAuditModal(false)">✕</button>
            </div>
            
            <div id="auditLoadingState" style="text-align:center; padding:30px; display:none;">
                <div style="font-size:2.2rem; margin-bottom:8px;">🦍</div>
                <div style="font-weight:700; color:#f59e0b;">Koç tüm antrenman, sakatlık, makro ve uyku verilerini denetliyor...</div>
            </div>

            <div class="audit-content-area" id="auditContentText">
                Rapor yükleniyor...
            </div>

            <button class="btn-log" onclick="toggleAuditModal(false)" style="background:#f59e0b; color:#000; margin-top:0;">Anlaşıldı</button>
        </div>
    </div>

    <div class="header-bar">
        <div style="display:flex; align-items:center; gap:12px;">
            <div class="brand" onclick="openView('hub')">⚡ LOOKSMAX HUB</div>
            <button class="back-hub-btn" id="backHubBtn" onclick="openView('hub')">← Ana Menü</button>
        </div>
        
        <div class="user-section">
            <button class="audit-trigger-btn" onclick="triggerCoachAudit()">🧠 Koçun Raporu</button>
            <div class="user-tag" id="activeUserName">Giriş Yapılmadı</div>
            <button class="logout-btn" onclick="logout()">Çıkış</button>
        </div>
    </div>

    <div class="content-container">

        <div class="view-panel active" id="hubView">
            <div class="hub-title">
                <h1>Looksmax HUB</h1>
                <p>Hipertrofi, beslenme, biyometrik toparlanma, rehabilitasyon ve fizik takibi.</p>
            </div>
            <div class="hub-grid">
                <div class="hub-card" onclick="openView('coach')">
                    <div>
                        <div class="card-icon">🤖</div>
                        <div class="card-heading">AI Koç & Vision</div>
                        <div class="card-desc">Sakatlık duyarlı hipertrofi koçluğu, form kontrolü ve anlık taktikler.</div>
                    </div>
                    <div class="card-action">Koçla Konuş →</div>
                </div>

                <div class="hub-card" onclick="openView('overload')">
                    <div>
                        <div class="card-icon">📈</div>
                        <div class="card-heading">Progressive Overload</div>
                        <div class="card-desc">Set ve ağırlıklarını gün gün kaydet. Gelişim grafiklerini incele.</div>
                    </div>
                    <div class="card-action">Overload Takip →</div>
                </div>

                <div class="hub-card" onclick="openView('nutrition')">
                    <div>
                        <div class="card-icon">🥗</div>
                        <div class="card-heading">Beslenme & Makro</div>
                        <div class="card-desc">Deterministik LLM + Pydantic motoruyla yediklerini gramı gramına işle.</div>
                    </div>
                    <div class="card-action">Makro Takip →</div>
                </div>

                <div class="hub-card" onclick="openView('profile')">
                    <div>
                        <div class="card-icon">📸</div>
                        <div class="card-heading">Profil & Before/After</div>
                        <div class="card-desc">Ölçülerini kaydet, sınırsız dönem açıp Front/Back formlarını karşılaştır.</div>
                    </div>
                    <div class="card-action">Fizik Takip →</div>
                </div>

                <div class="hub-card" onclick="openView('health')">
                    <div>
                        <div class="card-icon">🫀</div>
                        <div class="card-heading">Recovery & Sakatlık</div>
                        <div class="card-desc">Apple Watch ile HRV, uyku, nabız ve Aktif Sakatlık / Rehabilitasyon takibi.</div>
                    </div>
                    <div class="card-action">Sağlık Takip →</div>
                </div>
            </div>
        </div>

        <div class="view-panel" id="coachView">
            <div class="chat-container">
                <div class="messages" id="chatBox">
                    <div class="msg coach">Selam kral! Ben senin Looksmax & Overload başantrenörünüm. Sakatlığın varsa doğrudan güvenli açı ve rehab protokolü veririm; toparlanman iyiyse hedefleri koyar geçerim. Sorunu sor veya sağ üstteki <b>🧠 Koçun Raporu</b> butonuna bas.</div>
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
                        <div style="display:flex; align-items:center; gap:8px;">
                            <span style="font-size:0.75rem; color:#9ca3af;">Hafta Seç:</span>
                            <select id="weekSelectorDropdown" onchange="changeActiveWeek(this.value)" style="background:#0a0c10; border:1px solid #2b354d; color:#00f2fe; padding:4px 8px; border-radius:6px; font-weight:700; font-size:0.75rem; outline:none;"></select>
                        </div>
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
                        <span>📊 Hareket Gelişim Grafiği (Tüm Haftalar)</span>
                        <select id="chartExerciseSelect" onchange="updateChart()" style="background:#0a0c10; border:1px solid #2b354d; color:#00f2fe; padding:6px 12px; border-radius:7px; font-weight:700; outline:none;"></select>
                    </div>
                    <div class="chart-box" style="flex:1; min-height:340px; position:relative;">
                        <canvas id="progressionChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <div class="view-panel" id="nutritionView">
            <div class="overload-col-left">
                <div class="chat-container">
                    <div class="messages" id="nutriChatBox">
                        <div class="msg coach">Afiyet olsun kral! Ne yediysen doğal dilde yaz (örn: <i>"1 adet sahanda yumurta"</i>, <i>"3 haşlanmış yumurta 2 dilim tam buğday"</i>); tam porsiyon üzerinden net hesaplarım.</div>
                    </div>

                    <div class="preview-box" id="nutriPreviewBox">
                        <img id="nutriImagePreview" src="" alt="Görsel" />
                        <button onclick="clearNutriImage()">✕</button>
                        <span style="font-size:0.75rem; color:#9ca3af;">Yemek görseli seçildi</span>
                    </div>

                    <div class="chat-input-area">
                        <label class="file-btn" for="nutriImageInput" title="Yemek Fotoğrafı">📷</label>
                        <input type="file" id="nutriImageInput" accept="image/*" onchange="handleImageSelect(event, 'nutri')" />
                        <input type="text" class="chat-input" id="nutriUserInput" placeholder="Yediklerini yaz..." onkeypress="handleKey(event, 'nutri')" />
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

        <div class="view-panel" id="profileView">
            <div class="overload-col-left">
                <div class="panel-card">
                    <div class="panel-header">
                        <span>👤 Sporcu Bilgileri & Hedef</span>
                        <span class="badge-cyan" id="bmrCalculatedBadge">BMR: - kcal</span>
                    </div>
                    <div class="input-form">
                        <div class="form-grid-2x2">
                            <input type="text" id="profFullName" placeholder="Ad Soyad" />
                            <input type="number" id="profAge" placeholder="Yaş" min="14" max="80" />
                        </div>
                        <div class="form-grid-3x1">
                            <input type="number" id="profHeight" placeholder="Boy (cm)" step="0.5" />
                            <input type="number" id="profWeight" placeholder="Kilo (kg)" step="0.1" />
                            <input type="number" id="profBodyfat" placeholder="Yağ Oranı (%)" step="0.5" />
                        </div>
                        <div class="form-grid-2x2">
                            <select id="profGoal">
                                <option value="Recomposition">Hedef: Recomposition (Clean)</option>
                                <option value="Lean Bulk">Hedef: Lean Bulk (Hacim)</option>
                                <option value="Aggressive Cut">Hedef: Cut (Yağ Yakımı)</option>
                                <option value="Maintenance">Hedef: Koruma</option>
                            </select>
                            <select id="profActivity">
                                <option value="1.55">Aktivite: Orta (Haftada 3-5 Gün İdman)</option>
                                <option value="1.725">Aktivite: Yüksek (Haftada 6 Gün İdman)</option>
                                <option value="1.375">Aktivite: Düşük (Haftada 1-2 Gün İdman)</option>
                            </select>
                        </div>

                        <div class="panel-header" style="margin-top:4px; font-size:0.85rem;">
                            <span>📏 Vücut Çevre Ölçüleri (cm)</span>
                        </div>
                        <div class="form-grid-3x1">
                            <input type="number" id="profArm" placeholder="Kol (cm)" step="0.5" />
                            <input type="number" id="profWaist" placeholder="Bel (cm)" step="0.5" />
                            <input type="number" id="profShoulder" placeholder="Omuz (cm)" step="0.5" />
                        </div>

                        <button class="btn-log" onclick="saveUserProfile()">Profili & Hedefleri Kaydet</button>
                    </div>
                </div>

                <div class="panel-card" style="flex:1;">
                    <div class="panel-header">
                        <span>🎯 Önerilen Günlük Hedefler</span>
                    </div>
                    <div class="macro-stat-grid">
                        <div class="macro-card">
                            <div class="macro-label">Hedef Kalori</div>
                            <div class="macro-val macro-c-cal" id="calcTargetCal">0 kcal</div>
                        </div>
                        <div class="macro-card">
                            <div class="macro-label">Protein (x2.2)</div>
                            <div class="macro-val macro-c-pro" id="calcTargetPro">0g</div>
                        </div>
                        <div class="macro-card">
                            <div class="macro-label">Hedef Karb</div>
                            <div class="macro-val macro-c-carb" id="calcTargetCarb">0g</div>
                        </div>
                        <div class="macro-card">
                            <div class="macro-label">Hedef Yağ</div>
                            <div class="macro-val macro-c-fat" id="calcTargetFat">0g</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="overload-col-right">
                <div class="panel-card" style="height: 100%;">
                    <div class="phase-header-bar">
                        <div style="display:flex; align-items:center; gap:8px;">
                            <span style="font-size:0.85rem; font-weight:800; color:#fff;">Dönem:</span>
                            <select class="phase-selector" id="phaseSelectDropdown" onchange="switchPhase(this.value)"></select>
                        </div>
                        <div style="display:flex; gap:6px;">
                            <button onclick="createNewPhasePrompt()" style="background:#00f2fe; color:#000; border:none; padding:6px 12px; border-radius:6px; font-weight:800; font-size:0.75rem; cursor:pointer;">➕ Yeni Dönem Başlat</button>
                            <button onclick="deleteCurrentPhase()" style="background:#ef4444; color:#fff; border:none; padding:6px 10px; border-radius:6px; font-weight:700; font-size:0.75rem; cursor:pointer;">🗑️</button>
                        </div>
                    </div>

                    <div class="photo-matrix-4x">
                        <div class="photo-card-slot" id="slot_before_front" onclick="triggerSlotUpload('before_front')">
                            <div class="slot-badge">BEFORE • FRONT (ÖN)</div>
                            <button class="btn-remove-photo" id="btn_rem_before_front" onclick="removePhoto(event, 'before_front')">✕</button>
                            <img id="img_before_front" src="" style="display:none;" />
                            <div class="slot-placeholder" id="hint_before_front">
                                <div style="font-size:1.6rem; margin-bottom:4px;">📷</div>
                                <div>Ön Form Yükle</div>
                            </div>
                        </div>

                        <div class="photo-card-slot" id="slot_after_front" onclick="triggerSlotUpload('after_front')">
                            <div class="slot-badge" style="border-color:#10b981; color:#10b981;">AFTER • FRONT (ÖN)</div>
                            <button class="btn-remove-photo" id="btn_rem_after_front" onclick="removePhoto(event, 'after_front')">✕</button>
                            <img id="img_after_front" src="" style="display:none;" />
                            <div class="slot-placeholder" id="hint_after_front">
                                <div style="font-size:1.6rem; margin-bottom:4px;">🔥</div>
                                <div>Güncel Ön Form</div>
                            </div>
                        </div>

                        <div class="photo-card-slot" id="slot_before_back" onclick="triggerSlotUpload('before_back')">
                            <div class="slot-badge">BEFORE • BACK/SIDE (SIRT)</div>
                            <button class="btn-remove-photo" id="btn_rem_before_back" onclick="removePhoto(event, 'before_back')">✕</button>
                            <img id="img_before_back" src="" style="display:none;" />
                            <div class="slot-placeholder" id="hint_before_back">
                                <div style="font-size:1.6rem; margin-bottom:4px;">📷</div>
                                <div>Sırt/Yan Form Yükle</div>
                            </div>
                        </div>

                        <div class="photo-card-slot" id="slot_after_back" onclick="triggerSlotUpload('after_back')">
                            <div class="slot-badge" style="border-color:#10b981; color:#10b981;">AFTER • BACK/SIDE (SIRT)</div>
                            <button class="btn-remove-photo" id="btn_rem_after_back" onclick="removePhoto(event, 'after_back')">✕</button>
                            <img id="img_after_back" src="" style="display:none;" />
                            <div class="slot-placeholder" id="hint_after_back">
                                <div style="font-size:1.6rem; margin-bottom:4px;">🔥</div>
                                <div>Güncel Sırt/Yan Form</div>
                            </div>
                        </div>
                    </div>

                    <input type="file" id="universalPhotoInput" accept="image/*" onchange="handleUniversalPhotoUpload(event)" style="display:none;" />
                </div>
            </div>
        </div>

        <div class="view-panel" id="healthView">
            <div class="overload-col-left">
                <div class="recovery-banner" id="recoveryBannerBox">
                    <div class="recovery-circle" id="recoveryScoreDisplay">--<span>SKOR</span></div>
                    <div class="recovery-info">
                        <h3 id="recoveryStatusTitle">Toparlanma Durumu</h3>
                        <p id="recoveryAdviceText">Bugüne ait uyku, HRV ve dinlenik nabız verilerini kaydedin.</p>
                    </div>
                </div>

                <!-- AKTIF SAKATLIK & REHABILITASYON PANELI -->
                <div class="panel-card" style="border: 1px solid rgba(239, 68, 68, 0.4);">
                    <div class="panel-header">
                        <span style="color:#ef4444;">🩹 Aktif Sakatlık & Rehabilitasyon</span>
                        <span class="badge-cyan" id="activeInjuryCountBadge" style="border-color:#ef4444; color:#ef4444;">0 Aktif</span>
                    </div>
                    <div class="input-form">
                        <div class="form-grid-2x2">
                            <select id="injuryArea">
                                <option value="Omuz (Rotator Cuff / Ön Omuz)">Omuz (Rotator / Ön Omuz)</option>
                                <option value="Dirsek (Tendinit / Medial-Lateral)">Dirsek (Tendinit)</option>
                                <option value="Bel (Lower Back / Disk)">Bel (Lower Back)</option>
                                <option value="Diz (Patellar / Menisküs)">Diz (Patellar / Eklem)</option>
                                <option value="Bilek (Wrist)">Bilek (Wrist)</option>
                                <option value="Göğüs (Pec Bağlantısı)">Göğüs (Pec Bağlantısı)</option>
                                <option value="Diğer">Diğer Bölge</option>
                            </select>
                            <select id="injurySeverity">
                                <option value="Hafif Sızı (RIR 2-3 Koru)">Hafif Sızı (1-3 / 10)</option>
                                <option value="Orta Derece Rahatsızlık (Hareketi Değiştir)">Orta Rahatsızlık (4-6 / 10)</option>
                                <option value="Ciddi Ağrı (O Bölgeyi Tamamen Dinlendir)">Ciddi Ağrı (7-10 / 10)</option>
                            </select>
                        </div>
                        <input type="text" id="injuryDetails" placeholder="Tetikleyen hareket veya detay (Örn: 30kg Dumbbell Press'te batma)" />
                        <button class="btn-log" onclick="saveInjuryLog()" style="background:#ef4444; color:#fff;">Sakatlığı Kaydet & Koça Bildir</button>
                    </div>
                    <div class="history-list" id="injuryListDisplay" style="max-height:160px; margin-top:4px;"></div>
                </div>

                <div class="panel-card">
                    <div class="panel-header">
                        <span>📲 Günlük Biyometrik Veri Girişi</span>
                        <span class="badge-cyan" id="healthSelectedDateBadge">Bugün</span>
                    </div>
                    <div class="input-form">
                        <div class="form-grid-2x2">
                            <input type="number" id="healthSleep" placeholder="😴 Toplam Uyku (Saat, örn: 7.5)" step="0.1" />
                            <input type="number" id="healthDeepSleep" placeholder="💤 Derin Uyku (Saat, örn: 1.5)" step="0.1" />
                        </div>
                        <div class="form-grid-2x2">
                            <input type="number" id="healthHrv" placeholder="💓 HRV (ms, örn: 65)" step="1" />
                            <input type="number" id="healthRestingHr" placeholder="🫀 Dinlenik Nabız (BPM, örn: 54)" step="1" />
                        </div>
                        <div class="form-grid-3x1">
                            <input type="number" id="healthAvgWorkoutHr" placeholder="🏋️ İdman Ort. Nabız" step="1" />
                            <input type="number" id="healthMaxWorkoutHr" placeholder="🔥 İdman Max Nabız" step="1" />
                            <input type="number" id="healthSteps" placeholder="👟 Günlük Adım" step="100" />
                        </div>
                        <button class="btn-log" onclick="saveManualHealthData()">Sağlık Verilerini Kaydet & Skoru Hesapla</button>
                    </div>
                </div>

                <div class="guide-btn-card" onclick="toggleGuideModal(true)">
                    <div style="display:flex; align-items:center; gap:12px;">
                        <div style="font-size:1.6rem;">⌚</div>
                        <div>
                            <div style="font-size:0.9rem; font-weight:800; color:#fff;">Apple Watch Otomasyon Rehberi</div>
                            <div style="font-size:0.75rem; color:#9ca3af;">Verilerin her sabah otomatik akması için tıkla</div>
                        </div>
                    </div>
                    <div style="color:#00f2fe; font-size:1.1rem; font-weight:800;">→</div>
                </div>
            </div>

            <div class="overload-col-right">
                <div class="panel-card" style="height: 100%;">
                    <div class="panel-header">
                        <span>📈 7 Günlük HRV & Dinlenik Nabız Trendi</span>
                    </div>
                    <div style="flex:1; min-height:380px; position:relative;">
                        <canvas id="healthTrendChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

    </div>

    <script>
        function compressImage(file, maxWidth = 800, quality = 0.7) {
            return new Promise((resolve) => {
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onload = (event) => {
                    const img = new Image();
                    img.src = event.target.result;
                    img.onload = () => {
                        let width = img.width;
                        let height = img.height;
                        if (width > maxWidth) {
                            height = Math.round((height * maxWidth) / width);
                            width = maxWidth;
                        }
                        const canvas = document.createElement('canvas');
                        canvas.width = width;
                        canvas.height = height;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0, width, height);
                        resolve(canvas.toDataURL('image/jpeg', quality));
                    };
                };
            });
        }

        function getMondayOfWeek(d) {
            d = new Date(d);
            var day = d.getDay(),
                diff = d.getDate() - day + (day === 0 ? -6 : 1);
            var mon = new Date(d.setDate(diff));
            mon.setHours(0, 0, 0, 0);
            return mon;
        }

        const mondayObj = getMondayOfWeek(new Date());
        let currentWeekKey = mondayObj.toISOString().split('T')[0];
        const todayKey = new Date().toLocaleDateString('tr-TR', { day: '2-digit', month: '2-digit', year: 'numeric' });

        const dayNames = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"];
        let weekDaysData = [];

        function buildWeekDays(mondayDate) {
            weekDaysData = [];
            for (let i = 0; i < 7; i++) {
                const d = new Date(mondayDate);
                d.setDate(mondayDate.getDate() + i);
                const fullDate = d.toLocaleDateString('tr-TR', { day: '2-digit', month: '2-digit', year: 'numeric' });
                const shortDate = d.toLocaleDateString('tr-TR', { day: '2-digit', month: '2-digit' });
                weekDaysData.push({ index: i, dayName: dayNames[i], fullDate: fullDate, shortDate: shortDate });
            }
        }
        buildWeekDays(mondayObj);

        let selectedWorkoutDayIdx = weekDaysData.findIndex(item => item.fullDate === todayKey);
        if (selectedWorkoutDayIdx === -1) selectedWorkoutDayIdx = 0;

        let selectedNutriDayIdx = weekDaysData.findIndex(item => item.fullDate === todayKey);
        if (selectedNutriDayIdx === -1) selectedNutriDayIdx = 0;

        function getStorageUsers() { return JSON.parse(localStorage.getItem("app_registered_users") || "{}"); }
        function saveStorageUsers(users) { localStorage.setItem("app_registered_users", JSON.stringify(users)); }

        function getAllUserWeeks(username) {
            return JSON.parse(localStorage.getItem("user_weeks_" + username) || "{}");
        }

        function getUserWeeklyLogs(username) {
            const allWeeks = getAllUserWeeks(username);
            return allWeeks[currentWeekKey] || [];
        }
        function saveUserWeeklyLogs(username, logs) {
            const allWeeks = getAllUserWeeks(username);
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

        function getUserProfileData(username) { return JSON.parse(localStorage.getItem("user_profile_" + username) || "{}"); }
        function saveUserProfileData(username, profData) { localStorage.setItem("user_profile_" + username, JSON.stringify(profData)); }

        function getUserPhases(username) { return JSON.parse(localStorage.getItem("user_phases_" + username) || "[]"); }
        function saveUserPhases(username, phases) { localStorage.setItem("user_phases_" + username, JSON.stringify(phases)); }

        function getUserHealthLogs(username) { return JSON.parse(localStorage.getItem("user_health_" + username) || "{}"); }
        function saveUserHealthLogs(username, healthLogs) { localStorage.setItem("user_health_" + username, JSON.stringify(healthLogs)); }

        function getUserInjuries(username) { return JSON.parse(localStorage.getItem("user_injuries_" + username) || "[]"); }
        function saveUserInjuries(username, injuries) { localStorage.setItem("user_injuries_" + username, JSON.stringify(injuries)); }

        let currentUser = JSON.parse(localStorage.getItem("active_user") || "null");
        let isRegisterMode = false;
        let weeklyLogs = [];
        let weeklyNutrition = {};
        let userProfile = {};
        let userPhases = [];
        let userHealthLogs = {};
        let activePhaseId = null;
        let pendingUploadSlot = null;
        let chartInstance = null;
        let healthChartInstance = null;

        document.getElementById("exerciseDate").value = weekDaysData[selectedWorkoutDayIdx].fullDate;

        function populateWeekSelector() {
            const select = document.getElementById("weekSelectorDropdown");
            if (!select || !currentUser) return;
            select.innerHTML = "";
            const allWeeks = getAllUserWeeks(currentUser.username);
            const weekKeys = Object.keys(allWeeks);
            if (!weekKeys.includes(currentWeekKey)) weekKeys.push(currentWeekKey);
            weekKeys.sort().reverse();

            weekKeys.forEach(wk => {
                const opt = document.createElement("option");
                opt.value = wk;
                opt.innerText = wk === currentWeekKey ? `${wk} (Bu Hafta)` : wk;
                if (wk === currentWeekKey) opt.selected = true;
                select.appendChild(opt);
            });
        }

        function changeActiveWeek(selectedWeek) {
            currentWeekKey = selectedWeek;
            const parts = selectedWeek.split("-");
            const mon = new Date(parts[0], parts[1] - 1, parts[2]);
            buildWeekDays(mon);
            selectedWorkoutDayIdx = 0;
            selectedNutriDayIdx = 0;
            document.getElementById("exerciseDate").value = weekDaysData[0].fullDate;
            loadUserWorkouts();
            loadUserNutrition();
        }

        function toggleGuideModal(show) {
            const modal = document.getElementById("appleWatchModal");
            modal.style.display = show ? "flex" : "none";
            if (show) {
                const currentOrigin = window.location.origin;
                document.getElementById("webhookUrlText").innerText = `${currentOrigin}/api/health-sync`;
            }
        }

        function closeGuideModal(e) {
            if (e.target.id === "appleWatchModal") toggleGuideModal(false);
        }

        function toggleAuditModal(show) {
            const modal = document.getElementById("coachAuditModal");
            modal.style.display = show ? "flex" : "none";
        }

        function closeAuditModal(e) {
            if (e.target.id === "coachAuditModal") toggleAuditModal(false);
        }

        function copyWebhookUrl() {
            const url = document.getElementById("webhookUrlText").innerText;
            navigator.clipboard.writeText(url).then(() => {
                alert("Webhook URL adresi kopyalandı kral!");
            });
        }

        async function triggerCoachAudit() {
            if (!currentUser) return alert("Lütfen önce giriş yap kral!");
            toggleAuditModal(true);

            const loadEl = document.getElementById("auditLoadingState");
            const textEl = document.getElementById("auditContentText");
            loadEl.style.display = "block";
            textEl.style.display = "none";

            const prof = getUserProfileData(currentUser.username) || {};
            const wLogs = getUserWeeklyLogs(currentUser.username) || [];
            const nutri = getUserWeeklyNutrition(currentUser.username) || {};
            const health = getUserHealthLogs(currentUser.username) || {};
            const injuries = getUserInjuries(currentUser.username) || [];

            const hasProfile = prof.weight || prof.height;
            const hasWorkouts = Array.isArray(wLogs) && wLogs.length > 0;
            const hasNutri = Object.keys(nutri).length > 0;
            const hasHealth = Object.keys(health).length > 0;
            const hasInjuries = injuries.length > 0;

            if (!hasProfile && !hasWorkouts && !hasNutri && !hasHealth && !hasInjuries) {
                loadEl.style.display = "none";
                textEl.style.display = "block";
                textEl.innerHTML = "<b>Henüz yeterli veri girişi yapmadın kral.</b><br><br>Sana özel haftalık karne çıkarabilmem için:<br>• <b>Profil</b> bilgilerini kaydetmeli,<br>• <b>Overload</b> sekmesinden birkaç set veya <b>Beslenme</b> öğünü girmelisin.<br><br>Verilerini girdikten sonra tekrar dene!";
                return;
            }

            try {
                const res = await fetch("/coach-audit", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        profile_data: prof,
                        recent_workouts: wLogs,
                        recent_nutrition: nutri,
                        recent_health: health,
                        active_injuries: injuries
                    })
                });
                const data = await res.json();
                loadEl.style.display = "none";
                textEl.style.display = "block";
                if (data.is_error) {
                    textEl.innerHTML = `<span style="color:#ef4444; font-weight:700;">HATA DETAYI:</span><br><pre style="white-space:pre-wrap; margin-top:8px;">${data.audit_report}</pre>`;
                } else {
                    textEl.innerHTML = (data.audit_report || "Değerlendirme alınamadı.").replace(/\n/g, "<br>").replace(/\*\*(.*?)\*\*/g, "<b>$1</b>");
                }
            } catch (err) {
                loadEl.style.display = "none";
                textEl.style.display = "block";
                textEl.innerHTML = `<span style="color:#ef4444;">İstemci bağlantı hatası: ${err.message}</span>`;
            }
        }

        function openView(viewName) {
            document.querySelectorAll(".view-panel").forEach(p => p.classList.remove("active"));
            const target = document.getElementById(viewName + "View");
            if (target) target.classList.add("active");

            document.getElementById("backHubBtn").style.display = (viewName === 'hub') ? 'none' : 'block';

            if (viewName === 'overload') setTimeout(updateChart, 150);
            if (viewName === 'nutrition') { renderNutriDayTabs(); renderSelectedDayNutrition(); }
            if (viewName === 'profile') { loadUserProfileUI(); loadUserPhasesUI(); }
            if (viewName === 'health') { loadHealthUI(); renderInjuriesUI(); }
        }

        function checkAuth() {
            if (!currentUser) {
                document.getElementById("authOverlay").style.display = "flex";
            } else {
                document.getElementById("authOverlay").style.display = "none";
                document.getElementById("activeUserName").innerText = "👤 " + currentUser.username;
                populateWeekSelector();
                loadUserWorkouts();
                loadUserNutrition();
                loadUserProfileUI();
                loadUserPhasesUI();
                loadHealthUI();
                renderInjuriesUI();
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

        // ================= SAKATLIK & REHABILITASYON =================
        function saveInjuryLog() {
            if (!currentUser) return;
            const area = document.getElementById("injuryArea").value;
            const severity = document.getElementById("injurySeverity").value;
            const details = document.getElementById("injuryDetails").value.trim();

            const injuries = getUserInjuries(currentUser.username);
            injuries.unshift({
                id: Date.now(),
                area: area,
                severity: severity,
                details: details || "Detay belirtilmedi",
                date: todayKey
            });
            saveUserInjuries(currentUser.username, injuries);
            document.getElementById("injuryDetails").value = "";
            renderInjuriesUI();
            alert("Sakatlık kaydı alındı! AI Koç antrenman ve önerilerini bu kısıtlamaya göre uyarlayacak kral.");
        }

        function resolveInjury(id) {
            let injuries = getUserInjuries(currentUser.username);
            injuries = injuries.filter(inj => inj.id !== id);
            saveUserInjuries(currentUser.username, injuries);
            renderInjuriesUI();
        }

        function renderInjuriesUI() {
            const list = document.getElementById("injuryListDisplay");
            const badge = document.getElementById("activeInjuryCountBadge");
            if (!list || !currentUser) return;
            
            const injuries = getUserInjuries(currentUser.username);
            badge.innerText = `${injuries.length} Aktif`;
            list.innerHTML = "";

            if (injuries.length === 0) {
                list.innerHTML = `<div style="font-size:0.75rem; color:#6b7280; text-align:center; padding:10px;">Aktif sakatlık kaydı yok. Vücut sağlam! 🦍</div>`;
                return;
            }

            injuries.forEach(inj => {
                list.innerHTML += `
                    <div class="log-item" style="border-left: 3px solid #ef4444; flex-direction:column; align-items:flex-start; gap:4px;">
                        <div style="display:flex; justify-content:space-between; width:100%; align-items:center;">
                            <span style="font-weight:700; color:#ef4444; font-size:0.8rem;">🚨 ${inj.area}</span>
                            <button onclick="resolveInjury(${inj.id})" style="color:#10b981; font-weight:700; font-size:0.75rem;">İyileşti ✓</button>
                        </div>
                        <div style="font-size:0.75rem; color:#d1d5db;">${inj.details} <span style="color:#9ca3af;">(${inj.severity})</span></div>
                    </div>
                `;
            });
        }

        // ================= HEALTH & RECOVERY =================
        function loadHealthUI() {
            if (!currentUser) return;
            userHealthLogs = getUserHealthLogs(currentUser.username);
            document.getElementById("healthSelectedDateBadge").innerText = todayKey;

            const todayLog = userHealthLogs[todayKey];
            if (todayLog) {
                document.getElementById("healthSleep").value = todayLog.sleep_hours || "";
                document.getElementById("healthDeepSleep").value = todayLog.deep_sleep_hours || "";
                document.getElementById("healthHrv").value = todayLog.hrv_ms || "";
                document.getElementById("healthRestingHr").value = todayLog.resting_hr || "";
                document.getElementById("healthAvgWorkoutHr").value = todayLog.avg_workout_hr || "";
                document.getElementById("healthMaxWorkoutHr").value = todayLog.max_workout_hr || "";
                document.getElementById("healthSteps").value = todayLog.steps || "";

                renderRecoveryScoreUI(todayLog);
            }
            updateHealthTrendChart();
        }

        function renderRecoveryScoreUI(log) {
            const sleep = parseFloat(log.sleep_hours) || 7.0;
            const hrv = parseFloat(log.hrv_ms) || 60.0;
            const rhr = parseFloat(log.resting_hr) || 55.0;

            const sleepScore = Math.min(40.0, (sleep / 8.0) * 40.0);
            const hrvScore = Math.min(35.0, (hrv / 75.0) * 35.0);
            let rhrScore = 25.0;
            if (rhr > 60) rhrScore = Math.max(5.0, 25.0 - (rhr - 60) * 0.8);

            const total = Math.round(Math.min(100.0, Math.max(10.0, sleepScore + hrvScore + rhrScore)));

            const scoreCircle = document.getElementById("recoveryScoreDisplay");
            scoreCircle.innerHTML = `${total}<span>SKOR</span>`;

            let color = "#00f2fe";
            let title = "Orta / Yeterli Toparlanma ⚡";
            let advice = "Vücudun antrenmana hazır. Formu bozmadan setlerde 1 tekrar cepte bırak (RIR 1).";

            if (total >= 80) {
                color = "#10b981";
                title = "Optimal Toparlanma 🔥";
                advice = "Merkezi sinir sistemin zirvede! Bahanen sıfır, bugün ağırlıkların içinden geç.";
            } else if (total < 60) {
                color = "#ef4444";
                title = "Yetersiz Toparlanma / Yüksek Stres ⚠️";
                advice = "Otonom sinir sistemin yorgun. Sakatlanmamak için PR zorlama, form ve hipertrofi odaklı kal.";
            }

            scoreCircle.style.borderColor = color;
            document.getElementById("recoveryStatusTitle").innerText = title;
            document.getElementById("recoveryStatusTitle").style.color = color;
            document.getElementById("recoveryAdviceText").innerText = advice;
        }

        function saveManualHealthData() {
            if (!currentUser) return;
            const sleep = parseFloat(document.getElementById("healthSleep").value) || 0;
            const deepSleep = parseFloat(document.getElementById("healthDeepSleep").value) || 0;
            const hrv = parseFloat(document.getElementById("healthHrv").value) || 0;
            const rhr = parseFloat(document.getElementById("healthRestingHr").value) || 0;
            const avgHr = parseFloat(document.getElementById("healthAvgWorkoutHr").value) || null;
            const maxHr = parseFloat(document.getElementById("healthMaxWorkoutHr").value) || null;
            const steps = parseInt(document.getElementById("healthSteps").value, 10) || 0;

            if (sleep <= 0 || hrv <= 0 || rhr <= 0) {
                return alert("Lütfen en az Uyku Süresi, HRV ve Dinlenik Nabız değerlerini gir kral!");
            }

            const log = {
                date: todayKey,
                sleep_hours: sleep,
                deep_sleep_hours: deepSleep,
                hrv_ms: hrv,
                resting_hr: rhr,
                avg_workout_hr: avgHr,
                max_workout_hr: maxHr,
                steps: steps
            };

            userHealthLogs[todayKey] = log;
            saveUserHealthLogs(currentUser.username, userHealthLogs);
            renderRecoveryScoreUI(log);
            updateHealthTrendChart();
            alert("Sağlık ve toparlanma verilerin başarıyla kaydedildi! 🫀");
        }

        function updateHealthTrendChart() {
            const dates = weekDaysData.map(d => d.fullDate);
            const hrvData = dates.map(d => userHealthLogs[d] ? userHealthLogs[d].hrv_ms : null);
            const rhrData = dates.map(d => userHealthLogs[d] ? userHealthLogs[d].resting_hr : null);

            const canvas = document.getElementById('healthTrendChart');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (healthChartInstance) healthChartInstance.destroy();

            healthChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: weekDaysData.map(d => d.dayName + " (" + d.shortDate + ")"),
                    datasets: [
                        {
                            label: 'HRV (ms)',
                            data: hrvData,
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.12)',
                            borderWidth: 3,
                            yAxisID: 'y',
                            tension: 0.35,
                            fill: true,
                            pointBackgroundColor: '#10b981',
                            pointRadius: 5
                        },
                        {
                            label: 'Dinlenik Nabız (BPM)',
                            data: rhrData,
                            borderColor: '#ef4444',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            borderDash: [4, 4],
                            yAxisID: 'y1',
                            tension: 0.35,
                            pointBackgroundColor: '#ef4444',
                            pointRadius: 5
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: '#1a2230' }, ticks: { color: '#9ca3af' } },
                        y: { type: 'linear', position: 'left', grid: { color: '#1a2230' }, ticks: { color: '#10b981' }, title: { display: true, text: 'HRV (ms)', color: '#10b981' } },
                        y1: { type: 'linear', position: 'right', grid: { display: false }, ticks: { color: '#ef4444' }, title: { display: true, text: 'Dinlenik Nabız (BPM)', color: '#ef4444' } }
                    },
                    plugins: { legend: { labels: { color: '#e5e7eb', font: { size: 11, weight: 'bold' } } } }
                }
            });
        }

        // ================= PROFİL & BEFORE/AFTER =================
        function loadUserProfileUI() {
            if (!currentUser) return;
            userProfile = getUserProfileData(currentUser.username);

            document.getElementById("profFullName").value = userProfile.fullName || "";
            document.getElementById("profAge").value = userProfile.age || "";
            document.getElementById("profHeight").value = userProfile.height || "";
            document.getElementById("profWeight").value = userProfile.weight || "";
            document.getElementById("profBodyfat").value = userProfile.bodyfat || "";
            if (userProfile.goal) document.getElementById("profGoal").value = userProfile.goal;
            if (userProfile.activity) document.getElementById("profActivity").value = userProfile.activity;

            document.getElementById("profArm").value = userProfile.arm || "";
            document.getElementById("profWaist").value = userProfile.waist || "";
            document.getElementById("profShoulder").value = userProfile.shoulder || "";

            calculateMetabolismAndMacros();
        }

        function calculateMetabolismAndMacros() {
            const h = parseFloat(document.getElementById("profHeight").value) || 0;
            const w = parseFloat(document.getElementById("profWeight").value) || 0;
            const a = parseFloat(document.getElementById("profAge").value) || 22;
            const act = parseFloat(document.getElementById("profActivity").value) || 1.55;
            const goal = document.getElementById("profGoal").value;

            if (h > 0 && w > 0) {
                const bmr = 10 * w + 6.25 * h - 5 * a + 5;
                const tdee = bmr * act;

                let targetCal = tdee;
                if (goal === "Lean Bulk") targetCal += 300;
                else if (goal === "Aggressive Cut") targetCal -= 500;
                else if (goal === "Recomposition") targetCal -= 150;

                const proteinGrams = Math.round(w * 2.2);
                const fatGrams = Math.round((targetCal * 0.25) / 9);
                const carbGrams = Math.max(0, Math.round((targetCal - (proteinGrams * 4 + fatGrams * 9)) / 4));

                document.getElementById("bmrCalculatedBadge").innerText = `BMR: ${Math.round(bmr)} kcal | TDEE: ${Math.round(tdee)} kcal`;
                document.getElementById("calcTargetCal").innerText = `${Math.round(targetCal)} kcal`;
                document.getElementById("calcTargetPro").innerText = `${proteinGrams}g`;
                document.getElementById("calcTargetCarb").innerText = `${carbGrams}g`;
                document.getElementById("calcTargetFat").innerText = `${fatGrams}g`;
            }
        }

        function saveUserProfile() {
            if (!currentUser) return;
            userProfile.fullName = document.getElementById("profFullName").value.trim();
            userProfile.age = document.getElementById("profAge").value;
            userProfile.height = document.getElementById("profHeight").value;
            userProfile.weight = document.getElementById("profWeight").value;
            userProfile.bodyfat = document.getElementById("profBodyfat").value;
            userProfile.goal = document.getElementById("profGoal").value;
            userProfile.activity = document.getElementById("profActivity").value;

            userProfile.arm = document.getElementById("profArm").value;
            userProfile.waist = document.getElementById("profWaist").value;
            userProfile.shoulder = document.getElementById("profShoulder").value;

            saveUserProfileData(currentUser.username, userProfile);
            calculateMetabolismAndMacros();
            alert("Profil bilgileri ve hedeflerin başarıyla kaydedildi kral! 🦍");
        }

        function loadUserPhasesUI() {
            if (!currentUser) return;
            userPhases = getUserPhases(currentUser.username);

            if (userPhases.length === 0) {
                const defaultPhase = {
                    id: "phase_" + Date.now(),
                    name: "1. Dönem",
                    photos: { before_front: null, before_back: null, after_front: null, after_back: null }
                };
                userPhases.push(defaultPhase);
                saveUserPhases(currentUser.username, userPhases);
            }

            if (!activePhaseId || !userPhases.find(p => p.id === activePhaseId)) {
                activePhaseId = userPhases[0].id;
            }

            renderPhaseDropdown();
            renderActivePhasePhotos();
        }

        function renderPhaseDropdown() {
            const select = document.getElementById("phaseSelectDropdown");
            if (!select) return;
            select.innerHTML = "";
            userPhases.forEach(p => {
                const opt = document.createElement("option");
                opt.value = p.id;
                opt.innerText = p.name;
                if (p.id === activePhaseId) opt.selected = true;
                select.appendChild(opt);
            });
        }

        function switchPhase(phaseId) {
            activePhaseId = phaseId;
            renderActivePhasePhotos();
        }

        function createNewPhasePrompt() {
            const name = prompt("Yeni Dönem Adı (Örn: '3 Temmuz - 10 Ağustos Lean Bulk'):");
            if (!name || !name.trim()) return;

            const newPhase = {
                id: "phase_" + Date.now(),
                name: name.trim(),
                photos: { before_front: null, before_back: null, after_front: null, after_back: null }
            };
            userPhases.unshift(newPhase);
            saveUserPhases(currentUser.username, userPhases);
            activePhaseId = newPhase.id;
            renderPhaseDropdown();
            renderActivePhasePhotos();
        }

        function deleteCurrentPhase() {
            if (userPhases.length <= 1) return alert("En az bir dönem bulunmalıdır kral!");
            if (!confirm("Bu dönemi ve içindeki fotoğrafları silmek istiyor musun?")) return;

            userPhases = userPhases.filter(p => p.id !== activePhaseId);
            saveUserPhases(currentUser.username, userPhases);
            activePhaseId = userPhases[0].id;
            renderPhaseDropdown();
            renderActivePhasePhotos();
        }

        function renderActivePhasePhotos() {
            const phase = userPhases.find(p => p.id === activePhaseId);
            if (!phase) return;

            const slots = ["before_front", "before_back", "after_front", "after_back"];
            slots.forEach(slotKey => {
                const imgEl = document.getElementById("img_" + slotKey);
                const hintEl = document.getElementById("hint_" + slotKey);
                const btnRem = document.getElementById("btn_rem_" + slotKey);
                const photoSrc = phase.photos ? phase.photos[slotKey] : null;

                if (photoSrc) {
                    imgEl.src = photoSrc;
                    imgEl.style.display = "block";
                    hintEl.style.display = "none";
                    btnRem.style.display = "block";
                } else {
                    imgEl.src = "";
                    imgEl.style.display = "none";
                    hintEl.style.display = "block";
                    btnRem.style.display = "none";
                }
            });
        }

        function triggerSlotUpload(slotKey) {
            pendingUploadSlot = slotKey;
            document.getElementById("universalPhotoInput").click();
        }

        async function handleUniversalPhotoUpload(event) {
            const file = event.target.files[0];
            if (!file || !pendingUploadSlot) return;

            const compressedBase64 = await compressImage(file, 800, 0.7);
            const phase = userPhases.find(p => p.id === activePhaseId);
            if (phase) {
                if (!phase.photos) phase.photos = {};
                phase.photos[pendingUploadSlot] = compressedBase64;
                saveUserPhases(currentUser.username, userPhases);
                renderActivePhasePhotos();
            }
            pendingUploadSlot = null;
            document.getElementById("universalPhotoInput").value = "";
        }

        function removePhoto(event, slotKey) {
            event.stopPropagation();
            const phase = userPhases.find(p => p.id === activePhaseId);
            if (phase && phase.photos) {
                phase.photos[slotKey] = null;
                saveUserPhases(currentUser.username, userPhases);
                renderActivePhasePhotos();
            }
        }

        // ================= ANTRENMAN YÖNETİMİ =================
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
                    <div class="empty-day-box" style="text-align:center; padding:20px; color:#6b7280;">
                        <div style="font-size:1.8rem; margin-bottom:4px;">😴</div>
                        <div style="font-weight:700; color:#9ca3af;">Dinlenme Günü (Off Day)</div>
                        <div style="font-size:0.75rem;">${currentDay.fullDate} tarihinde kayıtlı set yok kral.</div>
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

        function getAllWeeksWorkouts() {
            if (!currentUser) return [];
            const allWeeks = getAllUserWeeks(currentUser.username);
            let combined = [];
            Object.values(allWeeks).forEach(arr => {
                if (Array.isArray(arr)) combined = combined.concat(arr);
            });
            return combined;
        }

        function populateDropdown() {
            const select = document.getElementById("chartExerciseSelect");
            if (!select) return;
            const currentSelected = select.value;
            const allLogs = getAllWeeksWorkouts();
            const unique = [...new Set(allLogs.map(item => item.exercise))];

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
            const allLogs = getAllWeeksWorkouts();
            const filtered = allLogs.filter(item => item.exercise === selectedEx);

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

        // ================= BESLENME YÖNETİMİ =================
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
                    <div class="empty-day-box" style="text-align:center; padding:20px; color:#6b7280;">
                        <div style="font-size:1.8rem; margin-bottom:4px;">🍽️</div>
                        <div style="font-weight:700; color:#9ca3af;">Öğün Girilmedi</div>
                        <div style="font-size:0.75rem;">${currentDay.fullDate} tarihi için henüz yemek kaydedilmedi.</div>
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

        // ================= CHAT & VISION =================
        let conversationHistory = [];
        let selectedBase64Image = null;
        let nutriSelectedImage = null;

        async function handleImageSelect(event, type) {
            const file = event.target.files[0];
            if (!file) return;
            const compressed = await compressImage(file, 800, 0.7);
            if (type === 'coach') {
                selectedBase64Image = compressed;
                document.getElementById("imagePreview").src = selectedBase64Image;
                document.getElementById("previewBox").style.display = "flex";
            } else {
                nutriSelectedImage = compressed;
                document.getElementById("nutriImagePreview").src = nutriSelectedImage;
                document.getElementById("nutriPreviewBox").style.display = "flex";
            }
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
            userHtml += `<span>${text || "Görsel analizi"}</span>`;

            chatBox.innerHTML += `<div class="msg user">${userHtml}</div>`;
            const currentImg = selectedBase64Image;
            const currentText = text || "Bu görseli değerlendir kral.";

            input.value = "";
            clearImage();
            btn.disabled = true;
            chatBox.scrollTop = chatBox.scrollHeight;

            const loadingId = "load-" + Date.now();
            chatBox.innerHTML += `<div class="msg coach" id="${loadingId}"><i>Analiz ediliyor...</i></div>`;
            chatBox.scrollTop = chatBox.scrollHeight;

            const lastSets = weeklyLogs.slice(-6).map(s => `${s.exercise} (${s.set_num}.Set, ${s.date}): ${s.weight}kg x ${s.reps}`).join(", ");
            const profSummary = userProfile.weight ? `Boy: ${userProfile.height}cm, Kilo: ${userProfile.weight}kg, Yağ: %${userProfile.bodyfat || '?'}, Hedef: ${userProfile.goal}` : "Profil girilmedi.";
            
            const todayH = userHealthLogs[todayKey];
            const healthSummary = todayH ? `Uyku: ${todayH.sleep_hours}s, HRV: ${todayH.hrv_ms}ms, Dinlenik Nabız: ${todayH.resting_hr}bpm` : "Bugünkü sağlık/toparlanma verisi henüz girilmedi.";

            const injuries = getUserInjuries(currentUser.username);
            const injuriesSummary = injuries.length > 0 ? injuries.map(i => `${i.area} (${i.severity}: ${i.details})`).join(" | ") : "Aktif sakatlık yok.";

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        user_message: currentText,
                        workout_summary: lastSets,
                        user_profile_summary: profSummary,
                        health_summary: healthSummary,
                        injuries_summary: injuriesSummary,
                        image_base64: currentImg,
                        history: conversationHistory
                    })
                });
                const data = await response.json();
                const loadEl = document.getElementById(loadingId);
                
                if (data.is_error) {
                    loadEl.className = "msg coach error";
                    loadEl.innerHTML = `⚠️ <b>HATA DETAYI:</b><br>${data.coach_reply}`;
                } else {
                    let replyText = data.coach_reply || "Hedefe odaklan kral. Formunu koru ve kontrollü devam et.";
                    let replyFormatted = replyText.replace(/\n/g, "<br>").replace(/\*\*(.*?)\*\*/g, "<b>$1</b>");
                    loadEl.innerHTML = replyFormatted;
                    conversationHistory.push({ role: "user", content: currentText });
                    conversationHistory.push({ role: "assistant", content: replyText });
                    if (conversationHistory.length > 8) conversationHistory = conversationHistory.slice(-8);
                }
            } catch (err) {
                const loadEl = document.getElementById(loadingId);
                loadEl.className = "msg coach error";
                loadEl.innerText = `İstemci bağlantı hatası: ${err.message}`;
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

                let replyFormatted = (data.coach_reply || "").replace(/\n/g, "<br>").replace(/\*\*(.*?)\*\*/g, "<b>$1</b>");
                document.getElementById(loadingId).innerHTML = replyFormatted || "Yanıt alındı.";

                if (data.detected_meal && Number(data.detected_meal.calories) > 0) {
                    const newMeal = {
                        id: Date.now() + Math.floor(Math.random() * 1000),
                        food_name: data.detected_meal.food_name || "Öğün",
                        items_summary: data.detected_meal.items_summary || currentText,
                        calories: Math.round(Number(data.detected_meal.calories) || 0),
                        protein: Number(data.detected_meal.protein) || 0,
                        carbs: Number(data.detected_meal.carbs) || 0,
                        fat: Number(data.detected_meal.fat) || 0
                    };

                    if (!weeklyNutrition[targetDateStr]) {
                        weeklyNutrition[targetDateStr] = [];
                    }
                    weeklyNutrition[targetDateStr] = [...weeklyNutrition[targetDateStr], newMeal];
                    saveUserWeeklyNutrition(currentUser.username, weeklyNutrition);
                    renderSelectedDayNutrition();
                }
            } catch (err) {
                document.getElementById(loadingId).innerText = `Hata: ${err.message}`;
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

@app.post("/api/health-sync")
def sync_apple_health_webhook(payload: HealthSyncInput):
    recovery = compute_recovery_score(
        sleep_hours=payload.sleep_hours,
        hrv=payload.hrv_ms,
        resting_hr=payload.resting_hr
    )
    return {
        "status": "success",
        "message": f"{payload.date} tarihli Apple Health verisi işlendi.",
        "recovery_metrics": recovery
    }

@app.post("/coach-audit")
def full_coach_audit(payload: CoachAuditInput):
    if not client:
        return {
            "audit_report": "GROQ_API_KEY bulunamadı! Lütfen sunucu ortam değişkenine veya .env dosyasına ekle.",
            "is_error": True
        }

    prof = payload.profile_data or {}
    workouts = payload.recent_workouts or []
    nutrition = payload.recent_nutrition or {}
    health = payload.recent_health or {}
    injuries = payload.active_injuries or []

    audit_prompt = f"""
Sen hem halden anlayan bilge bir mentor, hem de sıfır bahane kabul eden sert ve disiplinli bir 'Looksmax & Hipertrofi Başantrenörü'sün.
KULLANICI VERİLERİ:
- Profil: {prof.get('fullName', 'Bilinmiyor')}, Boy: {prof.get('height', '-')}cm, Kilo: {prof.get('weight', '-')}kg, Hedef: {prof.get('goal', '-')}
- Setler: {json.dumps(workouts, ensure_ascii=False) if workouts else "GİRİLMEDİ"}
- Beslenme: {json.dumps(nutrition, ensure_ascii=False) if nutrition else "GİRİLMEDİ"}
- Sağlık: {json.dumps(health, ensure_ascii=False) if health else "GİRİLMEDİ"}
- Aktif Sakatlıklar: {json.dumps(injuries, ensure_ascii=False) if injuries else "YOK"}

KURALLAR:
1. ASLA düşünme adımı, tablo veya veri matrisi üretme.
2. Doğrudan net maddelerle koçluk karne raporunu dök:
- 🔥 **DURUM TESPİTİ:** Genel gidişat.
- 🩹 **SAKATLIK & REHABİLİTASYON:** Varsa güvenli hareket reçetesi ve izolasyon.
- 🏋️ **ANTRENMAN & OVERLOAD:** Ağırlıkların durumu ve zorlama emri.
- 🥗 **MUTFAK & DİSİPLİN:** Makroların denetimi.
- ⚡ **3 NET EMİR:** Bu hafta yapılacaklar.
"""
    try:
        active_model = get_best_available_model()
        completion = client.chat.completions.create(
            messages=[{"role": "system", "content": audit_prompt}],
            model=active_model,
            temperature=0.3,
            max_tokens=800
        )
        raw_report = completion.choices[0].message.content or ""
        clean_report = strip_thinking_and_tables(raw_report)
        return {"audit_report": clean_report, "is_error": False}
    except Exception as e:
        full_err = traceback.format_exc()
        logger.error(f"Coach audit hatasi:\n{full_err}")
        return {"audit_report": f"Sunucu Hatası: {str(e)}\n\n{full_err}", "is_error": True}

@app.post("/chat")
def coach_dialogue(data: ChatInput):
    if not client:
        return {
            "user_message": data.user_message,
            "coach_reply": "GROQ_API_KEY bulunamadı! Lütfen sunucu ortam değişkenine veya .env dosyasına ekle.",
            "is_error": True
        }

    user_context = f"Kullanıcının Bu Haftaki Son Setleri: {data.workout_summary}" if data.workout_summary else "Bu hafta henüz set girilmedi."
    profile_context = f"Kullanıcı Profili: {data.user_profile_summary}" if data.user_profile_summary else "Profil bilgisi girilmedi."
    health_context = f"Biyometrik Sağlık & Recovery Durumu: {data.health_summary}" if data.health_summary else "Sağlık verisi yok."
    injuries_context = f"Aktif Sakatlıklar: {data.injuries_summary}" if data.injuries_summary else "Aktif sakatlık kaydı yok."

    system_prompt = f"""
Sen sporcusunu çok iyi anlayan ama asla laubaliliğe izin vermeyen bilge ve disiplinli bir 'Looksmax & Hipertrofi Başantrenörü'sün.
KULLANICI: {profile_context}
SAĞLIK: {health_context}
SAKATLIK DURUMU: {injuries_context}
SETLER: {user_context}

FORMAT VE ÇIKTI KURALLARI (MUTLAK KURAL):
1. ASLA DÜŞÜNME ADIMI, TABLO VEYA İŞLEM LİSTESİ YAZMA.
2. Uzun uzun 'Konu / Durum / Öneri' tabloları dökmek KESİNLİKLE YASAKTIR.
3. Yanıtını doğrudan, net, maddeli ve vurucu bir özet / aksiyon planı olarak ver.
4. Sakatlık varsa: Güvenli alternatif açıyı söyle ve 1 rehabilitasyon egzersizi emret.
"""
    messages = [{"role": "system", "content": system_prompt}]
    for msg in data.history:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    if data.image_base64:
        user_msg = f"[Kullanıcı bir fotoğraf yükledi]: {data.user_message}"
    else:
        user_msg = data.user_message

    messages.append({"role": "user", "content": user_msg})

    try:
        active_model = get_best_available_model()
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=active_model,
            temperature=0.3,
            max_tokens=500,
        )
        raw_text = chat_completion.choices[0].message.content or ""
        clean_text = strip_thinking_and_tables(raw_text)

        if not clean_text:
            clean_text = "Hedefe odaklan kral. Formunu koru ve kontrollü devam et."

        return {"user_message": data.user_message, "coach_reply": clean_text, "is_error": False}
    except Exception as e:
        full_err = traceback.format_exc()
        logger.error(f"Chat hatasi:\n{full_err}")
        return {
            "user_message": data.user_message,
            "coach_reply": f"Groq API Hatası: {str(e)}",
            "is_error": True
        }

@app.post("/nutrition-chat")
def nutrition_dialogue(data: NutritionChatInput):
    detected_meal = parse_meal_with_llm(data.user_message)
    
    if detected_meal:
        reply_text = (
            f"Afiyet olsun kral! Girdiğin **{detected_meal['items_summary']}** başarıyla listene eklendi:\n"
            f"🔥 **{detected_meal['calories']} kcal** | "
            f"🥩 **{detected_meal['protein']}g Protein** | "
            f"🍞 **{detected_meal['carbs']}g Karb** | "
            f"🥑 **{detected_meal['fat']}g Yağ**"
        )
    else:
        reply_text = "Öğün bilgisi ayrıştırılamadı kral. Lütfen girdiğin yiyecekleri kontrol et."
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
