import os
import json
import re
import secrets
import base64
import urllib.request
import urllib.parse
import logging
import traceback
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from groq import Groq

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None

try:
    import jwt as pyjwt
except ImportError:
    pyjwt = None

try:
    import bcrypt
except ImportError:
    bcrypt = None

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

# ================= KNOWLEDGE BASE (RAG / CHROMA) =================
# ingest.py tarafindan olusturulan ./looksmax_db vektor veritabanini
# uygulama ayaga kalkarken bir kez yukler. Bu klasor yoksa (ingest.py
# hic calistirilmamissa) RAG sessizce devre disi kalir, uygulama
# normal calismaya devam eder.

VECTOR_DB_DIR = "./looksmax_db"
embedding_model = None
vector_db = None

def load_knowledge_base():
    global embedding_model, vector_db
    if not os.path.isdir(VECTOR_DB_DIR):
        logger.warning(
            f"'{VECTOR_DB_DIR}' bulunamadi. RAG/knowledge-base devre disi. "
            f"PDF/txt eklemek icin ./knowledge_base klasorune dosya koyup ingest.py calistirilmali."
        )
        return
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embedding_model)
        count = vector_db._collection.count() if hasattr(vector_db, "_collection") else "?"
        logger.info(f"Knowledge base (Chroma) yuklendi. Parca sayisi: {count}")
    except Exception as e:
        logger.error(f"Knowledge base yuklenemedi: {e}")
        traceback.print_exc()
        vector_db = None

load_knowledge_base()


def retrieve_knowledge_context(query: str, k: int = 4) -> str:
    """Verilen sorguya en yakin k parcayi knowledge base'den ceker.
    RAG yuklu degilse veya sorgu bossa sessizce bos string doner."""
    if not vector_db or not query or not query.strip():
        return ""
    try:
        results = vector_db.similarity_search(query, k=k)
        if not results:
            return ""
        chunks = []
        for i, doc in enumerate(results):
            source = doc.metadata.get("source", "bilinmeyen kaynak")
            source_name = os.path.basename(str(source))
            snippet = doc.page_content.strip()
            chunks.append(f"[Kaynak {i+1}: {source_name}]\n{snippet}")
        return "\n\n".join(chunks)
    except Exception as e:
        logger.error(f"Knowledge base arama hatasi: {e}")
        return ""

app = FastAPI(title="Looksmax Hub - Elite Performance & Coaching Engine")

# ================= GERÇEK BACKEND AUTH (POSTGRES + JWT) =================
# FAZ 1: Kullanici hesaplarini localStorage'dan gercek bir veritabanina tasima.
# TASARIM: Bu katman OPSIYONEL/KATMANLI calisir. DATABASE_URL ayarlanmamissa
# (veya psycopg2/jwt/bcrypt kurulu degilse) asagidaki tum fonksiyonlar sessizce
# "yapilandirilmadi" durumuna duser ve frontend otomatik olarak eski
# localStorage-only auth akisina geri doner. Yani bu ozellik hicbir seyi
# KIRMADAN eklenir; Neon/Supabase kurulunca kendiliginden devreye girer.

DATABASE_URL = os.getenv("DATABASE_URL")  # orn: Neon/Supabase postgres connection string
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 30

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    JWT_SECRET = secrets.token_hex(32)
    logger.warning(
        "JWT_SECRET ortam degiskeni ayarlanmamis! Gecici/rastgele bir anahtar uretildi - "
        "bu, her sunucu yeniden baslatildiginda (her deploy'da) TUM kullanicilarin oturumunun "
        "sonlanacagi anlamina gelir. Render'da JWT_SECRET adinda sabit, rastgele bir deger "
        "eklemeniz siddetle onerilir."
    )

AUTH_BACKEND_AVAILABLE = bool(DATABASE_URL and psycopg2 and pyjwt and bcrypt)


def get_auth_db_connection():
    if not AUTH_BACKEND_AVAILABLE:
        return None
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def init_auth_db():
    if not DATABASE_URL:
        logger.warning(
            "DATABASE_URL ayarlanmamis. Backend auth (Postgres) devre disi; "
            "sistem otomatik olarak localStorage-only auth'a duser."
        )
        return
    if not (psycopg2 and pyjwt and bcrypt):
        logger.warning(
            "psycopg2/pyjwt/bcrypt paketleri kurulu degil (requirements.txt guncellenmemis olabilir). "
            "Backend auth devre disi."
        )
        return
    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            # FAZ 2: profil ve antrenman programi. Alanlar sik degisebildigi icin
            # (frontend'deki obje sekli evrilebilir) JSONB olarak tutuyoruz -
            # her alan icin ayri kolon acmak yerine tum objeyi oldugu gibi saklayip
            # frontend'in zaten bildigi sekli birebir geri veriyoruz.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    username TEXT PRIMARY KEY REFERENCES users(username) ON DELETE CASCADE,
                    profile_data JSONB NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_programs (
                    username TEXT PRIMARY KEY REFERENCES users(username) ON DELETE CASCADE,
                    program_data JSONB NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            # FAZ 3: haftalik antrenman loglari ve beslenme verisi. Frontend zaten
            # her seyi "hafta anahtari" (Pazartesi tarihi) bazinda organize ediyor,
            # o yuzden biz de kullanici+hafta basina bir JSONB satiri tutuyoruz -
            # boylece localStorage'daki mevcut yapiyi bozmadan birebir yansitiyoruz.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_workout_weeks (
                    username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
                    week_key TEXT NOT NULL,
                    logs_data JSONB NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT now(),
                    PRIMARY KEY (username, week_key)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_nutrition_weeks (
                    username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
                    week_key TEXT NOT NULL,
                    nutrition_data JSONB NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT now(),
                    PRIMARY KEY (username, week_key)
                )
            """)
            # FAZ 4: saglik/recovery (Apple Watch webhook + manuel giris - eskiden
            # Render'in ephemeral disk'indeki SQLite'ta tutuluyordu, her deploy'da
            # siliniyordu; artik Postgres'te kalici), sakatliklar ve before/after
            # fotograflari (dikkat: fotograflar base64 olarak JSONB icinde tutuluyor -
            # pratik ve az sayida kullanici icin sorunsuz calisir, ama coklu/buyuk
            # foto hacminde ileride ayri bir object storage'a (S3/R2) tasimak daha
            # dogru olur; simdilik bu kapsamda degil).
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_health_logs (
                    username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
                    date TEXT NOT NULL,
                    sleep_hours REAL,
                    deep_sleep_hours REAL,
                    hrv_ms REAL,
                    resting_hr REAL,
                    avg_workout_hr REAL,
                    max_workout_hr REAL,
                    steps INTEGER,
                    updated_at TIMESTAMPTZ DEFAULT now(),
                    PRIMARY KEY (username, date)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_injuries (
                    username TEXT PRIMARY KEY REFERENCES users(username) ON DELETE CASCADE,
                    injuries_data JSONB NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_phases (
                    username TEXT PRIMARY KEY REFERENCES users(username) ON DELETE CASCADE,
                    phases_data JSONB NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT now()
                )
            """)
        conn.commit()
        conn.close()
        logger.info(
            "Auth DB (Postgres) hazir: 'users', 'user_profiles', 'user_programs', "
            "'user_workout_weeks', 'user_nutrition_weeks', 'user_health_logs', "
            "'user_injuries', 'user_phases' tablolari dogrulandi."
        )
    except Exception as e:
        logger.error(f"Auth DB baslatilamadi: {e}")
        traceback.print_exc()


init_auth_db()


def create_jwt_token(username: str) -> str:
    payload = {
        "sub": username,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS),
    }
    return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt_token(token: str) -> Optional[str]:
    if not pyjwt or not token:
        return None
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("sub")
    except Exception:
        return None


def hash_password_for_storage(client_hashed_password: str) -> str:
    # Client zaten sifreyi SHA-256 ile hash'leyip gonderiyor (bkz. frontend hashPassword());
    # burada bunun uzerine bir de bcrypt uyguluyoruz (savunma katmani + salt).
    return bcrypt.hashpw(client_hashed_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(client_hashed_password: str, stored_hash: str) -> bool:
    try:
        return bcrypt.checkpw(client_hashed_password.encode("utf-8"), stored_hash.encode("utf-8"))
    except Exception:
        return False


def get_current_username(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Ileriki fazlarda korumali (Depends ile) endpoint'ler icin altyapi.
    Su an hicbir route'da ZORUNLU degil - sadece 'Authorization: Bearer <token>'
    header'i varsa kullaniciyi cozer, yoksa None doner (route kendi karar verir)."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    return decode_jwt_token(token)


def require_auth_username(authorization: Optional[str] = Header(None)) -> str:
    """FAZ 2+: gercekten korumali endpoint'ler icin - token yoksa/gecersizse 401 fırlatır."""
    username = get_current_username(authorization)
    if not username:
        raise HTTPException(status_code=401, detail="unauthorized")
    return username


class AuthRequest(BaseModel):
    username: str
    password_hash: str  # client-side SHA-256 hex digest (bkz. frontend hashPassword())


class ProfileSyncInput(BaseModel):
    profile_data: dict


class ProgramSyncInput(BaseModel):
    program_data: dict


class WorkoutWeekSyncInput(BaseModel):
    logs: list


class NutritionWeekSyncInput(BaseModel):
    nutrition: dict


class InjuriesSyncInput(BaseModel):
    injuries_data: list


class PhasesSyncInput(BaseModel):
    phases_data: list


class HealthLogSyncInput(BaseModel):
    date: str
    sleep_hours: Optional[float] = 0.0
    deep_sleep_hours: Optional[float] = 0.0
    hrv_ms: Optional[float] = 0.0
    resting_hr: Optional[float] = 0.0
    avg_workout_hr: Optional[float] = None
    max_workout_hr: Optional[float] = None
    steps: Optional[int] = 0

# ================= 0. DINAMIK MODEL SECICI & CIKTI TEMIZLEYICI =================
_CACHED_MODEL: Optional[str] = None

def get_best_available_model() -> str:
    global _CACHED_MODEL
    if _CACHED_MODEL:
        return _CACHED_MODEL

    preferences = [
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "gemma2-9b-it",
        "mixtral-8x7b-32768",
        "openai/gpt-oss-20b"
    ]
    if not client:
        _CACHED_MODEL = "llama-3.1-8b-instant"
        return _CACHED_MODEL
    try:
        models_response = client.models.list()
        active_models = [m.id for m in models_response.data]
        logger.info(f"Aktif Groq Modelleri: {active_models}")

        for pref in preferences:
            if pref in active_models:
                _CACHED_MODEL = pref
                return _CACHED_MODEL

        chat_models = [m for m in active_models if not any(x in m for x in ["whisper", "tts", "guard", "embed"])]
        if chat_models:
            _CACHED_MODEL = chat_models[0]
            return _CACHED_MODEL
    except Exception as e:
        logger.warning(f"Dinamik model secilemedi, varsayilana donuluyor: {e}")

    _CACHED_MODEL = "llama-3.1-8b-instant"
    return _CACHED_MODEL

def strip_thinking_and_tables(text: str) -> str:
    if not text:
        return ""
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    summary_match = re.search(r'(?:###?\s*Özet|Özet:?)(.*)', clean, flags=re.DOTALL | re.IGNORECASE)
    if summary_match and len(summary_match.group(1).strip()) > 20:
        return summary_match.group(1).strip()
    
    lines = clean.split('\n')
    filtered_lines = [l for l in lines if not l.strip().startswith('|') and not l.strip().startswith('+-')]
    result = '\n'.join(filtered_lines).strip()
    
    return result if result else clean


def violates_coach_format_rules(text: str) -> bool:
    """AI Koc chat'inde bazen model karakterden cikip Ingilizce'ye ve
    markdown baslik/numarali liste formatina kayabiliyor (orn. '## Weekly Plan',
    '1. Bench Press - 4x8'). Bunu tespit edip bir kez retry atmak icin kullanilir."""
    if not text:
        return False

    # Markdown baslik (## Baslik)
    if re.search(r'^#{1,6}\s', text, re.MULTILINE):
        return True

    # 2+ numarali liste satiri ustuste (1. ... / 2. ... tarzi program dokumu)
    numbered_lines = len(re.findall(r'^\s*\d+\.\s+\S', text, re.MULTILINE))
    if numbered_lines >= 2:
        return True

    # Basit dil tespiti: yaygin Ingilizce kelime yogunlugu Turkce'den fazlaysa
    english_hits = len(re.findall(r'\b(the|and|for|with|your|you|is|are|of|to|daily|weekly|week)\b', text, re.IGNORECASE))
    turkish_hits = len(re.findall(r'\b(ve|için|ile|senin|sen|bir|bu|kral|gün|hafta|kas|antrenman)\b', text, re.IGNORECASE))
    if english_hits >= 5 and english_hits > turkish_hits:
        return True

    return False

# ================= 1. NUTRITION ENGINE (LLM + DETERMINISTIK) =================
# =====================================================================
# BU DOSYA, orijinal app.py icindeki
#   "# ================= 1. NUTRITION ENGINE ================="
# baslayan bolumden, parse_meal_with_llm fonksiyonunun SONUNA (yani
#   "# ================= 2. RECOVERY ENGINE ================="
# satirindan hemen ONCESINE) kadar olan tum blogun YERINE gecmelidir.
# Geri kalan kod (recovery engine, endpoint'ler, HTML, vs.) AYNEN KALIR.
#
# NOT: Bu versiyon CIG (pismemis) agirlik bazlidir. Tahil/et gibi pisince
# agirlik/yogunluk degisen besinlerde LLM, kullanicinin tarif ettigi
# porsiyonu (istersen pismis tarif etsin) CIG karsiliga cevirmesi icin
# yonlendiriliyor (asagidaki sistem promptunda donusum katsayilari var).
# =====================================================================

# ================= 1. NUTRITION ENGINE (LLM + DETERMINISTIK, CIG GRAM BAZLI) =================

CANONICAL_FOODS = {
    # -- Yumurta (cig ile pismis arasi fark az, tek deger yeterli) --
    "haslanmis yumurta": {"cal": 155, "pro": 13.0, "carb": 1.1, "fat": 11.0, "piece_g": 50},
    "sahanda yumurta": {"cal": 155, "pro": 13.0, "carb": 1.1, "fat": 11.0, "piece_g": 50},
    "omlet": {"cal": 155, "pro": 13.0, "carb": 1.1, "fat": 11.0, "piece_g": 100},
    "yumurta beyazi": {"cal": 52, "pro": 11.0, "carb": 0.7, "fat": 0.2, "piece_g": 33},
    "yumurta": {"cal": 155, "pro": 13.0, "carb": 1.1, "fat": 11.0, "piece_g": 50},

    # -- Ekmek (zaten pisirilmis urun olarak satilir, cig karsiligi yok) --
    "tam bugday ekmegi": {"cal": 247, "pro": 13.0, "carb": 41.0, "fat": 3.4, "piece_g": 30},
    "cavdar ekmegi": {"cal": 259, "pro": 8.5, "carb": 48.0, "fat": 3.3, "piece_g": 30},
    "beyaz ekmek": {"cal": 265, "pro": 9.0, "carb": 49.0, "fat": 3.2, "piece_g": 30},
    "lavas": {"cal": 280, "pro": 9.0, "carb": 55.0, "fat": 2.0, "piece_g": 80},
    "ekmek": {"cal": 265, "pro": 9.0, "carb": 49.0, "fat": 3.2, "piece_g": 30},

    # -- CIG tahil / karbonhidrat (paketten cikan hal, pisirilmemis) --
    "pirinc": {"cal": 365, "pro": 7.1, "carb": 80.0, "fat": 0.6, "piece_g": None},
    "bulgur": {"cal": 342, "pro": 12.3, "carb": 76.0, "fat": 1.3, "piece_g": None},
    "makarna": {"cal": 371, "pro": 13.0, "carb": 75.0, "fat": 1.5, "piece_g": None},
    "yulaf": {"cal": 389, "pro": 16.9, "carb": 66.0, "fat": 6.9, "piece_g": None},

    # -- CIG et / protein --
    "tavuk gogsu": {"cal": 120, "pro": 22.5, "carb": 0.0, "fat": 2.6, "piece_g": None},
    "tavuk": {"cal": 215, "pro": 18.6, "carb": 0.0, "fat": 15.0, "piece_g": None},
    "kirmizi et": {"cal": 143, "pro": 21.0, "carb": 0.0, "fat": 6.0, "piece_g": None},
    "kiyma": {"cal": 254, "pro": 17.2, "carb": 0.0, "fat": 20.0, "piece_g": None},
    "kofte": {"cal": 200, "pro": 15.0, "carb": 5.0, "fat": 14.0, "piece_g": 60},
    "somon": {"cal": 208, "pro": 20.0, "carb": 0.0, "fat": 13.0, "piece_g": None},
    "balik": {"cal": 97, "pro": 18.0, "carb": 0.0, "fat": 2.5, "piece_g": None},

    # -- Sut urunleri --
    "suzme yogurt": {"cal": 97, "pro": 9.0, "carb": 4.0, "fat": 5.0, "piece_g": None},
    "yogurt": {"cal": 65, "pro": 3.5, "carb": 4.7, "fat": 3.3, "piece_g": None},
    "kasar peyniri": {"cal": 371, "pro": 25.0, "carb": 2.0, "fat": 29.0, "piece_g": 20},
    "beyaz peynir": {"cal": 264, "pro": 17.0, "carb": 1.5, "fat": 21.0, "piece_g": 30},
    "lor peyniri": {"cal": 98, "pro": 11.0, "carb": 3.4, "fat": 4.3, "piece_g": None},
    "sut": {"cal": 61, "pro": 3.2, "carb": 4.8, "fat": 3.3, "piece_g": None},

    # -- Meyve --
    "muz": {"cal": 89, "pro": 1.1, "carb": 23.0, "fat": 0.3, "piece_g": 118},
    "elma": {"cal": 52, "pro": 0.3, "carb": 14.0, "fat": 0.2, "piece_g": 182},

    # -- Sebze / Baklagil (cig) --
    "patates": {"cal": 77, "pro": 2.0, "carb": 17.5, "fat": 0.1, "piece_g": None},
    "nohut": {"cal": 364, "pro": 19.0, "carb": 61.0, "fat": 6.0, "piece_g": None},

    # -- Yag / Kuruyemis / Ek --
    "zeytinyagi": {"cal": 884, "pro": 0.0, "carb": 0.0, "fat": 100.0, "piece_g": 14},
    "fistik ezmesi": {"cal": 588, "pro": 25.0, "carb": 20.0, "fat": 50.0, "piece_g": None},
    "badem": {"cal": 579, "pro": 21.0, "carb": 22.0, "fat": 50.0, "piece_g": None},
    "ceviz": {"cal": 654, "pro": 15.0, "carb": 14.0, "fat": 65.0, "piece_g": None},

    # -- Takviye --
    "protein tozu": {"cal": 400, "pro": 80.0, "carb": 7.0, "fat": 5.0, "piece_g": 30},
    "whey": {"cal": 400, "pro": 80.0, "carb": 7.0, "fat": 5.0, "piece_g": 30},

    # -- PAKETLİ / MARKALI ÜRÜNLER (cig/pismis kavrami yok, hazir tuketim
    # halindeki gercek besin degerleri; "piece_g" = tipik paket/kutu boyutu) --
    "redbull": {"cal": 45, "pro": 0.0, "carb": 11.0, "fat": 0.0, "piece_g": 250},
    "red bull": {"cal": 45, "pro": 0.0, "carb": 11.0, "fat": 0.0, "piece_g": 250},
    "enerji icecegi": {"cal": 45, "pro": 0.0, "carb": 11.0, "fat": 0.0, "piece_g": 250},
    "kola": {"cal": 42, "pro": 0.0, "carb": 10.6, "fat": 0.0, "piece_g": 330},
    "cola": {"cal": 42, "pro": 0.0, "carb": 10.6, "fat": 0.0, "piece_g": 330},
    "gazoz": {"cal": 41, "pro": 0.0, "carb": 10.5, "fat": 0.0, "piece_g": 330},
    "ayran": {"cal": 34, "pro": 1.7, "carb": 2.3, "fat": 1.6, "piece_g": 250},
    "cips": {"cal": 536, "pro": 6.6, "carb": 53.0, "fat": 34.0, "piece_g": None},
    "cikolata": {"cal": 545, "pro": 4.9, "carb": 61.0, "fat": 31.0, "piece_g": None},
    "protein bar": {"cal": 350, "pro": 30.0, "carb": 35.0, "fat": 10.0, "piece_g": 60},
    "biskuvi": {"cal": 480, "pro": 6.5, "carb": 68.0, "fat": 20.0, "piece_g": None},
}

# CANONICAL_FOODS icinde "cig/kuru" agirlik bazli olan (grams -> ratio/100 ile
# hesaplanan) staple gida anahtarlari. Bunlarin disindaki her sey paketli/hazir
# tuketim urunu sayilir; asagidaki eski genel varsayilan artik SADECE gercekten
# hem canonical'da hem LLM'in kendi marka bilgisinde eslesme bulunamayan,
# nadir/bilinmeyen kalemler icin devreye girer (bkz. parse_meal_with_llm).

_SORTED_FOOD_KEYS = sorted(CANONICAL_FOODS.keys(), key=lambda k: -len(k))


def match_canonical_food(norm_name: str) -> Optional[Dict[str, Any]]:
    if norm_name in CANONICAL_FOODS:
        return CANONICAL_FOODS[norm_name]
    for key in _SORTED_FOOD_KEYS:
        if re.search(rf'\b{re.escape(key)}\b', norm_name) or re.search(rf'\b{re.escape(norm_name)}\b', key):
            return CANONICAL_FOODS[key]
    return None


class ParsedFoodItem(BaseModel):
    name: str = Field(description="Besinin turkce yalin adi (orn: sahanda yumurta, pirinc, tam bugday ekmegi, tavuk gogsu, Red Bull, kola)")
    amount: float = Field(description="Kullanicinin belirttigi sayisal miktar (orn: 1, 2, 150, 0.5) - sadece bilgi amacli")
    unit: str = Field(description="Kullanicinin belirttigi birim: 'adet', 'gram', 'dilim', 'scoop', 'kasik', 'kase', 'porsiyon', 'kutu', 'sise'")
    estimated_grams: float = Field(
        description=(
            "ZORUNLU: Bu kalemin CIG (pisirilmeden ONCEKI) TOPLAM agirligi/hacmi gram olarak. "
            "Kullanici pismis bir porsiyon tarif etse bile (orn '1 kase pilav'), bunu CIG pirince "
            "cevirerek yaz (pismis pilav agirligini ~2.5-3'e bolerek cig karsiligi bul). "
            "Ekmek, yumurta, meyve gibi zaten 'son hal'de tuketilen urunlerde direkt tuketilen agirligi yaz. "
            "Paketli/markali icecek-atistirmalik ise (Red Bull, kola, cips vb.) kutu/paket agirligini (ml=g kabul et) yaz."
        )
    )
    is_packaged_or_branded: bool = Field(
        default=False,
        description=(
            "true = bu kalem hazir/paketli/markali bir urun (enerji icecegi, gazli icecek, cips, cikolata, "
            "protein bari, fast food, restoran yemegi gibi) VE 'cig agirlik' kavrami bu urune uygulanamaz. "
            "false = ev yemegi / cig ham malzeme (et, tahil, sebze, yumurta, ekmek gibi)."
        )
    )
    branded_calories: Optional[float] = Field(
        default=None,
        description="SADECE is_packaged_or_branded=true ise doldur: kullanicinin belirttigi TOPLAM miktar icin bilinen gercek kalori (kcal). Ornek: 250ml Red Bull = 113 kcal."
    )
    branded_protein: Optional[float] = Field(default=None, description="SADECE is_packaged_or_branded=true ise: TOPLAM protein (gram). Cogu icecek/atistirmalikta 0'a yakindir, uydurma.")
    branded_carbs: Optional[float] = Field(default=None, description="SADECE is_packaged_or_branded=true ise: TOPLAM karbonhidrat (gram).")
    branded_fat: Optional[float] = Field(default=None, description="SADECE is_packaged_or_branded=true ise: TOPLAM yag (gram).")


class ParsedMealResponse(BaseModel):
    items: List[ParsedFoodItem]


def normalize_turkish(text: str) -> str:
    t = text.lower()
    t = t.replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return " ".join(t.split()).strip()


MIN_ITEM_GRAMS = 5.0
MAX_ITEM_GRAMS = 1200.0


def parse_meal_with_llm(user_text: str) -> Optional[Dict[str, Any]]:
    if not client:
        return None

    system_prompt = """
Sen MyFitnessPal tarzi calisan, cok titiz bir beslenme parser'isin. Hem ev yemegi/cig malzeme
takibi hem de paketli/markali urunleri (enerji icecegi, gazli icecek, cips, cikolata, fast food vb.)
dogru ayirt etmen gerekiyor. Gorevin: kullanicinin girdigi serbest metindeki HER besini tespit edip
YALNIZCA JSON formatinda ParsedMealResponse semasina uygun cikti vermek.

IKI FARKLI KATEGORI VAR, HER KALEM ICIN DOGRU OLANI SEC:

== KATEGORI A: EV YEMEGI / CIG HAM MALZEME (is_packaged_or_branded=false) ==
Pirinc, bulgur, makarna, yulaf, tavuk, kirmizi et, yumurta, ekmek, yogurt, peynir, meyve, sebze gibi
seyler. Bu kalemlerde estimated_grams (CIG agirlik) alanini doldurman yeterli, branded_* alanlarini
BOS BIRAK (null). Donusum katsayilari:
- Pirinc: pismis agirlik / 2.8  (orn 1 kase ~200g pismis pilav -> ~70g cig pirinc)
- Bulgur: pismis agirlik / 2.5
- Makarna: pismis agirlik / 2.2
- Yulaf (sutle/suyla lapasi): pismis agirlik / 2.0
Standart porsiyonlar: 1 adet yumurta=50g, 1 dilim ekmek=30g, 1 olcek protein tozu=30g,
1 yemek kasigi zeytinyagi/fistik ezmesi=14g, 1 orta boy cig tavuk gogsu=165g, 1 kofte(cig)=60g.

== KATEGORI B: PAKETLI / MARKALI URUN (is_packaged_or_branded=true) ==
Enerji icecegi (Red Bull vb.), kola/gazoz, cips, cikolata, bisküvi, protein bari, fast food,
restoran yemegi, hazir atistirmalik gibi seyler. Bu urunlerde "cig agirlik" kavrami YOKTUR ve
UYDURMA GENEL DEGER KULLANMA. Bunun yerine SENIN GERCEK DUNYA BILGINE dayanarak, urunun bilinen
gercek besin degerlerini branded_calories/branded_protein/branded_carbs/branded_fat alanlarina
TOPLAM miktar icin yaz. Ornekler (referans al):
- 250ml Red Bull (1 kutu) ≈ 113 kcal, 0g protein, 27-28g karbonhidrat, 0g yag
- 330ml kola (1 kutu) ≈ 139 kcal, 0g protein, 35g karbonhidrat, 0g yag
- 1 orta boy cikolata (50g) ≈ 270 kcal, 2.5g protein, 30g karbonhidrat, 15g yag
- 1 paket cips (30g) ≈ 160 kcal, 2g protein, 16g karbonhidrat, 10g yag
ONEMLI: enerji icecekleri ve gazli icecekler pratikte 0g PROTEIN icerir, protein UYDURMA.

KURALLAR:
1. Gercekci, olculu degerler ver. Asiri buyuk veya sifir deger UYDURMA.
2. Ayni mesajdaki her ayri besin icin ayri bir item olustur.
3. Cikti formati kesinlikle su JSON seklinde olmalidir (Kategori A ve B ornekleri):
{
  "items": [
    {"name": "sahanda yumurta", "amount": 1, "unit": "adet", "estimated_grams": 50, "is_packaged_or_branded": false},
    {"name": "Red Bull", "amount": 1, "unit": "kutu", "estimated_grams": 250, "is_packaged_or_branded": true, "branded_calories": 113, "branded_protein": 0, "branded_carbs": 28, "branded_fat": 0}
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
            matched = match_canonical_food(norm_name)

            has_branded_data = item.is_packaged_or_branded and any(
                v is not None for v in [item.branded_calories, item.branded_protein, item.branded_carbs, item.branded_fat]
            )

            if matched:
                # Yerel veritabanindaki deterministik deger her zaman en guvenilir kaynak
                grams = item.estimated_grams if item.estimated_grams and item.estimated_grams > 0 else 100.0
                grams = max(MIN_ITEM_GRAMS, min(MAX_ITEM_GRAMS, grams))
                ratio = grams / 100.0
                cal = matched["cal"] * ratio
                pro = matched["pro"] * ratio
                carb = matched["carb"] * ratio
                fat = matched["fat"] * ratio
                total_cal += cal
                total_pro += pro
                total_carb += carb
                total_fat += fat
                summary_items.append(f"{item.amount:g} {item.unit} {item.name.title()} (~{round(grams)}g)")

            elif has_branded_data:
                # Markali/paketli urun: LLM'in kendi urun bilgisine guven, "cig agirlik"
                # varsayimini uygulama, genel ortalama kullanma
                cal = max(0.0, item.branded_calories or 0.0)
                pro = max(0.0, item.branded_protein or 0.0)
                carb = max(0.0, item.branded_carbs or 0.0)
                fat = max(0.0, item.branded_fat or 0.0)
                total_cal += cal
                total_pro += pro
                total_carb += carb
                total_fat += fat
                grams_disp = item.estimated_grams if item.estimated_grams and item.estimated_grams > 0 else None
                grams_txt = f" (~{round(grams_disp)}ml/g)" if grams_disp else ""
                summary_items.append(f"{item.amount:g} {item.unit} {item.name.title()}{grams_txt}")

            else:
                # Ne canonical eslesme ne LLM marka verisi var: son care genel varsayim
                grams = item.estimated_grams if item.estimated_grams and item.estimated_grams > 0 else 100.0
                grams = max(MIN_ITEM_GRAMS, min(MAX_ITEM_GRAMS, grams))
                ratio = grams / 100.0
                total_cal += 180.0 * ratio
                total_pro += 8.0 * ratio
                total_carb += 20.0 * ratio
                total_fat += 6.0 * ratio
                summary_items.append(f"{item.amount:g} {item.unit} {item.name.title()} (~{round(grams)}g, tahmini)")

        if total_cal > 0:
            return {
                "food_name": " + ".join(s.split(" (~")[0] for s in summary_items),
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

# ================= PWA IKON (base64 gomulu PNG, 512x512) =================
# Ayri bir /static klasoru acmamak icin ikonu dogrudan koda gomduk; /icon.png
# route'u bunu decode edip PNG olarak sunuyor. iPhone'da "Ana Ekrana Ekle"
# yapinca bu ikon kullanilir.
APP_ICON_PNG_BASE64 = "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"

HTML_INTERFACE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover, maximum-scale=1.0, user-scalable=no">
    <title>Looksmax HUB - Elite Performance & Coaching</title>

    <!-- PWA / iPhone "Ana Ekrana Ekle" destegi -->
    <link rel="manifest" href="/manifest.json">
    <link rel="icon" type="image/png" href="/icon.png">
    <link rel="apple-touch-icon" href="/icon.png">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="Looksmax Hub">
    <meta name="theme-color" content="#0b0d12">

    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/html5-qrcode@2.3.8/html5-qrcode.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background-color: #0b0d12; color: #e5e7eb; display: flex; flex-direction: column; height: 100vh; overflow: hidden; padding-top: env(safe-area-inset-top); padding-bottom: env(safe-area-inset-bottom); }

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
                    <li><i>Sağlık Örneklerini Bul</i> &rarr; <b>Uyku Analizi</b> (Süre/Saat)</li>
                    <li><i>Sağlık Örneklerini Bul</i> &rarr; <b>Kalp Atış Hızı Değişkenliği (HRV)</b></li>
                    <li><i>Sağlık Örneklerini Bul</i> &rarr; <b>Dinlenme Sırasındaki Kalp Atış Hızı</b></li>
                </ul>
            </div>

            <div class="modal-step">
                <b>Adım 3:</b> <b>URL İçeriğini Al</b> eylemini ekleyip <b>POST</b> yöntemiyle şu adresi girin:
                <div class="url-box" id="webhookUrlBox">
                    <span id="webhookUrlText">https://.../api/health-sync</span>
                    <button onclick="copyWebhookUrl()" style="background:#00f2fe; color:#000; border:none; padding:4px 8px; border-radius:4px; font-weight:800; font-size:0.7rem; cursor:pointer;">Kopyala</button>
                </div>
            </div>

            <div class="modal-step" style="border: 1px solid rgba(245, 158, 11, 0.4);">
                <b style="color:#f59e0b;">⚠️ Önemli:</b> İstek gövdesine (Request Body, JSON) şu alanları eklemelisin, yoksa veri panelde görünmez:
                <ul style="margin-left: 18px; margin-top: 4px; color:#9ca3af;">
                    <li><b>username</b>: Uygulamadaki kullanıcı adın (<code id="webhookUsernameHint">-</code>) — birebir aynı yazılmalı</li>
                    <li><b>date</b>: <code>GG.AA.YYYY</code> formatında (örn: <code>02.09.2026</code>) — başka formatta gelirse kayıt alınır ama panelde o gün eşleşmez</li>
                </ul>
            </div>

            <div class="modal-step">
                <b>💡 Otomatikleştirme:</b> Kestirmeler &rarr; Otomasyon sekmesinden <i>"Sabah Alarmı Durdurulduğunda"</i> bu kestirmeyi seçerseniz verileriniz her sabah otomatik panele düşer.
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

    <div class="modal-overlay" id="barcodeModal" onclick="closeBarcodeModalOnBg(event)">
        <div class="modal-box" onclick="event.stopPropagation()" style="max-width:420px;">
            <div class="modal-header">
                <h3>🏷️ Barkod Tara</h3>
                <button class="modal-close-btn" onclick="closeBarcodeScanner()">✕</button>
            </div>

            <div id="barcodeScanState">
                <div id="barcodeReaderRegion" style="width:100%; border-radius:12px; overflow:hidden; background:#000; min-height:220px;"></div>
                <div style="font-size:0.75rem; color:#9ca3af; text-align:center; margin-top:10px;">
                    Ürünün barkodunu kameraya göster — otomatik algılanacak.
                </div>
            </div>

            <div id="barcodeLoadingState" style="display:none; text-align:center; padding:24px;">
                <div style="font-size:2rem; margin-bottom:8px;">🔎</div>
                <div style="font-weight:700; color:#00f2fe;">Ürün aranıyor...</div>
            </div>

            <div id="barcodeNotFoundState" style="display:none; text-align:center; padding:20px; color:#9ca3af; font-size:0.85rem;">
                Bu barkod veritabanında bulunamadı kral. Yazarak elle ekleyebilirsin.
                <button class="btn-log" onclick="resetBarcodeScanner()" style="margin-top:12px;">Tekrar Tara</button>
            </div>

            <div id="barcodeResultState" style="display:none; flex-direction:column; gap:10px;">
                <div style="background:#0a0c10; border:1px solid #1c2230; border-radius:10px; padding:12px;">
                    <div style="font-weight:800; color:#fff; font-size:0.95rem;" id="barcodeProductName">-</div>
                    <div style="font-size:0.72rem; color:#9ca3af; margin-top:2px;">100g başına: <span id="barcodeProductPer100">-</span></div>
                </div>
                <input type="number" id="barcodeAmountGrams" placeholder="Miktar (gram)" value="100" min="1" oninput="updateBarcodePreview()" />
                <div class="macro-stat-grid">
                    <div class="macro-card">
                        <div class="macro-label">Kalori</div>
                        <div class="macro-val macro-c-cal" id="barcodePreviewCal">0 kcal</div>
                    </div>
                    <div class="macro-card">
                        <div class="macro-label">Protein</div>
                        <div class="macro-val macro-c-pro" id="barcodePreviewPro">0g</div>
                    </div>
                    <div class="macro-card">
                        <div class="macro-label">Karb</div>
                        <div class="macro-val macro-c-carb" id="barcodePreviewCarb">0g</div>
                    </div>
                    <div class="macro-card">
                        <div class="macro-label">Yağ</div>
                        <div class="macro-val macro-c-fat" id="barcodePreviewFat">0g</div>
                    </div>
                </div>
                <button class="btn-log" onclick="addBarcodeProductToMeal()">Öğüne Ekle</button>
                <button onclick="resetBarcodeScanner()" style="background:none; border:none; color:#9ca3af; font-size:0.75rem; cursor:pointer;">Başka bir ürün tara</button>
            </div>
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

                <div class="hub-card" onclick="openView('program')">
                    <div>
                        <div class="card-icon">🗓️</div>
                        <div class="card-heading">Antrenman Programı</div>
                        <div class="card-desc">Hedefine ve toparlanma verine göre AI'nin çıkardığı, gün gün kişisel program.</div>
                    </div>
                    <div class="card-action">Programı Aç →</div>
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

                        <div style="background:#0a0c10; border:1px solid #1c2230; border-radius:9px; padding:10px 12px; display:flex; flex-direction:column; gap:8px;">
                            <label style="display:flex; align-items:center; gap:8px; font-size:0.8rem; color:#e5e7eb; cursor:pointer; font-weight:600;">
                                <input type="checkbox" id="addToProgramCheck" onchange="toggleProgramDaySelect()" style="width:16px; height:16px; accent-color:#00f2fe; cursor:pointer;" />
                                📋 Antrenman Programımla Eşleştir
                            </label>
                            <select id="addToProgramDaySelect" onchange="refreshProgramExerciseSelectOptions()" style="display:none; background:#131722; border:1px solid #2b354d; color:#00f2fe; padding:8px 10px; border-radius:7px; font-weight:700; font-size:0.8rem; outline:none;"></select>
                            <select id="addToProgramExerciseSelect" onchange="applyProgramExerciseToForm()" style="display:none; background:#131722; border:1px solid #2b354d; color:#00f2fe; padding:8px 10px; border-radius:7px; font-weight:700; font-size:0.8rem; outline:none;"></select>
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
                        <button class="file-btn" title="Barkod Tara" onclick="openBarcodeScanner()">🏷️</button>
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
                    <div style="color:#00f2fe; font-size:1.1rem; font-weight:800;">&rarr;</div>
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

        <div class="view-panel" id="programView">
            <div class="overload-col-left">
                <div class="panel-card">
                    <div class="panel-header">
                        <span>🧠 AI Program Oluşturucu</span>
                        <span class="badge-cyan" id="programStatusBadge">Program Yok</span>
                    </div>
                    <div class="input-form">
                        <select id="programSplitSelect">
                            <option value="3">3 Günlük Split (Full Body / Push-Pull-Legs)</option>
                            <option value="4">4 Günlük Split (Upper/Lower)</option>
                            <option value="5" selected>5 Günlük Split (Bro Split)</option>
                            <option value="6">6 Günlük Split (PPL x2)</option>
                        </select>
                        <button class="btn-log" id="generateProgramBtn" onclick="generateAiProgram()">🧠 Bana Özel Program Oluştur</button>
                        <div style="font-size:0.72rem; color:#6b7280; line-height:1.4;">
                            Profilindeki hedef, TDEE'n, aktif sakatlıkların ve bugünkü toparlanma skorun otomatik olarak dikkate alınır.
                        </div>
                    </div>
                </div>

                <div class="panel-card" style="flex:1;">
                    <div class="panel-header">
                        <span>➕ Manuel Hareket Ekle</span>
                    </div>
                    <div class="input-form">
                        <input type="text" id="progManualExercise" placeholder="Hareket Adı" list="defaultExercises" />
                        <div class="form-grid-3x1">
                            <input type="number" id="progManualSets" placeholder="Set" min="1" value="3" />
                            <input type="text" id="progManualReps" placeholder="Tekrar (örn: 8-10)" value="8-10" />
                            <input type="text" id="progManualNote" placeholder="Not (opsiyonel)" />
                        </div>
                        <button class="btn-log" onclick="addManualExerciseToProgram()">Seçili Güne Ekle</button>
                    </div>
                </div>
            </div>

            <div class="overload-col-right">
                <div class="panel-card" style="height: 100%;">
                    <div class="panel-header">
                        <span>🗓️ Kişisel Programın</span>
                        <button onclick="clearProgramDay()" style="background:none; border:none; color:#ef4444; font-size:0.75rem; cursor:pointer; font-weight:700;">Bu Günü Temizle</button>
                    </div>
                    <div class="days-tab-bar" id="programDaysTabBar"></div>
                    <div class="history-list" id="programExerciseList" style="max-height:none; flex:1;"></div>
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
    safeLocalStorageSet("user_weeks_" + username, JSON.stringify(allWeeks));
    if (currentUser && currentUser.token && currentUser.username === username) {
        fetch(`/api/workout-weeks/${encodeURIComponent(currentWeekKey)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + currentUser.token },
            body: JSON.stringify({ logs: logs })
        }).catch(err => console.warn("Antrenman haftası backend'e senkronize edilemedi (local'de kayıtlı kaldı):", err));
    }
}

        function getUserWeeklyNutrition(username) {
            const allWeeks = JSON.parse(localStorage.getItem("user_nutri_weeks_" + username) || "{}");
            return allWeeks[currentWeekKey] || {};
        }
        function saveUserWeeklyNutrition(username, nutriData) {
    const allWeeks = JSON.parse(localStorage.getItem("user_nutri_weeks_" + username) || "{}");
    allWeeks[currentWeekKey] = nutriData;
    safeLocalStorageSet("user_nutri_weeks_" + username, JSON.stringify(allWeeks));
    if (currentUser && currentUser.token && currentUser.username === username) {
        fetch(`/api/nutrition-weeks/${encodeURIComponent(currentWeekKey)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + currentUser.token },
            body: JSON.stringify({ nutrition: nutriData })
        }).catch(err => console.warn("Beslenme haftası backend'e senkronize edilemedi (local'de kayıtlı kaldı):", err));
    }
}
        function getUserProfileData(username) { return JSON.parse(localStorage.getItem("user_profile_" + username) || "{}"); }
        function saveUserProfileData(username, profData) {
    const success = safeLocalStorageSet("user_profile_" + username, JSON.stringify(profData));
    if (success && currentUser && currentUser.token && currentUser.username === username) {
        fetch('/api/profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + currentUser.token },
            body: JSON.stringify({ profile_data: profData })
        }).catch(err => console.warn("Profil backend'e senkronize edilemedi (local'de kayıtlı kaldı):", err));
    }
    return success;
}

        function getUserPhases(username) { return JSON.parse(localStorage.getItem("user_phases_" + username) || "[]"); }
        function saveUserPhases(username, phases) {
    const success = safeLocalStorageSet("user_phases_" + username, JSON.stringify(phases));
    if (success && currentUser && currentUser.token && currentUser.username === username) {
        fetch('/api/phases', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + currentUser.token },
            body: JSON.stringify({ phases_data: phases })
        }).catch(err => console.warn("Fazlar/fotoğraflar backend'e senkronize edilemedi (local'de kayıtlı kaldı):", err));
    }
    return success;
}

        function getUserHealthLogs(username) { return JSON.parse(localStorage.getItem("user_health_" + username) || "{}"); }
        function saveUserHealthLogs(username, healthLogs) {
    safeLocalStorageSet("user_health_" + username, JSON.stringify(healthLogs));
}

        function getUserInjuries(username) { return JSON.parse(localStorage.getItem("user_injuries_" + username) || "[]"); }
        function saveUserInjuries(username, injuries) {
    safeLocalStorageSet("user_injuries_" + username, JSON.stringify(injuries));
    if (currentUser && currentUser.token && currentUser.username === username) {
        fetch('/api/injuries', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + currentUser.token },
            body: JSON.stringify({ injuries_data: injuries })
        }).catch(err => console.warn("Sakatlıklar backend'e senkronize edilemedi (local'de kayıtlı kaldı):", err));
    }
}

        function getUserProgram(username) { return JSON.parse(localStorage.getItem("user_program_" + username) || "null"); }
        function saveUserProgram(username, program) {
    const success = safeLocalStorageSet("user_program_" + username, JSON.stringify(program));
    if (success && currentUser && currentUser.token && currentUser.username === username) {
        fetch('/api/program', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + currentUser.token },
            body: JSON.stringify({ program_data: program })
        }).catch(err => console.warn("Program backend'e senkronize edilemedi (local'de kayıtlı kaldı):", err));
    }
    return success;
}

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
        let userProgram = null;
        let selectedProgramDayIdx = 0;
        let barcodeScannerInstance = null;
        let currentBarcodeProduct = null;

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
                const hintEl = document.getElementById("webhookUsernameHint");
                if (hintEl) hintEl.innerText = currentUser ? currentUser.username : "-";
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

            if (viewName === 'overload') { setTimeout(updateChart, 150); refreshProgramDaySelectOptions(); }
            if (viewName === 'nutrition') { renderNutriDayTabs(); renderSelectedDayNutrition(); }
            if (viewName === 'profile') { loadUserProfileUI(); loadUserPhasesUI(); }
            if (viewName === 'health') { loadHealthUI(); renderInjuriesUI(); }
            if (viewName === 'program') { loadProgramUI(); }
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
                loadProgramUI();
                syncHealthDataFromServer();
                syncProfileFromServer();
                syncProgramFromServer();
                syncWorkoutWeeksFromServer();
                syncNutritionWeeksFromServer();
                syncInjuriesFromServer();
                syncPhasesFromServer();
            }
        }

        async function syncProfileFromServer() {
            if (!currentUser || !currentUser.token) return;
            try {
                const res = await fetch('/api/profile', {
                    headers: { 'Authorization': 'Bearer ' + currentUser.token }
                });
                if (!res.ok) return; // 503 (yapilandirilmadi) / 401 vs. - sessizce local'de kal
                const data = await res.json();
                if (data.profile_data) {
                    userProfile = data.profile_data;
                    // saveUserProfileData'yi DEGIL, dogrudan safeLocalStorageSet'i kullaniyoruz -
                    // yoksa backend'den okuyup tekrar backend'e yazma dongusune girer
                    safeLocalStorageSet("user_profile_" + currentUser.username, JSON.stringify(userProfile));
                    const profileViewEl = document.getElementById("profileView");
                    if (profileViewEl && profileViewEl.classList.contains("active")) {
                        loadUserProfileUI();
                    }
                }
            } catch (err) {
                console.warn("Profil sunucudan senkronize edilemedi:", err);
            }
        }

        async function syncProgramFromServer() {
            if (!currentUser || !currentUser.token) return;
            try {
                const res = await fetch('/api/program', {
                    headers: { 'Authorization': 'Bearer ' + currentUser.token }
                });
                if (!res.ok) return;
                const data = await res.json();
                if (data.program_data && Array.isArray(data.program_data.days)) {
                    userProgram = data.program_data;
                    safeLocalStorageSet("user_program_" + currentUser.username, JSON.stringify(userProgram));
                    const programViewEl = document.getElementById("programView");
                    if (programViewEl && programViewEl.classList.contains("active")) {
                        loadProgramUI();
                    }
                    refreshProgramDaySelectOptions();
                }
            } catch (err) {
                console.warn("Program sunucudan senkronize edilemedi:", err);
            }
        }

        async function syncWorkoutWeeksFromServer() {
            if (!currentUser || !currentUser.token) return;
            try {
                const res = await fetch('/api/workout-weeks', {
                    headers: { 'Authorization': 'Bearer ' + currentUser.token }
                });
                if (!res.ok) return;
                const data = await res.json();
                if (data.weeks && typeof data.weeks === 'object') {
                    // Backend'de olan haftalar local'in uzerine yazilir (backend esas alinir),
                    // sadece backend'de HIC olmayan (henuz senkronize edilmemis) local haftalar korunur.
                    const localAllWeeks = getAllUserWeeks(currentUser.username);
                    const merged = { ...localAllWeeks, ...data.weeks };
                    safeLocalStorageSet("user_weeks_" + currentUser.username, JSON.stringify(merged));
                    weeklyLogs = merged[currentWeekKey] || [];
                    const overloadViewEl = document.getElementById("overloadView");
                    if (overloadViewEl && overloadViewEl.classList.contains("active")) {
                        loadUserWorkouts();
                    }
                }
            } catch (err) {
                console.warn("Antrenman geçmişi sunucudan senkronize edilemedi:", err);
            }
        }

        async function syncNutritionWeeksFromServer() {
            if (!currentUser || !currentUser.token) return;
            try {
                const res = await fetch('/api/nutrition-weeks', {
                    headers: { 'Authorization': 'Bearer ' + currentUser.token }
                });
                if (!res.ok) return;
                const data = await res.json();
                if (data.weeks && typeof data.weeks === 'object') {
                    const localAllWeeks = JSON.parse(localStorage.getItem("user_nutri_weeks_" + currentUser.username) || "{}");
                    const merged = { ...localAllWeeks, ...data.weeks };
                    safeLocalStorageSet("user_nutri_weeks_" + currentUser.username, JSON.stringify(merged));
                    weeklyNutrition = merged[currentWeekKey] || {};
                    const nutritionViewEl = document.getElementById("nutritionView");
                    if (nutritionViewEl && nutritionViewEl.classList.contains("active")) {
                        renderNutriDayTabs();
                        renderSelectedDayNutrition();
                    }
                }
            } catch (err) {
                console.warn("Beslenme geçmişi sunucudan senkronize edilemedi:", err);
            }
        }

        async function syncInjuriesFromServer() {
            if (!currentUser || !currentUser.token) return;
            try {
                const res = await fetch('/api/injuries', {
                    headers: { 'Authorization': 'Bearer ' + currentUser.token }
                });
                if (!res.ok) return;
                const data = await res.json();
                if (data.injuries_data) {
                    safeLocalStorageSet("user_injuries_" + currentUser.username, JSON.stringify(data.injuries_data));
                    renderInjuriesUI();
                }
            } catch (err) {
                console.warn("Sakatlıklar sunucudan senkronize edilemedi:", err);
            }
        }

        async function syncPhasesFromServer() {
            if (!currentUser || !currentUser.token) return;
            try {
                const res = await fetch('/api/phases', {
                    headers: { 'Authorization': 'Bearer ' + currentUser.token }
                });
                if (!res.ok) return;
                const data = await res.json();
                if (data.phases_data) {
                    safeLocalStorageSet("user_phases_" + currentUser.username, JSON.stringify(data.phases_data));
                    const profileViewEl = document.getElementById("profileView");
                    if (profileViewEl && profileViewEl.classList.contains("active")) {
                        loadUserPhasesUI();
                    }
                }
            } catch (err) {
                console.warn("Fazlar/fotoğraflar sunucudan senkronize edilemedi:", err);
            }
        }

        async function syncHealthDataFromServer() {
            if (!currentUser) return;
            try {
                const res = await fetch(`/api/health-sync/${encodeURIComponent(currentUser.username)}`);
                const data = await res.json();
                if (data.status !== "success" || !Array.isArray(data.logs) || data.logs.length === 0) return;

                const localLogs = getUserHealthLogs(currentUser.username);
                let changed = false;

                data.logs.forEach(row => {
                    if (!row.date) return;
                    const existing = localLogs[row.date];
                    const isNewer = !existing || !existing.synced_at || (row.updated_at && row.updated_at > existing.synced_at);
                    if (isNewer) {
                        localLogs[row.date] = {
                            date: row.date,
                            sleep_hours: row.sleep_hours,
                            deep_sleep_hours: row.deep_sleep_hours,
                            hrv_ms: row.hrv_ms,
                            resting_hr: row.resting_hr,
                            avg_workout_hr: row.avg_workout_hr,
                            max_workout_hr: row.max_workout_hr,
                            steps: row.steps,
                            synced_at: row.updated_at,
                            source: "apple_watch"
                        };
                        changed = true;
                    }
                });

                if (changed) {
                    userHealthLogs = localLogs;
                    saveUserHealthLogs(currentUser.username, userHealthLogs);
                    const healthViewEl = document.getElementById("healthView");
                    if (healthViewEl && healthViewEl.classList.contains("active")) {
                        loadHealthUI();
                    }
                }
            } catch (err) {
                console.error("Sağlık verisi senkronizasyon hatası:", err);
            }
        }

        function toggleAuthMode() {
            isRegisterMode = !isRegisterMode;
            document.getElementById("authTitle").innerText = isRegisterMode ? "⚡ KAYIT OL" : "⚡ GİRİŞ YAP";
            document.getElementById("authSubmitBtn").innerText = isRegisterMode ? "Hesap Oluştur" : "Giriş Yap";
            document.getElementById("authToggle").innerHTML = isRegisterMode ? "Zaten hesabın var mı? <b>Giriş Yap</b>" : "Hesabın yok mu? <b>Kayıt Ol</b>";
        }

        async function hashPassword(password) {
            if (!window.crypto || !window.crypto.subtle) {
                throw new Error("Bu bağlantı (HTTP) şifre güvenliği için desteklenmiyor, HTTPS üzerinden açman gerekiyor.");
            }
            const enc = new TextEncoder().encode(password);
            const hashBuffer = await crypto.subtle.digest('SHA-256', enc);
            return Array.from(new Uint8Array(hashBuffer)).map(b => b.toString(16).padStart(2, '0')).join('');
        }

        function looksLikeSha256Hash(str) {
            return typeof str === "string" && /^[a-f0-9]{64}$/.test(str);
        }

        function setAuthSession(username, token) {
            currentUser = { username: username, token: token || null };
            localStorage.setItem("active_user", JSON.stringify(currentUser));
        }

        async function tryMigrateLegacyUserToBackend(username, plainPassword, hashedPassword) {
            // localStorage'daki eski hesabin sifresi dogruysa, backend'de sessizce (kullaniciya
            // hissettirmeden) ayni hesabi acar. Boylece Neon/Supabase kurulunca eski kullanicilar
            // "hesabim kayboldu" demeden gecis yapmis olur.
            const allUsers = getStorageUsers();
            const stored = allUsers[username];
            if (!stored) return false;

            const isLegacyPlainText = !looksLikeSha256Hash(stored);
            const matches = isLegacyPlainText ? (stored === plainPassword) : (stored === hashedPassword);
            if (!matches) return false;

            try {
                const res = await fetch('/api/auth/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: username, password_hash: hashedPassword })
                });
                const data = await res.json();
                if (res.ok && data.token) {
                    setAuthSession(username, data.token);
                    checkAuth();
                    return true;
                }
            } catch (err) {
                console.error("Legacy kullanıcı migrasyon hatası:", err);
            }
            return false;
        }

        async function handleAuthSubmit() {
            const u = document.getElementById("authUsername").value.trim();
            const p = document.getElementById("authPassword").value.trim();
            if (!u || !p) return alert("Kullanıcı adı ve şifre gir!");

            let hashedP;
            try {
                hashedP = await hashPassword(p);
            } catch (err) {
                return alert(err.message);
            }

            // ---- 1) Once gercek backend'i (Postgres+JWT) dene ----
            try {
                const endpoint = isRegisterMode ? '/api/auth/register' : '/api/auth/login';
                const res = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: u, password_hash: hashedP })
                });

                if (res.status !== 503) {
                    // Backend yapilandirilmis (DATABASE_URL ayarli) - yaniti otoriter kabul et
                    const data = await res.json();

                    if (res.ok && data.token) {
                        setAuthSession(u, data.token);
                        checkAuth();
                        return;
                    }
                    if (data.detail === "username_taken") {
                        return alert("Bu kullanıcı adı zaten var! Giriş yapmayı dene.");
                    }
                    if (data.detail === "invalid_credentials") {
                        return alert("Kullanıcı adı veya şifre hatalı!");
                    }
                    if (data.detail === "user_not_found") {
                        const migrated = await tryMigrateLegacyUserToBackend(u, p, hashedP);
                        if (migrated) return;
                        return alert("Böyle bir kullanıcı bulunamadı. Önce 'Kayıt Ol' ile hesap açmalısın.");
                    }
                    // beklenmedik bir backend hatasi - asagidaki localStorage fallback'ine devam
                }
                // res.status === 503 (backend henuz yapilandirilmadi) ise fallback'e devam
            } catch (networkErr) {
                console.warn("Backend auth'a ulaşılamadı, localStorage moduna geçiliyor:", networkErr);
            }

            // ---- 2) LOCALSTORAGE FALLBACK (backend henuz kurulmadiysa/erisilemiyorsa) ----
            const allUsers = getStorageUsers();

            if (isRegisterMode) {
                if (allUsers[u]) return alert("Bu kullanıcı adı zaten var! Giriş yapmayı dene.");
                allUsers[u] = hashedP;
                if (!safeLocalStorageSet("app_registered_users", JSON.stringify(allUsers))) return;
                setAuthSession(u, null);
                checkAuth();
            } else {
                if (!allUsers[u]) {
                    return alert("Böyle bir kullanıcı bulunamadı. Önce 'Kayıt Ol' ile hesap açmalısın.");
                }
                const stored = allUsers[u];
                const isLegacyPlainText = !looksLikeSha256Hash(stored);
                const matches = isLegacyPlainText ? (stored === p) : (stored === hashedP);
                if (!matches) {
                    return alert("Kullanıcı adı veya şifre hatalı!");
                }
                if (isLegacyPlainText) {
                    // Eski, duz metin olarak kayitli sifreyi sessizce hash'e yukselt
                    allUsers[u] = hashedP;
                    safeLocalStorageSet("app_registered_users", JSON.stringify(allUsers));
                }
                setAuthSession(u, null);
                checkAuth();
            }
        }

        function logout() {
            localStorage.removeItem("active_user");
            currentUser = null;
            location.reload();
        }
        function safeLocalStorageSet(key, value) {
    try {
        localStorage.setItem(key, value);
        return true;
    } catch (e) {
        if (e.name === "QuotaExceededError" || e.code === 22 || e.code === 1014) {
            alert(
                "⚠️ Depolama alanı doldu kral!\n\n" +
                "Tarayıcın için ayrılan alan (fotoğraflar + veriler) limitine ulaştı. " +
                "Kayıt gerçekleşmedi.\n\n" +
                "Çözüm: Profil sekmesinden eski/gereksiz dönem fotoğraflarını sil, " +
                "sonra tekrar dene."
            );
        } else {
            alert("Kayıt sırasında beklenmeyen bir hata oluştu: " + e.message);
        }
        return false;
    }
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
            if (currentUser.token) {
                fetch('/api/health-log', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + currentUser.token },
                    body: JSON.stringify(log)
                }).catch(err => console.warn("Sağlık verisi backend'e senkronize edilemedi (local'de kayıtlı kaldı):", err));
            }
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
 
    const compressedBase64 = await compressImage(file, 640, 0.55);
    const phase = userPhases.find(p => p.id === activePhaseId);
    if (phase) {
        if (!phase.photos) phase.photos = {};
        const previousPhoto = phase.photos[pendingUploadSlot];
        phase.photos[pendingUploadSlot] = compressedBase64;
 
        const success = saveUserPhases(currentUser.username, userPhases);
        if (!success) {
            // Kayit basarisiz oldu, degisikligi geri al
            phase.photos[pendingUploadSlot] = previousPhoto;
        } else {
            renderActivePhasePhotos();
        }
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
            refreshProgramDaySelectOptions();
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

            const addToProgramChecked = document.getElementById("addToProgramCheck").checked;
            if (addToProgramChecked) {
                const dayIdx = parseInt(document.getElementById("addToProgramDaySelect").value, 10);
                if (!isNaN(dayIdx)) {
                    const prog = userProgram && userProgram.days ? userProgram : getUserProgram(currentUser.username);
                    const dayExercises = (prog && prog.days && prog.days[dayIdx]) ? prog.days[dayIdx].exercises : [];
                    const alreadyInProgram = dayExercises.some(e => e.name.trim().toLowerCase() === name.trim().toLowerCase());
                    // Sadece programda henuz olmayan YENI bir hareket girildiyse programa ekle.
                    // Zaten programdan secilmis bir hareketse dokunma - yoksa hedef set/tekrar
                    // (orn "4x8-10"), o an girilen tek setin degerleriyle (orn "1x8") ustune yazilip bozulur.
                    if (!alreadyInProgram) {
                        addExerciseToProgramDay(dayIdx, name, setNum, `${reps}`, "");
                    }
                }
            }

            document.getElementById("exerciseSet").value = setNum + 1;
            document.getElementById("exerciseWeight").value = "";
            // Program'dan eslesen bir hareket girildiyse tekrar sayisi genelde sabittir (orn hep 8),
            // bir sonraki set icin tekrar yazmaya gerek kalmasin diye temizlenmiyor.
            if (!addToProgramChecked) {
                document.getElementById("exerciseReps").value = "";
            }
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

        // ================= ANTRENMAN PROGRAMI =================
        function defaultEmptyProgram(numDays) {
            const days = [];
            for (let i = 0; i < numDays; i++) {
                days.push({ day_name: `Gün ${i + 1}`, focus: "", exercises: [] });
            }
            return { days: days, generated_at: null, goal_used: "" };
        }

        function loadProgramUI() {
            if (!currentUser) return;
            userProgram = getUserProgram(currentUser.username);
            if (!userProgram || !Array.isArray(userProgram.days) || userProgram.days.length === 0) {
                userProgram = defaultEmptyProgram(5);
            }
            if (selectedProgramDayIdx >= userProgram.days.length) selectedProgramDayIdx = 0;

            const badge = document.getElementById("programStatusBadge");
            if (badge) {
                badge.innerText = userProgram.generated_at ? `${userProgram.days.length} Günlük Program` : "Henüz Oluşturulmadı";
            }

            renderProgramDayTabs();
            renderProgramExercises();
            refreshProgramDaySelectOptions();
        }

        function renderProgramDayTabs() {
            const bar = document.getElementById("programDaysTabBar");
            if (!bar || !userProgram) return;
            bar.innerHTML = "";

            userProgram.days.forEach((d, idx) => {
                const isActive = (idx === selectedProgramDayIdx) ? "active" : "";
                bar.innerHTML += `
                    <button class="day-tab-btn ${isActive}" onclick="selectProgramDayTab(${idx})">
                        Gün ${idx + 1}
                        <span class="tab-sub">${(d.focus || "").substring(0, 10) || "—"}</span>
                    </button>
                `;
            });
        }

        function selectProgramDayTab(idx) {
            selectedProgramDayIdx = idx;
            renderProgramDayTabs();
            renderProgramExercises();
        }

        function renderProgramExercises() {
            const list = document.getElementById("programExerciseList");
            if (!list || !userProgram) return;
            const day = userProgram.days[selectedProgramDayIdx];
            list.innerHTML = "";

            if (!day || !day.exercises || day.exercises.length === 0) {
                list.innerHTML = `
                    <div class="empty-day-box" style="text-align:center; padding:20px; color:#6b7280;">
                        <div style="font-size:1.8rem; margin-bottom:4px;">🗓️</div>
                        <div style="font-weight:700; color:#9ca3af;">Bu güne henüz hareket eklenmedi</div>
                        <div style="font-size:0.75rem;">AI ile program oluştur veya soldan manuel hareket ekle.</div>
                    </div>
                `;
                return;
            }

            if (day.focus) {
                list.innerHTML += `<div class="badge-cyan" style="align-self:flex-start;">🎯 Odak: ${day.focus}</div>`;
            }

            day.exercises.forEach((ex, exIdx) => {
                const noteHtml = ex.note ? `<div style="font-size:0.72rem; color:#9ca3af; margin-top:2px;">💡 ${ex.note}</div>` : "";
                list.innerHTML += `
                    <div class="log-item" style="flex-direction:column; align-items:flex-start; gap:4px;">
                        <div style="display:flex; justify-content:space-between; width:100%; align-items:center;">
                            <span class="ex-title">${ex.name}</span>
                            <button onclick="removeExerciseFromProgram(${exIdx})">Sil</button>
                        </div>
                        <div style="font-size:0.8rem; color:#38bdf8; font-weight:700;">${ex.sets} Set × ${ex.reps} Tekrar</div>
                        ${noteHtml}
                    </div>
                `;
            });
        }

        function addExerciseToProgramDay(dayIdx, name, sets, reps, note) {
            if (!currentUser) return;
            if (!userProgram || !Array.isArray(userProgram.days)) userProgram = defaultEmptyProgram(5);
            if (dayIdx < 0 || dayIdx >= userProgram.days.length) return;

            const day = userProgram.days[dayIdx];
            const already = day.exercises.find(e => e.name.trim().toLowerCase() === name.trim().toLowerCase());
            if (already) {
                already.sets = sets;
                already.reps = reps;
                if (note) already.note = note;
            } else {
                day.exercises.push({ name: name, sets: sets, reps: reps, note: note || "" });
            }
            saveUserProgram(currentUser.username, userProgram);
            if (dayIdx === selectedProgramDayIdx) renderProgramExercises();
            renderProgramDayTabs();
        }

        function addManualExerciseToProgram() {
            if (!currentUser) return;
            const name = document.getElementById("progManualExercise").value.trim();
            const sets = parseInt(document.getElementById("progManualSets").value, 10) || 3;
            const reps = document.getElementById("progManualReps").value.trim() || "8-10";
            const note = document.getElementById("progManualNote").value.trim();

            if (!name) return alert("Lütfen hareket adını gir kral!");

            addExerciseToProgramDay(selectedProgramDayIdx, name, sets, reps, note);

            document.getElementById("progManualExercise").value = "";
            document.getElementById("progManualNote").value = "";
        }

        function removeExerciseFromProgram(exIdx) {
            if (!userProgram) return;
            const day = userProgram.days[selectedProgramDayIdx];
            day.exercises.splice(exIdx, 1);
            saveUserProgram(currentUser.username, userProgram);
            renderProgramExercises();
        }

        function clearProgramDay() {
            if (!userProgram) return;
            if (!confirm(`Gün ${selectedProgramDayIdx + 1} programını tamamen temizlemek istiyor musun?`)) return;
            userProgram.days[selectedProgramDayIdx].exercises = [];
            saveUserProgram(currentUser.username, userProgram);
            renderProgramExercises();
        }

        function refreshProgramDaySelectOptions() {
            const select = document.getElementById("addToProgramDaySelect");
            if (!select || !currentUser) return;
            const prog = userProgram && userProgram.days ? userProgram : getUserProgram(currentUser.username);
            const numDays = (prog && Array.isArray(prog.days) && prog.days.length > 0) ? prog.days.length : 5;
            const prevVal = select.value;
            select.innerHTML = "";
            for (let i = 0; i < numDays; i++) {
                const opt = document.createElement("option");
                opt.value = i;
                const focus = (prog && prog.days && prog.days[i] && prog.days[i].focus) ? ` · ${prog.days[i].focus}` : "";
                opt.innerText = `Gün ${i + 1}${focus}`;
                select.appendChild(opt);
            }
            if (prevVal !== "" && Number(prevVal) < numDays) select.value = prevVal;
            refreshProgramExerciseSelectOptions();
        }

        function refreshProgramExerciseSelectOptions() {
            const daySelect = document.getElementById("addToProgramDaySelect");
            const exSelect = document.getElementById("addToProgramExerciseSelect");
            if (!daySelect || !exSelect || !currentUser) return;

            const dayIdx = parseInt(daySelect.value, 10);
            const prog = userProgram && userProgram.days ? userProgram : getUserProgram(currentUser.username);
            exSelect.innerHTML = "";

            const dayExercises = (prog && Array.isArray(prog.days) && !isNaN(dayIdx) && prog.days[dayIdx] && Array.isArray(prog.days[dayIdx].exercises))
                ? prog.days[dayIdx].exercises : [];

            if (dayExercises.length === 0) {
                const opt = document.createElement("option");
                opt.value = "";
                opt.innerText = "Bu güne henüz hareket eklenmedi";
                exSelect.appendChild(opt);
                return;
            }

            const placeholderOpt = document.createElement("option");
            placeholderOpt.value = "";
            placeholderOpt.innerText = "— Hareket seç —";
            exSelect.appendChild(placeholderOpt);

            dayExercises.forEach((ex, exIdx) => {
                const opt = document.createElement("option");
                opt.value = exIdx;
                opt.innerText = `${ex.name} (${ex.sets}×${ex.reps})`;
                exSelect.appendChild(opt);
            });
        }

        function applyProgramExerciseToForm() {
            if (!currentUser) return;
            const daySelect = document.getElementById("addToProgramDaySelect");
            const exSelect = document.getElementById("addToProgramExerciseSelect");
            if (!daySelect || !exSelect || exSelect.value === "") return;

            const dayIdx = parseInt(daySelect.value, 10);
            const exIdx = parseInt(exSelect.value, 10);
            const prog = userProgram && userProgram.days ? userProgram : getUserProgram(currentUser.username);
            if (!prog || !prog.days || !prog.days[dayIdx]) return;
            const ex = prog.days[dayIdx].exercises[exIdx];
            if (!ex) return;

            // Hareket adi programdaki gibi otomatik doluyor
            document.getElementById("exerciseName").value = ex.name;

            // Tekrar: "8-10" gibi bir aralik ise ilk sayiyi al, tek sayiysa direkt kullan
            const repsMatch = String(ex.reps).match(/\d+/);
            document.getElementById("exerciseReps").value = repsMatch ? repsMatch[0] : "";

            // Set: programda yazan hedef set sayisi neyse (orn "4x8" -> 4) direkt onu yaz
            const setsMatch = String(ex.sets).match(/\d+/);
            document.getElementById("exerciseSet").value = setsMatch ? setsMatch[0] : (ex.sets || 1);

            // Sadece agirlik bos kalsin, kullanici doldursun
            document.getElementById("exerciseWeight").value = "";
            document.getElementById("exerciseWeight").focus();
        }

        function toggleProgramDaySelect() {
            const checked = document.getElementById("addToProgramCheck").checked;
            document.getElementById("addToProgramDaySelect").style.display = checked ? "block" : "none";
            document.getElementById("addToProgramExerciseSelect").style.display = checked ? "block" : "none";
            if (checked) refreshProgramDaySelectOptions();
        }

        async function generateAiProgram() {
            if (!currentUser) return alert("Lütfen önce giriş yap kral!");
            const numDays = parseInt(document.getElementById("programSplitSelect").value, 10) || 5;
            const btn = document.getElementById("generateProgramBtn");

            const prof = getUserProfileData(currentUser.username) || {};
            if (!prof.weight || !prof.height) {
                return alert("AI programı oluşturmadan önce Profil sekmesinden boy/kilo/hedef bilgini kaydetmelisin kral!");
            }

            const injuries = getUserInjuries(currentUser.username) || [];
            const health = getUserHealthLogs(currentUser.username) || {};
            const todayHealth = health[todayKey] || null;

            btn.disabled = true;
            const originalText = btn.innerText;
            btn.innerText = "🧠 Program hazırlanıyor...";

            try {
                const res = await fetch("/generate-program", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        profile_data: prof,
                        active_injuries: injuries,
                        today_health: todayHealth,
                        num_days: numDays
                    })
                });
                const data = await res.json();

                if (data.is_error || !data.program || !Array.isArray(data.program.days) || data.program.days.length === 0) {
                    alert("Program oluşturulamadı: " + (data.error_detail || "Bilinmeyen hata"));
                    return;
                }

                userProgram = {
                    days: data.program.days,
                    generated_at: new Date().toISOString(),
                    goal_used: prof.goal || ""
                };
                saveUserProgram(currentUser.username, userProgram);
                selectedProgramDayIdx = 0;
                loadProgramUI();
                alert("Programın hazır kral! Her gün için hareketleri, set ve tekrarları sol üstteki sekmelerden inceleyebilirsin. 🦍");
            } catch (err) {
                alert("Sunucu bağlantı hatası: " + err.message);
            } finally {
                btn.disabled = false;
                btn.innerText = originalText;
            }
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

        // ================= BARKOD TARAMA =================
        function openBarcodeScanner() {
            document.getElementById("barcodeModal").style.display = "flex";
            resetBarcodeScanner();
        }

        function closeBarcodeModalOnBg(e) {
            if (e.target.id === "barcodeModal") closeBarcodeScanner();
        }

        function stopBarcodeCamera() {
            if (barcodeScannerInstance) {
                barcodeScannerInstance.stop().then(() => {
                    barcodeScannerInstance.clear();
                    barcodeScannerInstance = null;
                }).catch(() => { barcodeScannerInstance = null; });
            }
        }

        function closeBarcodeScanner() {
            stopBarcodeCamera();
            document.getElementById("barcodeModal").style.display = "none";
            currentBarcodeProduct = null;
        }

        function resetBarcodeScanner() {
            stopBarcodeCamera();
            currentBarcodeProduct = null;
            document.getElementById("barcodeScanState").style.display = "block";
            document.getElementById("barcodeLoadingState").style.display = "none";
            document.getElementById("barcodeNotFoundState").style.display = "none";
            document.getElementById("barcodeResultState").style.display = "none";

            if (typeof Html5Qrcode === "undefined") {
                document.getElementById("barcodeReaderRegion").innerHTML =
                    "<div style='padding:20px; color:#ef4444; font-size:0.8rem; text-align:center;'>Kamera kütüphanesi yüklenemedi. İnternet bağlantını kontrol et.</div>";
                return;
            }

            barcodeScannerInstance = new Html5Qrcode("barcodeReaderRegion");
            const supportedFormats = [
                Html5QrcodeSupportedFormats.EAN_13,
                Html5QrcodeSupportedFormats.EAN_8,
                Html5QrcodeSupportedFormats.UPC_A,
                Html5QrcodeSupportedFormats.UPC_E
            ];

            barcodeScannerInstance.start(
                { facingMode: "environment" },
                { fps: 10, qrbox: { width: 250, height: 140 }, formatsToSupport: supportedFormats },
                (decodedText) => {
                    stopBarcodeCamera();
                    onBarcodeDetected(decodedText);
                },
                () => { /* kare basina tarama basarisiz - normal, sessizce yoksay */ }
            ).catch((err) => {
                document.getElementById("barcodeReaderRegion").innerHTML =
                    "<div style='padding:20px; color:#ef4444; font-size:0.8rem; text-align:center;'>Kameraya erişilemedi. Tarayıcı izinlerini kontrol et.</div>";
                console.error("Barkod kamera hatasi:", err);
            });
        }

        async function onBarcodeDetected(code) {
            document.getElementById("barcodeScanState").style.display = "none";
            document.getElementById("barcodeLoadingState").style.display = "block";

            try {
                const res = await fetch(`/api/barcode-lookup/${encodeURIComponent(code)}`);
                const data = await res.json();
                document.getElementById("barcodeLoadingState").style.display = "none";

                if (!data.found) {
                    document.getElementById("barcodeNotFoundState").style.display = "block";
                    return;
                }

                currentBarcodeProduct = data;
                document.getElementById("barcodeProductName").innerText = data.name;
                document.getElementById("barcodeProductPer100").innerText =
                    `${data.calories_100g} kcal, P:${data.protein_100g}g K:${data.carbs_100g}g Y:${data.fat_100g}g`;

                const defaultGrams = data.serving_size_g || 100;
                document.getElementById("barcodeAmountGrams").value = defaultGrams;
                document.getElementById("barcodeResultState").style.display = "flex";
                updateBarcodePreview();
            } catch (err) {
                document.getElementById("barcodeLoadingState").style.display = "none";
                document.getElementById("barcodeNotFoundState").style.display = "block";
                console.error("Barkod lookup hatasi:", err);
            }
        }

        function updateBarcodePreview() {
            if (!currentBarcodeProduct) return;
            const grams = parseFloat(document.getElementById("barcodeAmountGrams").value) || 0;
            const ratio = grams / 100.0;

            document.getElementById("barcodePreviewCal").innerText = Math.round(currentBarcodeProduct.calories_100g * ratio) + " kcal";
            document.getElementById("barcodePreviewPro").innerText = (currentBarcodeProduct.protein_100g * ratio).toFixed(1) + "g";
            document.getElementById("barcodePreviewCarb").innerText = (currentBarcodeProduct.carbs_100g * ratio).toFixed(1) + "g";
            document.getElementById("barcodePreviewFat").innerText = (currentBarcodeProduct.fat_100g * ratio).toFixed(1) + "g";
        }

        function addBarcodeProductToMeal() {
            if (!currentUser || !currentBarcodeProduct) return;
            const grams = parseFloat(document.getElementById("barcodeAmountGrams").value) || 0;
            if (grams <= 0) return alert("Lütfen geçerli bir miktar gir kral!");
            const ratio = grams / 100.0;

            const newMeal = {
                id: Date.now() + Math.floor(Math.random() * 1000),
                food_name: currentBarcodeProduct.name,
                items_summary: `${grams}g ${currentBarcodeProduct.name} (barkod)`,
                calories: Math.round(currentBarcodeProduct.calories_100g * ratio),
                protein: Number((currentBarcodeProduct.protein_100g * ratio).toFixed(1)),
                carbs: Number((currentBarcodeProduct.carbs_100g * ratio).toFixed(1)),
                fat: Number((currentBarcodeProduct.fat_100g * ratio).toFixed(1))
            };

            const targetDateStr = weekDaysData[selectedNutriDayIdx].fullDate;
            if (!weeklyNutrition[targetDateStr]) weeklyNutrition[targetDateStr] = [];
            weeklyNutrition[targetDateStr] = [...weeklyNutrition[targetDateStr], newMeal];
            saveUserWeeklyNutrition(currentUser.username, weeklyNutrition);
            renderSelectedDayNutrition();

            closeBarcodeScanner();
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


@app.get("/icon.png")
def serve_app_icon():
    icon_bytes = base64.b64decode(APP_ICON_PNG_BASE64)
    return Response(content=icon_bytes, media_type="image/png")


@app.get("/manifest.json")
def serve_pwa_manifest():
    manifest = {
        "name": "Looksmax Hub - Elite Performance & Coaching",
        "short_name": "Looksmax Hub",
        "description": "Hipertrofi, beslenme, biyometrik toparlanma ve fizik takibi.",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "orientation": "portrait",
        "background_color": "#0b0d12",
        "theme_color": "#0b0d12",
        "icons": [
            {"src": "/icon.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/icon.png", "sizes": "512x512", "type": "image/png"},
            {"src": "/icon.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
        ],
    }
    return JSONResponse(content=manifest)


@app.post("/api/auth/register", status_code=201)
def auth_register(payload: AuthRequest):
    if not AUTH_BACKEND_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "not_configured"})

    username = payload.username.strip()
    if not username or not payload.password_hash:
        raise HTTPException(status_code=400, detail="missing_fields")

    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cur.fetchone():
                conn.close()
                return JSONResponse(status_code=409, content={"detail": "username_taken"})

            stored_hash = hash_password_for_storage(payload.password_hash)
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, stored_hash)
            )
        conn.commit()
        conn.close()

        token = create_jwt_token(username)
        return {"token": token, "username": username}
    except Exception as e:
        logger.error(f"Auth register hatasi: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "server_error"})


@app.post("/api/auth/login")
def auth_login(payload: AuthRequest):
    if not AUTH_BACKEND_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "not_configured"})

    username = payload.username.strip()

    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT password_hash FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
        conn.close()

        if not row:
            return JSONResponse(status_code=404, content={"detail": "user_not_found"})

        if not verify_password(payload.password_hash, row["password_hash"]):
            return JSONResponse(status_code=401, content={"detail": "invalid_credentials"})

        token = create_jwt_token(username)
        return {"token": token, "username": username}
    except Exception as e:
        logger.error(f"Auth login hatasi: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "server_error"})


@app.get("/api/profile")
def get_profile_backend(username: str = Depends(require_auth_username)):
    if not AUTH_BACKEND_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "not_configured"})
    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT profile_data FROM user_profiles WHERE username = %s", (username,))
            row = cur.fetchone()
        conn.close()
        return {"profile_data": row["profile_data"] if row else None}
    except Exception as e:
        logger.error(f"Profil okuma hatasi ({username}): {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "server_error"})


@app.post("/api/profile")
def save_profile_backend(payload: ProfileSyncInput, username: str = Depends(require_auth_username)):
    if not AUTH_BACKEND_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "not_configured"})
    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_profiles (username, profile_data, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (username) DO UPDATE SET profile_data = EXCLUDED.profile_data, updated_at = now()
            """, (username, psycopg2.extras.Json(payload.profile_data)))
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Profil yazma hatasi ({username}): {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "server_error"})


@app.get("/api/program")
def get_program_backend(username: str = Depends(require_auth_username)):
    if not AUTH_BACKEND_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "not_configured"})
    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT program_data FROM user_programs WHERE username = %s", (username,))
            row = cur.fetchone()
        conn.close()
        return {"program_data": row["program_data"] if row else None}
    except Exception as e:
        logger.error(f"Program okuma hatasi ({username}): {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "server_error"})


@app.post("/api/program")
def save_program_backend(payload: ProgramSyncInput, username: str = Depends(require_auth_username)):
    if not AUTH_BACKEND_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "not_configured"})
    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_programs (username, program_data, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (username) DO UPDATE SET program_data = EXCLUDED.program_data, updated_at = now()
            """, (username, psycopg2.extras.Json(payload.program_data)))
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Program yazma hatasi ({username}): {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "server_error"})


@app.get("/api/workout-weeks")
def get_workout_weeks_backend(username: str = Depends(require_auth_username)):
    if not AUTH_BACKEND_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "not_configured"})
    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT week_key, logs_data FROM user_workout_weeks WHERE username = %s", (username,))
            rows = cur.fetchall()
        conn.close()
        weeks = {row["week_key"]: row["logs_data"] for row in rows}
        return {"weeks": weeks}
    except Exception as e:
        logger.error(f"Workout weeks okuma hatasi ({username}): {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "server_error"})


@app.post("/api/workout-weeks/{week_key}")
def save_workout_week_backend(week_key: str, payload: WorkoutWeekSyncInput, username: str = Depends(require_auth_username)):
    if not AUTH_BACKEND_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "not_configured"})
    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_workout_weeks (username, week_key, logs_data, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (username, week_key) DO UPDATE SET logs_data = EXCLUDED.logs_data, updated_at = now()
            """, (username, week_key, psycopg2.extras.Json(payload.logs)))
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Workout week yazma hatasi ({username}, {week_key}): {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "server_error"})


@app.get("/api/nutrition-weeks")
def get_nutrition_weeks_backend(username: str = Depends(require_auth_username)):
    if not AUTH_BACKEND_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "not_configured"})
    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT week_key, nutrition_data FROM user_nutrition_weeks WHERE username = %s", (username,))
            rows = cur.fetchall()
        conn.close()
        weeks = {row["week_key"]: row["nutrition_data"] for row in rows}
        return {"weeks": weeks}
    except Exception as e:
        logger.error(f"Nutrition weeks okuma hatasi ({username}): {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "server_error"})


@app.post("/api/nutrition-weeks/{week_key}")
def save_nutrition_week_backend(week_key: str, payload: NutritionWeekSyncInput, username: str = Depends(require_auth_username)):
    if not AUTH_BACKEND_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "not_configured"})
    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_nutrition_weeks (username, week_key, nutrition_data, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (username, week_key) DO UPDATE SET nutrition_data = EXCLUDED.nutrition_data, updated_at = now()
            """, (username, week_key, psycopg2.extras.Json(payload.nutrition)))
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Nutrition week yazma hatasi ({username}, {week_key}): {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "server_error"})


@app.post("/api/health-sync")
def sync_apple_health_webhook(payload: HealthSyncInput):
    recovery = compute_recovery_score(
        sleep_hours=payload.sleep_hours,
        hrv=payload.hrv_ms,
        resting_hr=payload.resting_hr
    )

    saved = False
    if AUTH_BACKEND_AVAILABLE:
        try:
            conn = get_auth_db_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_health_logs (
                        username, date, sleep_hours, deep_sleep_hours, hrv_ms, resting_hr,
                        avg_workout_hr, max_workout_hr, steps, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (username, date) DO UPDATE SET
                        sleep_hours = EXCLUDED.sleep_hours,
                        deep_sleep_hours = EXCLUDED.deep_sleep_hours,
                        hrv_ms = EXCLUDED.hrv_ms,
                        resting_hr = EXCLUDED.resting_hr,
                        avg_workout_hr = EXCLUDED.avg_workout_hr,
                        max_workout_hr = EXCLUDED.max_workout_hr,
                        steps = EXCLUDED.steps,
                        updated_at = now()
                """, (
                    payload.username, payload.date, payload.sleep_hours, payload.deep_sleep_hours,
                    payload.hrv_ms, payload.resting_hr, payload.avg_workout_hr, payload.max_workout_hr,
                    payload.steps
                ))
            conn.commit()
            conn.close()
            saved = True
        except Exception as e:
            # orn: bu username 'users' tablosunda yok (FK ihlali) - sessizce partial_success'e dus
            logger.error(f"Health sync DB yazma hatasi: {e}")
            traceback.print_exc()

    return {
        "status": "success" if saved else "partial_success",
        "message": f"{payload.date} tarihli Apple Health verisi işlendi." if saved
                    else f"{payload.date} verisi hesaplandı ama kaydedilemedi (backend yapılandırılmamış veya kullanıcı bulunamadı).",
        "recovery_metrics": recovery
    }


@app.get("/api/health-sync/{username}")
def get_synced_health_data(username: str):
    if not AUTH_BACKEND_AVAILABLE:
        return {"status": "not_configured", "logs": []}
    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM user_health_logs WHERE username = %s ORDER BY date DESC LIMIT 60",
                (username,)
            )
            rows = cur.fetchall()
        conn.close()
        logs = [dict(row) for row in rows]
        return {"status": "success", "logs": logs}
    except Exception as e:
        logger.error(f"Health sync okuma hatasi: {e}")
        return {"status": "error", "logs": [], "detail": str(e)}


@app.post("/api/health-log")
def save_manual_health_log(payload: HealthLogSyncInput, username: str = Depends(require_auth_username)):
    """Uygulama icindeki manuel saglik veri girisi icin (JWT korumali).
    Apple Watch webhook'uyla AYNI tabloyu (user_health_logs) kullanir - upsert semantigi
    sayesinde hangisi once/sonra gelirse gelsin ayni tarih uzerinde birbirini gunceller."""
    if not AUTH_BACKEND_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "not_configured"})
    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_health_logs (
                    username, date, sleep_hours, deep_sleep_hours, hrv_ms, resting_hr,
                    avg_workout_hr, max_workout_hr, steps, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (username, date) DO UPDATE SET
                    sleep_hours = EXCLUDED.sleep_hours,
                    deep_sleep_hours = EXCLUDED.deep_sleep_hours,
                    hrv_ms = EXCLUDED.hrv_ms,
                    resting_hr = EXCLUDED.resting_hr,
                    avg_workout_hr = EXCLUDED.avg_workout_hr,
                    max_workout_hr = EXCLUDED.max_workout_hr,
                    steps = EXCLUDED.steps,
                    updated_at = now()
            """, (
                username, payload.date, payload.sleep_hours, payload.deep_sleep_hours,
                payload.hrv_ms, payload.resting_hr, payload.avg_workout_hr, payload.max_workout_hr,
                payload.steps
            ))
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Manuel health log yazma hatasi ({username}): {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "server_error"})


@app.get("/api/injuries")
def get_injuries_backend(username: str = Depends(require_auth_username)):
    if not AUTH_BACKEND_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "not_configured"})
    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT injuries_data FROM user_injuries WHERE username = %s", (username,))
            row = cur.fetchone()
        conn.close()
        return {"injuries_data": row["injuries_data"] if row else None}
    except Exception as e:
        logger.error(f"Sakatlik okuma hatasi ({username}): {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "server_error"})


@app.post("/api/injuries")
def save_injuries_backend(payload: InjuriesSyncInput, username: str = Depends(require_auth_username)):
    if not AUTH_BACKEND_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "not_configured"})
    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_injuries (username, injuries_data, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (username) DO UPDATE SET injuries_data = EXCLUDED.injuries_data, updated_at = now()
            """, (username, psycopg2.extras.Json(payload.injuries_data)))
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Sakatlik yazma hatasi ({username}): {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "server_error"})


@app.get("/api/phases")
def get_phases_backend(username: str = Depends(require_auth_username)):
    if not AUTH_BACKEND_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "not_configured"})
    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT phases_data FROM user_phases WHERE username = %s", (username,))
            row = cur.fetchone()
        conn.close()
        return {"phases_data": row["phases_data"] if row else None}
    except Exception as e:
        logger.error(f"Fazlar okuma hatasi ({username}): {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "server_error"})


@app.post("/api/phases")
def save_phases_backend(payload: PhasesSyncInput, username: str = Depends(require_auth_username)):
    if not AUTH_BACKEND_AVAILABLE:
        return JSONResponse(status_code=503, content={"detail": "not_configured"})
    try:
        conn = get_auth_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_phases (username, phases_data, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (username) DO UPDATE SET phases_data = EXCLUDED.phases_data, updated_at = now()
            """, (username, psycopg2.extras.Json(payload.phases_data)))
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Fazlar yazma hatasi ({username}): {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "server_error"})

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

    injuries_text_audit = ", ".join(
        f"{i.get('area', '?')} ({i.get('severity', '?')})" for i in injuries
    ) if injuries else ""
    audit_query = f"{prof.get('goal', '')} hedefi haftalık antrenman ve beslenme denetimi {injuries_text_audit}".strip()
    knowledge_snippets = retrieve_knowledge_context(audit_query, k=4)
    knowledge_block = (
        f"\nBİLGİ BANKASI (kaynak dokümanlardan ilgili pasajlar — varsa raporuna dayanak yap, "
        f"yoksa görmezden gel, kaynak adını kullanıcıya söyleme):\n{knowledge_snippets}\n"
        if knowledge_snippets else ""
    )

    audit_prompt = f"""
Sen hem halden anlayan bilge bir mentor, hem de sıfır bahane kabul eden sert ve disiplinli bir 'Looksmax & Hipertrofi Başantrenörü'sün.
KULLANICI VERİLERİ:
- Profil: {prof.get('fullName', 'Bilinmiyor')}, Boy: {prof.get('height', '-')}cm, Kilo: {prof.get('weight', '-')}kg, Hedef: {prof.get('goal', '-')}
- Setler: {json.dumps(workouts, ensure_ascii=False) if workouts else "GİRİLMEDİ"}
- Beslenme: {json.dumps(nutrition, ensure_ascii=False) if nutrition else "GİRİLMEDİ"}
- Sağlık: {json.dumps(health, ensure_ascii=False) if health else "GİRİLMEDİ"}
- Aktif Sakatlıklar: {json.dumps(injuries, ensure_ascii=False) if injuries else "YOK"}
{knowledge_block}
KURALLAR:
1. SADECE TÜRKÇE YAZ, tek bir İngilizce cümle bile kullanma (hareket isimleri hariç).
2. ASLA düşünme adımı, tablo veya veri matrisi üretme.
3. Doğrudan net maddelerle koçluk karne raporunu dök:
- 🔥 **DURUM TESPİTİ:** Genel gidişat.
- 🩹 **SAKATLIK & REHABİLİTASYON:** Varsa güvenli hareket reçetesi ve izolasyon.
- 🏋️ **ANTRENMAN & OVERLOAD:** Ağırlıkların durumu ve zorlama emri.
- 🥗 **MUTFAK & DİSİPLİN:** Makroların denetimi.
- ⚡ **3 NET EMİR:** Bu hafta yapılacaklar.
4. Bilgi bankasında ilgili bir pasaj varsa raporuna doğal şekilde harmanla, doğrudan alıntılama.
5. UZUNLUK KURALI (ÇOK ÖNEMLİ): Her bölümde EN FAZLA 3 kısa madde kullan. Aşırı ayrıntıya (örn. dakika dakika ısı/soğuk protokolü, spesifik su sıcaklığı gibi gereksiz detaylara) GİRME — öz, vurucu ve uygulanabilir ol. Rapor kesinlikle yarım cümlede bitmemeli; bitiremeyeceğin kadar uzun yazacaksan, en başından beri daha kısa ve öz yaz.
"""

    def _run_audit_completion(prompt_text: str, token_budget: int):
        completion = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt_text}],
            model=active_model,
            temperature=0.3,
            max_tokens=token_budget
        )
        choice = completion.choices[0]
        return (choice.message.content or ""), getattr(choice, "finish_reason", None)

    try:
        active_model = get_best_available_model()
        raw_report, finish_reason = _run_audit_completion(audit_prompt, 2000)

        if finish_reason == "length":
            # Yanit token siniri yuzunden yarim kaldi - daha kisa yazmasi icin bir kez daha dene
            logger.warning("Coach audit yaniti kesildi (finish_reason=length), daha kisa versiyon icin tekrar deneniyor.")
            shorter_prompt = audit_prompt + (
                "\n\nEK UYARI: Önceki yanıtın çok uzundu ve kesildi. Bu sefer HER bölümde EN FAZLA 2 kısa madde kullan, "
                "hiçbir bölümü uzatma, kısa ve tamamlanmış bir rapor yaz."
            )
            retry_report, _ = _run_audit_completion(shorter_prompt, 1600)
            if retry_report.strip():
                raw_report = retry_report

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

    knowledge_snippets = retrieve_knowledge_context(data.user_message, k=4)
    knowledge_block = (
        f"\nBİLGİ BANKASI (kaynak dokümanlardan çekilen ilgili pasajlar — varsa cevabını buna dayandır, "
        f"yoksa görmezden gel, ASLA kaynak adını veya bu başlığı kullanıcıya söyleme):\n{knowledge_snippets}\n"
        if knowledge_snippets else ""
    )

    system_prompt = f"""
Sen sporcusunu çok iyi anlayan ama asla laubaliliğe izin vermeyen bilge ve disiplinli bir 'Looksmax & Hipertrofi Başantrenörü'sün.
KULLANICI: {profile_context}
SAĞLIK: {health_context}
SAKATLIK DURUMU: {injuries_context}
SETLER: {user_context}
{knowledge_block}
FORMAT VE ÇIKTI KURALLARI (MUTLAK KURAL):
1. SADECE TÜRKÇE YAZ. Tek bir İngilizce cümle, başlık veya kelime bile kullanma (özel isimler/hareket adları hariç, örn "Bench Press" kalabilir).
2. ASLA markdown başlık (##, ###) kullanma. ASLA numaralı liste (1. 2. 3. ...) ile program/plan dökme.
3. ASLA DÜŞÜNME ADIMI, TABLO VEYA İŞLEM LİSTESİ YAZMA.
4. Uzun uzun 'Konu / Durum / Öneri' tabloları dökmek KESİNLİKLE YASAKTIR.
5. Yanıtını doğrudan, net, kısa paragraflar ve tire (-) ile başlayan maddeler halinde, sohbet tonunda ver — resmi bir doküman gibi değil.
6. Sakatlık varsa: Güvenli alternatif açıyı söyle ve 1 rehabilitasyon egzersizi emret.
7. Bilgi bankasında ilgili bir pasaj varsa onu kendi cümlelerinle harmanla, doğrudan alıntılama.
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
        clean_text = ""

        for attempt in range(1, 3):  # ilk deneme + (format ihlali veya kesilme) olursa 1 retry
            attempt_messages = list(messages)
            if attempt > 1:
                attempt_messages.append({
                    "role": "system",
                    "content": (
                        "UYARI: ÖNCEKİ YANITIN KURALLARA UYMADI (İngilizce'ye kaydın, başlık/numaralı liste "
                        "kullandın ya da yanıt yarım kaldı). BU SEFER SADECE TÜRKÇE, KISA VE TAMAMLANMIŞ, "
                        "DÜZ CÜMLELER VE TİRE (-) İLE BAŞLAYAN MADDELER KULLAN. BAŞLIK (##), NUMARALI LİSTE "
                        "(1. 2. 3.) VE İNGİLİZCE KESİNLİKLE YASAK."
                    )
                })

            chat_completion = client.chat.completions.create(
                messages=attempt_messages,
                model=active_model,
                temperature=0.3 if attempt == 1 else 0.15,
                max_tokens=700,
            )
            choice = chat_completion.choices[0]
            raw_text = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
            clean_text = strip_thinking_and_tables(raw_text)

            if finish_reason != "length" and not violates_coach_format_rules(clean_text):
                break

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


@app.get("/api/barcode-lookup/{barcode}")
def barcode_lookup(barcode: str):
    """Open Food Facts uzerinden barkod ile gercek urun besin degerlerini ceker.
    API key gerektirmez, urllib (zaten stdlib) disinda ekstra bagimlilik yok."""
    clean_barcode = re.sub(r'\D', '', barcode or "")
    if not clean_barcode:
        return {"found": False, "error": "Geçersiz barkod."}

    try:
        url = f"https://world.openfoodfacts.org/api/v2/product/{clean_barcode}.json"
        req = urllib.request.Request(url, headers={"User-Agent": "LooksmaxHub/1.0 (fitness-app)"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if data.get("status") != 1 or "product" not in data:
            return {"found": False, "barcode": clean_barcode}

        product = data["product"]
        nutriments = product.get("nutriments", {}) or {}

        name = (
            product.get("product_name_tr")
            or product.get("product_name")
            or product.get("generic_name")
            or "Bilinmeyen Ürün"
        ).strip()
        brand = (product.get("brands") or "").split(",")[0].strip()
        display_name = f"{brand} {name}".strip() if brand and brand.lower() not in name.lower() else name

        cal_100g = nutriments.get("energy-kcal_100g")
        if cal_100g is None:
            kj = nutriments.get("energy_100g") or nutriments.get("energy-kj_100g")
            if kj is not None:
                try:
                    cal_100g = float(kj) / 4.184
                except (TypeError, ValueError):
                    cal_100g = None

        if cal_100g is None:
            return {"found": False, "barcode": clean_barcode, "error": "Bu ürün için besin değeri verisi yok."}

        def _safe_float(v):
            try:
                return round(float(v), 1)
            except (TypeError, ValueError):
                return 0.0

        serving_size_g = None
        serving_size_raw = product.get("serving_size", "") or ""
        m = re.search(r'([\d.,]+)\s*g', serving_size_raw)
        if m:
            try:
                serving_size_g = float(m.group(1).replace(",", "."))
            except ValueError:
                serving_size_g = None

        return {
            "found": True,
            "barcode": clean_barcode,
            "name": display_name,
            "calories_100g": _safe_float(cal_100g),
            "protein_100g": _safe_float(nutriments.get("proteins_100g")),
            "carbs_100g": _safe_float(nutriments.get("carbohydrates_100g")),
            "fat_100g": _safe_float(nutriments.get("fat_100g")),
            "serving_size_g": serving_size_g,
        }
    except Exception as e:
        logger.error(f"Barkod lookup hatasi ({clean_barcode}): {e}")
        return {"found": False, "barcode": clean_barcode, "error": "Ürün veritabanına ulaşılamadı."}


def _coerce_int(value: Any, default: int = 3) -> int:
    """LLM 'sets' alanina bazen '3-4', '3 set', 3.0 gibi seyler dondurebiliyor.
    Pydantic'e gitmeden once temizleyip guvenli bir int'e ceviriyoruz."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        match = re.search(r'\d+', value)
        if match:
            return int(match.group())
    return default


def _sanitize_program_json(parsed_json: Dict[str, Any]) -> Dict[str, Any]:
    """LLM ciktisindaki kucuk format sapmalarini (sets stringi, eksik note,
    reps'in sayi olarak gelmesi vb.) pydantic validasyonundan ONCE duzeltir.
    Boylece kucuk bir format hatasi yuzunden tum istek patlamiyor."""
    days = parsed_json.get("days")
    if not isinstance(days, list):
        return parsed_json

    for day in days:
        if not isinstance(day, dict):
            continue
        day["day_name"] = str(day.get("day_name") or "Gün")
        day["focus"] = str(day.get("focus") or "")
        exercises = day.get("exercises")
        if not isinstance(exercises, list):
            day["exercises"] = []
            continue
        for ex in exercises:
            if not isinstance(ex, dict):
                continue
            ex["name"] = str(ex.get("name") or "Hareket")
            ex["sets"] = _coerce_int(ex.get("sets"), default=3)
            ex["reps"] = str(ex.get("reps") if ex.get("reps") is not None else "8-10")
            note_val = ex.get("note")
            ex["note"] = "" if note_val is None else str(note_val)

    return parsed_json


class ProgramExercise(BaseModel):
    name: str = Field(description="Hareketin adi (orn: Bench Press, Squat, Lat Pulldown)")
    sets: int = Field(description="Set sayisi, orn 3, 4, 5")
    reps: str = Field(description="Tekrar araligi, orn '8-10', '12-15', '5'")
    note: Optional[str] = Field(default="", description="Kisa, opsiyonel teknik/güvenlik notu (sakatlik varsa uyari)")


class ProgramDay(BaseModel):
    day_name: str = Field(description="Gun adi, orn 'Gün 1'")
    focus: str = Field(description="O gunun odak bolgesi, orn 'Göğüs & Triceps', 'Push', 'Üst Vücut'")
    exercises: List[ProgramExercise]


class WorkoutProgramResponse(BaseModel):
    days: List[ProgramDay]


class GenerateProgramInput(BaseModel):
    profile_data: Optional[dict] = {}
    active_injuries: Optional[list] = []
    today_health: Optional[dict] = None
    num_days: int = 5


def generate_workout_program_with_llm(payload: GenerateProgramInput):
    """Program uretir. (program_dict, None) basarili; (None, hata_mesaji) basarisiz doner."""
    if not client:
        return None, "GROQ_API_KEY bulunamadı."

    prof = payload.profile_data or {}
    injuries = payload.active_injuries or []
    health = payload.today_health or {}
    num_days = max(3, min(6, payload.num_days or 5))

    injuries_text = ", ".join(
        f"{i.get('area', '?')} ({i.get('severity', '?')}: {i.get('details', '?')})" for i in injuries
    ) if injuries else "Aktif sakatlık yok."

    health_text = (
        f"Uyku: {health.get('sleep_hours', '-')}s, HRV: {health.get('hrv_ms', '-')}ms, Dinlenik Nabız: {health.get('resting_hr', '-')}bpm"
        if health else "Bugüne ait toparlanma verisi girilmedi."
    )

    goal = prof.get('goal', 'Recomposition')
    goal_query_hints = {
        "Lean Bulk": "hacim artirma kas kutlesi kazanma bulk progressive overload agir bilesik hareketler",
        "Aggressive Cut": "yag yakimi kalori acigi cut kas koruma kondisyon metabolik finisher",
        "Recomposition": "recomposition kas koruma yag kaybi dengeli hacim orta tekrar",
        "Maintenance": "idame antrenmani dengeli hacim surdurulebilir program",
    }.get(goal, "hipertrofi antrenman programlama hacim periyotlama")

    program_query = f"{goal} {goal_query_hints} {num_days} gunluk split antrenman programi {injuries_text}"
    knowledge_snippets = retrieve_knowledge_context(program_query, k=6)
    knowledge_block = (
        f"\nBİLGİ BANKASI (yüklenen PDF/makalelerden bu isteğe en ilgili pasajlar):\n{knowledge_snippets}\n"
        f"YUKARIDAKİ BİLGİ BANKASI VARSA, programı bunun üzerine kur — split mantığı, hacim/yoğunluk önerisi, "
        f"periyotlama ya da sakatlık protokolü kaynaklarda geçiyorsa jenerik şablonlar yerine ONU esas al. "
        f"Kaynak adını veya bu başlığı kullanıcıya asla söyleme.\n"
        if knowledge_snippets else ""
    )

    system_prompt = f"""
Sen elit seviyede bir hipertrofi ve güç antrenörüsün. Görevin, kullanıcıya TAM OLARAK {num_days} GÜNLÜK
kişiye özel bir antrenman programı (split) çıkarmak ve YALNIZCA JSON formatında,
WorkoutProgramResponse şemasına uygun cikti vermek.

KULLANICI PROFİLİ:
- Boy: {prof.get('height', '-')}cm, Kilo: {prof.get('weight', '-')}kg, Yaş: {prof.get('age', '-')}, Yağ Oranı: %{prof.get('bodyfat', '-')}
- Hedef: {goal}
- Aktivite Seviyesi: {prof.get('activity', '-')}

AKTİF SAKATLIKLAR: {injuries_text}
BUGÜNKÜ TOPARLANMA: {health_text}
{knowledge_block}
KURALLAR:
1. Tam olarak {num_days} gün oluştur (days dizisinde {num_days} eleman olmalı). Dinlenme günü ekleme, sadece antrenman günleri.
2. Split seçimini gün sayısına göre mantıklı yap: 3 gün=Full Body veya Push/Pull/Legs, 4 gün=Upper/Lower x2, 5 gün=Bro Split (Göğüs, Sırt, Bacak, Omuz, Kol), 6 gün=Push/Pull/Legs x2.
3. Her gün için 5-7 hareket belirle.
4. HEDEFE GÖRE BİLİMSEL OLARAK FARKLILAŞTIR (bu kurala sıkı uy):
   - Hedef "Lean Bulk" ise: Hacim ve progressive overload önceliklidir. Ağır bileşik hareketlere (bench, squat, deadlift, row, overhead press) ağırlık ver, set sayısını orta-yüksek tut (4-5 set), tekrar aralığını güç-hipertrofi karışımı seç (örn '6-10'), kas kütlesi için ekstra izolasyon hareketi eklemekten çekinme.
   - Hedef "Aggressive Cut" ise: Kullanıcı kalori açığında olduğu için toparlanma kapasitesi kısıtlıdır. Hacmi ölçülü tut (aşırı yorma, 3-4 set), tekrar aralığını biraz yükselt (örn '10-15') hem kas korumak hem ekstra kalori harcamak için, günün sonuna kısa bir kondisyon/metabolik finisher hareketi ekle, notlarda dinlenme sürelerini kısaltmayı öner.
   - Hedef "Recomposition" ise: Kas koruma ve yağ kaybı dengeli hedeflenir. Orta hacim (3-4 set), orta tekrar aralığı (örn '8-12'), bileşik ve izolasyon hareketlerini dengeli dağıt.
   - Hedef "Maintenance" ise: Sürdürülebilir, aşırı yormayan orta hacim (3 set) ve çeşitlilik önceliklidir.
5. AKTİF SAKATLIK VARSA O BÖLGEYİ ZORLAYAN HAREKETLERİ KESİNLİKLE PROGRAMA KOYMA, güvenli alternatifleri seç ve note alanına kısa uyarı yaz.
6. Bugünkü toparlanma skoru düşükse (uyku az, HRV düşük, nabız yüksek gibi belirtiler varsa) o günün hacmini hafif azalt ve note'a belirt.
7. Bilgi bankasında ilgili bir pasaj varsa program tasarımının TEMELİ bu olsun — jenerik şablon değil, kaynaktaki metodolojiyi yansıt.
8. "sets" alanı SADECE düz bir tam sayı olmalı (örn 4), ASLA '3-4' gibi bir aralık veya metin yazma. Aralık gerekiyorsa onu "reps" alanına yaz.
9. "note" alanı olmayan hareketlerde boş string "" kullan, ASLA null/None döndürme.
10. ASLA açıklama, markdown, yorum ekleme. Sadece saf JSON döndür.
"""

    last_error = "Bilinmeyen hata"
    max_attempts = 2

    for attempt in range(1, max_attempts + 1):
        try:
            active_model = get_best_available_model()
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{num_days} günlük programımı oluştur."}
                ],
                model=active_model,
                response_format={"type": "json_object"},
                temperature=0.4 if attempt == 1 else 0.15  # tekrar denemede daha tutarli/duz cikti icin sicakligi dusur
            )
            raw_content = completion.choices[0].message.content
            parsed_json = json.loads(raw_content)
            parsed_json = _sanitize_program_json(parsed_json)
            data = WorkoutProgramResponse(**parsed_json)

            if not data.days:
                raise ValueError("LLM boş bir gün listesi döndürdü.")
            if any(len(d.exercises) == 0 for d in data.days):
                raise ValueError("LLM en az bir günü hareketsiz bıraktı.")

            return data.model_dump(), None

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Program üretim denemesi {attempt}/{max_attempts} başarısız: {e}")

    logger.error(f"LLM Program Uretim Hatasi (tum denemeler basarisiz): {last_error}")
    traceback.print_exc()
    return None, last_error


@app.post("/generate-program")
def generate_program(payload: GenerateProgramInput):
    if not client:
        return {
            "program": None,
            "is_error": True,
            "error_detail": "GROQ_API_KEY bulunamadı! Lütfen sunucu ortam değişkenine veya .env dosyasına ekle."
        }

    program, error_detail = generate_workout_program_with_llm(payload)
    if not program:
        return {
            "program": None,
            "is_error": True,
            "error_detail": f"AI programı oluştururken bir hata oluştu: {error_detail}" if error_detail
                             else "AI programı oluştururken bir hata oluştu. Lütfen tekrar dene."
        }

    return {"program": program, "is_error": False}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
