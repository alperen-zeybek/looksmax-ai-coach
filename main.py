import os
import time
import sqlite3
import hashlib
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chromadb
from groq import Groq

# ----------------- VERİTABANI KURULUMU (SQLite) -----------------
DB_FILE = "coach_app.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS workouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            exercise TEXT NOT NULL,
            weight REAL NOT NULL,
            reps INTEGER NOT NULL,
            date TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ----------------- GROQ & CHROMADB -----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Looksmax Coach & Tracker")

chroma_client = chromadb.PersistentClient(path="./looksmax_db")
collection = chroma_client.get_or_create_collection(name="looksmax_knowledge")

# ----------------- PYDANTIC MODELLERİ -----------------
class AuthInput(BaseModel):
    username: str
    password: str

class WorkoutLogInput(BaseModel):
    user_id: int
    exercise: str
    weight: float
    reps: int
    date: str

class ChatInput(BaseModel):
    user_id: Optional[int] = None
    user_message: str
    image_base64: Optional[str] = None
    history: List[dict] = []

# ----------------- FRONTEND HTML / CSS / JS -----------------
HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Looksmax AI Coach & Performance Tracker</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background-color: #0b0d10; color: #e5e7eb; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

        /* Auth Modal */
        .auth-overlay { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.88); display: flex; justify-content: center; align-items: center; z-index: 9999; backdrop-filter: blur(6px); }
        .auth-box { background: #131720; border: 1px solid #1f2937; padding: 32px; border-radius: 16px; width: 350px; display: flex; flex-direction: column; gap: 14px; box-shadow: 0 10px 30px rgba(0,242,254,0.15); }
        .auth-box h2 { font-size: 1.25rem; font-weight: 700; color: #00f2fe; text-align: center; }
        .auth-box input { background: #0b0d10; border: 1px solid #2b3547; color: #fff; padding: 12px; border-radius: 8px; font-size: 0.9rem; outline: none; }
        .auth-box input:focus { border-color: #00f2fe; }
        .auth-box button { background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); color: #000; border: none; font-weight: 700; padding: 12px; border-radius: 8px; cursor: pointer; }
        .auth-toggle { font-size: 0.8rem; color: #9ca3af; text-align: center; cursor: pointer; }
        .auth-toggle b { color: #00f2fe; }

        /* Üst Menü / Navbar */
        .navbar { height: 64px; background: #0e1117; border-bottom: 1px solid #1c212b; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; flex-shrink: 0; }
        .nav-brand { font-size: 1.15rem; font-weight: 800; color: #00f2fe; display: flex; align-items: center; gap: 8px; }
        .nav-tabs { display: flex; gap: 8px; background: #161b24; padding: 4px; border-radius: 10px; border: 1px solid #232a38; }
        .tab-btn { background: transparent; border: none; color: #9ca3af; padding: 8px 18px; border-radius: 8px; font-weight: 600; font-size: 0.85rem; cursor: pointer; transition: 0.2s; }
        .tab-btn.active { background: #00f2fe; color: #000; font-weight: 700; }
        .nav-user { display: flex; align-items: center; gap: 12px; }
        .user-badge { font-size: 0.8rem; background: #161b24; padding: 6px 12px; border-radius: 8px; color: #10b981; border: 1px solid #2b3547; }
        .logout-btn { background: none; border: none; color: #ef4444; cursor: pointer; font-size: 0.8rem; font-weight: 600; }

        /* Sayfa İçerik Alanı */
        .main-content { flex: 1; display: flex; justify-content: center; overflow: hidden; position: relative; }
        .tab-view { display: none; width: 100%; max-width: 1200px; height: 100%; padding: 20px; }
        .tab-view.active { display: flex; }

        /* --- 1. KOÇ / CHAT GÖRÜNÜMÜ --- */
        #coachView { flex-direction: column; max-width: 900px; }
        .chat-container { flex: 1; display: flex; flex-direction: column; background: #131720; border-radius: 16px; border: 1px solid #1c212b; overflow: hidden; }
        .messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 14px; }
        .msg { max-width: 82%; padding: 13px 16px; border-radius: 12px; font-size: 0.9rem; line-height: 1.5; word-wrap: break-word; }
        .msg.user { align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 4px; }
        .msg.coach { align-self: flex-start; background: #1a202c; border: 1px solid #283347; border-bottom-left-radius: 4px; }
        .msg img.preview-img { max-width: 220px; border-radius: 8px; margin-bottom: 8px; display: block; }
        
        .preview-container { display: none; padding: 8px 16px; background: #0e1117; align-items: center; gap: 10px; border-top: 1px solid #1c212b; }
        .preview-container img { height: 45px; border-radius: 6px; border: 1px solid #00f2fe; }
        .preview-container button { background: #ef4444; color: white; border: none; border-radius: 50%; width: 20px; height: 20px; cursor: pointer; }

        .chat-input-area { padding: 14px 18px; border-top: 1px solid #1c212b; background: #0e1117; display: flex; gap: 10px; align-items: center; }
        .chat-input { flex: 1; background: #1a202c; border: 1px solid #283347; color: #fff; padding: 12px 16px; border-radius: 10px; font-size: 0.9rem; outline: none; }
        .chat-input:focus { border-color: #00f2fe; }
        .file-label { background: #1a202c; border: 1px solid #283347; color: #00f2fe; padding: 10px 14px; border-radius: 10px; cursor: pointer; font-size: 1rem; }
        input[type="file"] { display: none; }
        .send-btn { background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); color: #000; border: none; font-weight: 700; padding: 12px 22px; border-radius: 10px; cursor: pointer; }

        /* --- 2. PROGRESSIVE OVERLOAD GÖRÜNÜMÜ --- */
        #overloadView { gap: 20px; }
        .overload-left { width: 38%; display: flex; flex-direction: column; gap: 16px; }
        .overload-right { width: 62%; display: flex; flex-direction: column; gap: 16px; }

        .card { background: #131720; border: 1px solid #1c212b; border-radius: 14px; padding: 18px; display: flex; flex-direction: column; gap: 12px; }
        .card-title { font-size: 0.95rem; font-weight: 700; color: #00f2fe; display: flex; justify-content: space-between; align-items: center; }

        .overload-form { display: flex; flex-direction: column; gap: 10px; }
        .overload-form input, .overload-form select { background: #0b0d10; border: 1px solid #2b3547; color: #fff; padding: 10px 12px; border-radius: 8px; font-size: 0.85rem; outline: none; }
        .overload-form input:focus, .overload-form select:focus { border-color: #00f2fe; }
        .form-row { display: flex; gap: 8px; }
        .btn-save { background: #00f2fe; color: #000; border: none; font-weight: 700; padding: 11px; border-radius: 8px; cursor: pointer; }

        .history-box { flex: 1; overflow-y: auto; max-height: 380px; display: flex; flex-direction: column; gap: 8px; }
        .history-item { display: flex; justify-content: space-between; align-items: center; background: #0b0d10; padding: 10px 14px; border-radius: 8px; font-size: 0.85rem; border: 1px solid #1c212b; }
        .history-item .ex-name { font-weight: 600; color: #fff; }
        .history-item .ex-stat { color: #00f2fe; font-weight: 700; }
        .history-item button { background: none; border: none; color: #ef4444; cursor: pointer; font-size: 0.85rem; }

        .chart-container-box { flex: 1; min-height: 320px; position: relative; }

        @media (max-width: 900px) {
            body { overflow: auto; height: auto; }
            .navbar { flex-direction: column; height: auto; padding: 12px; gap: 10px; }
            .main-content { overflow: visible; height: auto; }
            .tab-view { height: auto; flex-direction: column !important; }
            .overload-left, .overload-right { width: 100%; }
            #coachView { height: 80vh; }
        }
    </style>
</head>
<body>

    <!-- AUTH POPUP -->
    <div class="auth-overlay" id="authOverlay">
        <div class="auth-box">
            <h2 id="authTitle">⚡ GİRİŞ YAP</h2>
            <input type="text" id="authUsername" placeholder="Kullanıcı Adı" />
            <input type="password" id="authPassword" placeholder="Şifre" />
            <button id="authSubmitBtn" onclick="handleAuthSubmit()">Giriş Yap</button>
            <div class="auth-toggle" id="authToggle" onclick="toggleAuthMode()">Hesabın yok mu? <b>Kayıt Ol</b></div>
        </div>
    </div>

    <!-- ÜST NAVBAR & TABLAR -->
    <div class="navbar">
        <div class="nav-brand">⚡ LOOKSMAX PRO</div>
        <div class="nav-tabs">
            <button class="tab-btn active" id="tabCoachBtn" onclick="switchTab('coach')">💬 AI Koç</button>
            <button class="tab-btn" id="tabOverloadBtn" onclick="switchTab('overload')">📈 Progressive Overload</button>
        </div>
        <div class="nav-user">
            <div class="user-badge" id="activeUserName">Giriş Yok</div>
            <button class="logout-btn" onclick="logout()">Çıkış</button>
        </div>
    </div>

    <!-- ANA İÇERİK -->
    <div class="main-content">

        <!-- 1. TAB: AI KOÇ -->
        <div class="tab-view active" id="coachView">
            <div class="chat-container">
                <div class="messages" id="chatBox">
                    <div class="msg coach">Selam kral! Ben senin Looksmax & Overload koçunum. Antrenman taktikleri sorabilir, formunu ve yemek fotoğraflarını atarak makro analizi alabilirsin.</div>
                </div>

                <div class="preview-container" id="previewContainer">
                    <img id="imagePreview" src="" alt="Görsel" />
                    <button onclick="clearImage()">✕</button>
                    <span style="font-size:0.75rem; color:#9ca3af;">Görsel eklendi</span>
                </div>

                <div class="chat-input-area">
                    <label class="file-label" for="imageInput" title="Fotoğraf Yükle">📷</label>
                    <input type="file" id="imageInput" accept="image/*" onchange="handleImageSelect(event)" />
                    <input type="text" class="chat-input" id="userInput" placeholder="Koça soru sor..." onkeypress="handleKey(event)" />
                    <button class="send-btn" id="sendBtn" onclick="sendMessage()">Gönder</button>
                </div>
            </div>
        </div>

        <!-- 2. TAB: PROGRESSIVE OVERLOAD & HAREKETLER -->
        <div class="tab-view" id="overloadView">
            <!-- Sol: Hareket Ekle & Günün Hareketleri -->
            <div class="overload-left">
                <div class="card">
                    <div class="card-title">➕ Set Kaydet</div>
                    <div class="overload-form">
                        <input type="text" id="exerciseName" placeholder="Hareket Adı (Örn: Bench Press, Incline DB)" list="exerciseList" />
                        <datalist id="exerciseList">
                            <option value="Bench Press">
                            <option value="Incline Dumbbell Press">
                            <option value="Squat">
                            <option value="Deadlift">
                            <option value="Barbell Row">
                            <option value="Overhead Press">
                            <option value="Lateral Raise">
                            <option value="Pull-up">
                        </datalist>
                        <div class="form-row">
                            <input type="number" id="exerciseWeight" placeholder="Kg" step="0.5" style="flex:1;" />
                            <input type="number" id="exerciseReps" placeholder="Tekrar" style="flex:1;" />
                            <input type="text" id="exerciseDate" placeholder="Tarih" style="flex:1.2;" />
                        </div>
                        <button class="btn-save" onclick="addWorkoutLog()">Seti İşle & Grafiğe Ekle</button>
                    </div>
                </div>

                <div class="card" style="flex:1;">
                    <div class="card-title">📋 Kayıtlı Setlerin Geçmişi</div>
                    <div class="history-box" id="historyList"></div>
                </div>
            </div>

            <!-- Sağ: Progressive Overload Grafiği -->
            <div class="overload-right">
                <div class="card" style="height: 100%;">
                    <div class="card-title">
                        <span>📊 Gelişim & Overload Eğrisi</span>
                        <select id="chartExerciseSelect" onchange="updateChart()" style="background:#0b0d10; border:1px solid #2b3547; color:#00f2fe; padding:6px 12px; border-radius:6px; font-weight:600; outline:none;"></select>
                    </div>
                    <div class="chart-container-box">
                        <canvas id="progressionChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

    </div>

    <script>
        let currentUser = JSON.parse(localStorage.getItem("active_user") || "null");
        let isRegisterMode = false;
        let workoutLogs = [];
        let chartInstance = null;

        document.getElementById("exerciseDate").value = new Date().toLocaleDateString('tr-TR', { day: '2-digit', month: '2-digit' });

        // TAB GEÇİŞİ
        function switchTab(tab) {
            document.getElementById("tabCoachBtn").classList.toggle("active", tab === 'coach');
            document.getElementById("tabOverloadBtn").classList.toggle("active", tab === 'overload');
            document.getElementById("coachView").classList.toggle("active", tab === 'coach');
            document.getElementById("overloadView").classList.toggle("active", tab === 'overload');
            if (tab === 'overload') {
                setTimeout(updateChart, 100);
            }
        }

        // AUTH
        function checkAuth() {
            if (!currentUser) {
                document.getElementById("authOverlay").style.display = "flex";
            } else {
                document.getElementById("authOverlay").style.display = "none";
                document.getElementById("activeUserName").innerText = "👤 " + currentUser.username;
                loadUserWorkouts();
            }
        }

        function toggleAuthMode() {
            isRegisterMode = !isRegisterMode;
            document.getElementById("authTitle").innerText = isRegisterMode ? "⚡ KAYIT OL" : "⚡ GİRİŞ YAP";
            document.getElementById("authSubmitBtn").innerText = isRegisterMode ? "Hesap Oluştur" : "Giriş Yap";
            document.getElementById("authToggle").innerHTML = isRegisterMode ? "Zaten hesabın var mı? <b>Giriş Yap</b>" : "Hesabın yok mu? <b>Kayıt Ol</b>";
        }

        async function handleAuthSubmit() {
            const u = document.getElementById("authUsername").value.trim();
            const p = document.getElementById("authPassword").value.trim();
            if (!u || !p) return alert("Kullanıcı adı ve şifre girin!");

            const endpoint = isRegisterMode ? "/register" : "/login";
            try {
                const res = await fetch(endpoint, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ username: u, password: p })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || "İşlem başarısız");

                currentUser = { id: data.user_id, username: data.username };
                localStorage.setItem("active_user", JSON.stringify(currentUser));
                checkAuth();
            } catch (err) {
                alert(err.message);
            }
        }

        function logout() {
            localStorage.removeItem("active_user");
            currentUser = null;
            location.reload();
        }

        // WORKOUT & CHART
        async function loadUserWorkouts() {
            if (!currentUser) return;
            try {
                const res = await fetch(`/workouts/${currentUser.id}`);
                workoutLogs = await res.json();
                populateDropdown();
                renderHistory();
                updateChart();
            } catch (err) {
                console.error("Setler yüklenemedi", err);
            }
        }

        async function addWorkoutLog() {
            if (!currentUser) return;
            const name = document.getElementById("exerciseName").value.trim();
            const weight = parseFloat(document.getElementById("exerciseWeight").value);
            const reps = parseInt(document.getElementById("exerciseReps").value);
            const date = document.getElementById("exerciseDate").value.trim() || "Bugün";

            if (!name || isNaN(weight) || isNaN(reps)) return alert("Tüm alanları eksiksiz gir kral!");

            try {
                const res = await fetch("/workouts", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        user_id: currentUser.id,
                        exercise: name,
                        weight: weight,
                        reps: reps,
                        date: date
                    })
                });
                if (!res.ok) throw new Error("Kayıt başarısız");
                
                document.getElementById("exerciseWeight").value = "";
                document.getElementById("exerciseReps").value = "";
                loadUserWorkouts();
            } catch (err) {
                alert(err.message);
            }
        }

        async function deleteWorkout(id) {
            try {
                await fetch(`/workouts/${id}`, { method: "DELETE" });
                loadUserWorkouts();
            } catch (err) {
                console.error(err);
            }
        }

        function populateDropdown() {
            const select = document.getElementById("chartExerciseSelect");
            const currentSelected = select.value;
            const unique = [...new Set(workoutLogs.map(item => item.exercise))];

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

        function renderHistory() {
            const list = document.getElementById("historyList");
            list.innerHTML = "";
            const reversed = [...workoutLogs].reverse();

            if (reversed.length === 0) {
                list.innerHTML = "<div style='color:#6b7280; font-size:0.85rem; text-align:center; padding:15px;'>Henüz set girilmedi.</div>";
                return;
            }

            reversed.forEach(item => {
                list.innerHTML += `
                    <div class="history-item">
                        <div>
                            <span class="ex-name">${item.exercise}</span>: 
                            <span class="ex-stat">${item.weight} kg</span> × ${item.reps} tkr
                            <span style="color:#6b7280; font-size:0.75rem; margin-left:6px;">(${item.date})</span>
                        </div>
                        <button onclick="deleteWorkout(${item.id})">Sil</button>
                    </div>
                `;
            });
        }

        function updateChart() {
            const selectedEx = document.getElementById("chartExerciseSelect").value;
            const filtered = workoutLogs.filter(item => item.exercise === selectedEx);

            const labels = filtered.map(item => item.date);
            const weights = filtered.map(item => item.weight);
            const reps = filtered.map(item => item.reps);

            const ctx = document.getElementById('progressionChart').getContext('2d');
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
                        x: { grid: { color: '#1c212b' }, ticks: { color: '#9ca3af' } },
                        y: { type: 'linear', position: 'left', grid: { color: '#1c212b' }, ticks: { color: '#00f2fe' } },
                        y1: { type: 'linear', position: 'right', grid: { display: false }, ticks: { color: '#f59e0b' } }
                    },
                    plugins: { legend: { labels: { color: '#e5e7eb', font: { size: 11, weight: 'bold' } } } }
                }
            });
        }

        // CHAT & VISION
        let conversationHistory = [];
        let selectedBase64Image = null;

        function handleImageSelect(event) {
            const file = event.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = function(e) {
                selectedBase64Image = e.target.result;
                document.getElementById("imagePreview").src = selectedBase64Image;
                document.getElementById("previewContainer").style.display = "flex";
            };
            reader.readAsDataURL(file);
        }

        function clearImage() {
            selectedBase64Image = null;
            document.getElementById("imageInput").value = "";
            document.getElementById("previewContainer").style.display = "none";
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

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        user_id: currentUser ? currentUser.id : null,
                        user_message: currentText,
                        image_base64: currentImg,
                        history: conversationHistory
                    })
                });
                
                const data = await response.json();
                let replyFormatted = data.coach_reply.replace(/\\n/g, "<br>").replace(/\\*\\*(.*?)\\*\\*/g, "<b>$1</b>");
                document.getElementById(loadingId).innerHTML = replyFormatted;

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

        function handleKey(e) { if (e.key === "Enter") sendMessage(); }

        checkAuth();
    </script>
</body>
</html>
"""

# ----------------- BACKEND ENDPOINTS -----------------

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTML_INTERFACE

@app.post("/register")
def register(auth: AuthInput):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (auth.username, hash_pw(auth.password)))
        conn.commit()
        user_id = c.lastrowid
        return {"user_id": user_id, "username": auth.username}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Bu kullanıcı adı zaten alınmış!")
    finally:
        conn.close()

@app.post("/login")
def login(auth: AuthInput):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, username FROM users WHERE username = ? AND password_hash = ?", (auth.username, hash_pw(auth.password)))
    user = c.fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="Kullanıcı adı veya şifre hatalı!")
    return {"user_id": user[0], "username": user[1]}

@app.get("/workouts/{user_id}")
def get_workouts(user_id: int):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, exercise, weight, reps, date FROM workouts WHERE user_id = ? ORDER BY id ASC", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "exercise": r[1], "weight": r[2], "reps": r[3], "date": r[4]} for r in rows]

@app.post("/workouts")
def add_workout(data: WorkoutLogInput):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO workouts (user_id, exercise, weight, reps, date) VALUES (?, ?, ?, ?, ?)",
              (data.user_id, data.exercise, data.weight, data.reps, data.date))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.delete("/workouts/{workout_id}")
def delete_workout_item(workout_id: int):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM workouts WHERE id = ?", (workout_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}

@app.post("/chat")
def coach_dialogue(data: ChatInput):
    start_time = time.time()
    
    user_context = ""
    if data.user_id:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT exercise, weight, reps FROM workouts WHERE user_id = ? ORDER BY id DESC LIMIT 5", (data.user_id,))
        recent = c.fetchall()
        conn.close()
        if recent:
            user_context = "Kullanıcının Kayıtlı Son Setleri: " + ", ".join([f"{r[0]}: {r[1]}kg x {r[2]}" for r in recent])

    rag_text = ""
    try:
        results = collection.query(query_texts=[data.user_message], n_results=2)
        if results and results.get("documents") and len(results["documents"]) > 0:
            rag_text = "\n".join(results["documents"][0])
    except Exception:
        rag_text = "Hipertrofi ve progressive overload prensipleri."

    system_prompt = f"""
Sen elit seviyede bir 'Looksmaxxing, Hipertrofi & Fizik Koçu'sun.
KULLANICI VERİTABANI:
{user_context}

KAYNAK DOKÜMAN:
{rag_text}

1. SET / PROGRESSIVE OVERLOAD DEĞERLENDİRMESİ:
- Kullanıcının veritabanındaki son ağırlıklarına bakarak bir sonraki antrenmanda hedeflemesi gereken net kg ve tekrarı söyle.
2. FOTOĞRAF ANALİZİ (Varsa):
- Yemekse: Gramaj, kalori, makro çıkar.
- Fizikse: Tahmini yağ oranı ve eksik bölgeleri listele.
"""

    is_vision = bool(data.image_base64)
    messages = [{"role": "system", "content": system_prompt}]
    for msg in data.history:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    if is_vision:
        user_content = [
            {"type": "text", "text": data.user_message},
            {"type": "image_url", "image_url": {"url": data.image_base64}}
        ]
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": data.user_message})

    reply_text = None
    try:
        all_models = client.models.list()
        available_models = [
            m.id for m in all_models.data 
            if not any(x in m.id.lower() for x in ["whisper", "guard", "orpheus", "allam"])
        ]
    except Exception:
        available_models = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "groq/compound"]

    for model_name in available_models:
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0.4,
                max_tokens=500,
            )
            reply_text = chat_completion.choices[0].message.content
            break
        except Exception:
            continue

    if not reply_text:
        reply_text = "Analiz motoru şu anda yanıt veremedi kral."

    elapsed = round(time.time() - start_time, 2)
    print(f"--> [LOG] Yanıt üretildi: {elapsed}sn")

    return {
        "user_message": data.user_message,
        "coach_reply": reply_text
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
