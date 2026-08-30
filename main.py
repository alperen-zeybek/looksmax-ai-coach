import os
import time
from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chromadb
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Looksmaxxing & Physique AI Coach with Dynamic Tracking")

# ChromaDB Bağlantısı
chroma_client = chromadb.PersistentClient(path="./looksmax_db")
collection = chroma_client.get_or_create_collection(name="looksmax_knowledge")

class ChatInput(BaseModel):
    user_message: str
    image_base64: Optional[str] = None
    history: List[dict] = []

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
        body { background-color: #0b0d10; color: #e5e7eb; display: flex; justify-content: center; height: 100vh; overflow: hidden; }
        
        /* Ana Layout */
        .app-layout { display: flex; width: 100%; max-width: 1400px; height: 100vh; border-left: 1px solid #1c212b; border-right: 1px solid #1c212b; }
        
        /* Sol Panel: Tracker & Grafik */
        .tracker-panel { width: 45%; background: #10141b; border-right: 1px solid #1c212b; display: flex; flex-direction: column; padding: 20px; gap: 16px; overflow-y: auto; }
        .tracker-title { font-size: 1.1rem; font-weight: 700; color: #00f2fe; display: flex; align-items: center; justify-content: space-between; }
        
        .tracker-form { background: #161b24; padding: 14px; border-radius: 12px; border: 1px solid #232a38; display: flex; flex-direction: column; gap: 10px; }
        .form-row { display: flex; gap: 8px; }
        .tracker-form input, .tracker-form select { background: #0b0d10; border: 1px solid #2b3547; color: #fff; padding: 10px 12px; border-radius: 8px; font-size: 0.85rem; outline: none; }
        .tracker-form input:focus, .tracker-form select:focus { border-color: #00f2fe; }
        .btn-add { background: #00f2fe; color: #000; border: none; font-weight: 700; padding: 10px; border-radius: 8px; cursor: pointer; transition: 0.2s; }
        .btn-add:hover { opacity: 0.9; }

        .chart-box { background: #161b24; padding: 16px; border-radius: 12px; border: 1px solid #232a38; flex: 1; min-height: 260px; display: flex; flex-direction: column; }
        .chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .chart-header select { background: #0b0d10; border: 1px solid #2b3547; color: #00f2fe; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; outline: none; }

        .history-list { background: #161b24; padding: 12px; border-radius: 12px; border: 1px solid #232a38; max-height: 180px; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
        .history-item { display: flex; justify-content: space-between; background: #0b0d10; padding: 8px 12px; border-radius: 6px; font-size: 0.8rem; border: 1px solid #1c212b; }
        .history-item span.highlight { color: #00f2fe; font-weight: 600; }
        .history-item button { background: transparent; border: none; color: #ef4444; cursor: pointer; font-size: 0.8rem; }

        /* Sağ Panel: Chat */
        .chat-container { width: 55%; display: flex; flex-direction: column; height: 100vh; background: #131720; }
        .chat-header { padding: 18px 24px; border-bottom: 1px solid #1c212b; background: #0e1117; display: flex; align-items: center; justify-content: space-between; }
        .chat-header h1 { font-size: 1.1rem; font-weight: 700; color: #00f2fe; }
        .chat-header span { font-size: 0.75rem; background: #1c2433; padding: 4px 10px; border-radius: 12px; color: #10b981; font-weight: 600; }
        
        .messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
        .msg { max-width: 82%; padding: 13px 16px; border-radius: 12px; font-size: 0.9rem; line-height: 1.5; word-wrap: break-word; }
        .msg.user { align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 4px; }
        .msg.coach { align-self: flex-start; background: #1a202c; border: 1px solid #283347; border-bottom-left-radius: 4px; }
        .msg img.preview-img { max-width: 200px; border-radius: 8px; margin-bottom: 8px; display: block; }
        
        .preview-container { display: none; padding: 8px 20px 0; background: #0e1117; align-items: center; gap: 10px; }
        .preview-container img { height: 50px; border-radius: 6px; border: 1px solid #00f2fe; }
        .preview-container button { background: #ef4444; color: white; border: none; border-radius: 50%; width: 20px; height: 20px; cursor: pointer; font-size: 11px; }

        .input-area { padding: 14px 20px; border-top: 1px solid #1c212b; background: #0e1117; display: flex; gap: 10px; align-items: center; }
        input[type="text"].chat-input { flex: 1; background: #1a202c; border: 1px solid #283347; color: #fff; padding: 12px 16px; border-radius: 10px; font-size: 0.9rem; outline: none; }
        input[type="text"].chat-input:focus { border-color: #00f2fe; }
        
        .file-label { background: #1a202c; border: 1px solid #283347; color: #00f2fe; padding: 10px 14px; border-radius: 10px; cursor: pointer; font-size: 1rem; }
        .file-label:hover { border-color: #00f2fe; }
        input[type="file"] { display: none; }

        .send-btn { background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); color: #000; border: none; font-weight: 700; padding: 12px 20px; border-radius: 10px; cursor: pointer; }
        .send-btn:hover { opacity: 0.9; }

        @media (max-width: 900px) {
            body { overflow: auto; height: auto; }
            .app-layout { flex-direction: column; height: auto; }
            .tracker-panel, .chat-container { width: 100%; height: auto; }
            .chat-container { height: 80vh; }
        }
    </style>
</head>
<body>
    <div class="app-layout">
        <!-- SOL PANEL: PROGRESSIVE OVERLOAD TRACKER -->
        <div class="tracker-panel">
            <div class="tracker-title">
                <span>📈 OVERLOAD & SET TRACKER</span>
            </div>

            <!-- Kayıt Formu -->
            <div class="tracker-form">
                <input type="text" id="exerciseName" placeholder="Egzersiz Adı (örn: Incline DB Press, Bench, Squat)" list="exerciseList" />
                <datalist id="exerciseList">
                    <option value="Bench Press">
                    <option value="Incline Dumbbell Press">
                    <option value="Squat">
                    <option value="Deadlift">
                    <option value="Barbell Row">
                    <option value="Overhead Press">
                    <option value="Lateral Raise">
                    <option value="Vücut Ağırlığı (Kilo)">
                </datalist>
                <div class="form-row">
                    <input type="number" id="exerciseWeight" placeholder="Ağırlık (kg)" step="0.5" style="flex:1;" />
                    <input type="number" id="exerciseReps" placeholder="Tekrar" style="flex:1;" />
                    <input type="text" id="exerciseDate" placeholder="Tarih" style="flex:1.2;" />
                </div>
                <button class="btn-add" onclick="addWorkoutLog()">Seti Kaydet & Grafiğe Ekle</button>
            </div>

            <!-- Grafik Alanı -->
            <div class="chart-box">
                <div class="chart-header">
                    <span style="font-size:0.85rem; font-weight:600; color:#9ca3af;">Gelişim Grafiği</span>
                    <select id="chartExerciseSelect" onchange="updateChart()">
                        <!-- Dinamik doldurulacak -->
                    </select>
                </div>
                <div style="flex:1; position:relative; min-height:180px;">
                    <canvas id="progressionChart"></canvas>
                </div>
            </div>

            <!-- Son Eklenen Kayıtlar -->
            <span style="font-size:0.85rem; font-weight:600; color:#9ca3af; margin-top:4px;">Son Kaydedilen Setler</span>
            <div class="history-list" id="historyList">
                <!-- Dinamik liste -->
            </div>
        </div>

        <!-- SAĞ PANEL: AI CHAT & VISION -->
        <div class="chat-container">
            <div class="chat-header">
                <h1>⚡ LOOKSMAX AI COACH</h1>
                <span>● Vision & Hafıza Aktif</span>
            </div>
            <div class="messages" id="chatBox">
                <div class="msg coach">Selam kral! Antrenman, progressive overload ve beslenmeni takip ediyorum. Sol taraftan girdiğin tüm serbest egzersizleri (kilo & tekrar) grafik üzerinden inceleyip sana özel periyotlama yapabilirim.</div>
            </div>

            <div class="preview-container" id="previewContainer">
                <img id="imagePreview" src="" alt="Görsel" />
                <button onclick="clearImage()">✕</button>
                <span style="font-size:0.75rem; color:#9ca3af;">Görsel eklendi</span>
            </div>

            <div class="input-area">
                <label class="file-label" for="imageInput" title="Fotoğraf Yükle">📷</label>
                <input type="file" id="imageInput" accept="image/*" onchange="handleImageSelect(event)" />
                <input type="text" class="chat-input" id="userInput" placeholder="Koça soru sor veya tavsiye iste..." onkeypress="handleKey(event)" />
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">Gönder</button>
            </div>
        </div>
    </div>

    <script>
        // ----------------- TRACKER & CHART.JS MANTIĞI -----------------
        let workoutData = JSON.parse(localStorage.getItem("workout_logs") || "[]");
        let chartInstance = null;

        // Varsayılan tarihi bugüne ayarla
        document.getElementById("exerciseDate").value = new Date().toLocaleDateString('tr-TR', { day: '2-digit', month: '2-digit' });

        function saveAndRender() {
            localStorage.setItem("workout_logs", JSON.stringify(workoutData));
            populateExerciseDropdown();
            renderHistoryList();
            updateChart();
        }

        function addWorkoutLog() {
            const name = document.getElementById("exerciseName").value.trim();
            const weight = parseFloat(document.getElementById("exerciseWeight").value);
            const reps = parseInt(document.getElementById("exerciseReps").value);
            const date = document.getElementById("exerciseDate").value.trim() || "Bugün";

            if (!name || isNaN(weight) || isNaN(reps)) {
                alert("Lütfen egzersiz adı, kilo ve tekrar alanlarını eksiksiz doldur kral!");
                return;
            }

            workoutData.push({
                id: Date.now(),
                exercise: name,
                weight: weight,
                reps: reps,
                date: date
            });

            document.getElementById("exerciseWeight").value = "";
            document.getElementById("exerciseReps").value = "";
            saveAndRender();
        }

        function deleteLog(id) {
            workoutData = workoutData.filter(item => item.id !== id);
            saveAndRender();
        }

        function populateExerciseDropdown() {
            const select = document.getElementById("chartExerciseSelect");
            const currentSelected = select.value;
            const uniqueExercises = [...new Set(workoutData.map(item => item.exercise))];

            select.innerHTML = "";
            if (uniqueExercises.length === 0) {
                select.innerHTML = "<option value=''>Kayıt Yok</option>";
                return;
            }

            uniqueExercises.forEach(ex => {
                const opt = document.createElement("option");
                opt.value = ex;
                opt.innerText = ex;
                select.appendChild(opt);
            });

            if (uniqueExercises.includes(currentSelected)) {
                select.value = currentSelected;
            } else {
                select.value = uniqueExercises[0];
            }
        }

        function renderHistoryList() {
            const list = document.getElementById("historyList");
            list.innerHTML = "";
            const reversed = [...workoutData].reverse().slice(0, 10);

            if (reversed.length === 0) {
                list.innerHTML = "<div style='color:#6b7280; font-size:0.8rem; text-align:center;'>Henüz set kaydedilmedi.</div>";
                return;
            }

            reversed.forEach(item => {
                list.innerHTML += `
                    <div class="history-item">
                        <div>
                            <b>${item.exercise}</b>: <span class="highlight">${item.weight} kg</span> × ${item.reps} tekrar
                            <span style="color:#6b7280; margin-left:6px;">(${item.date})</span>
                        </div>
                        <button onclick="deleteLog(${item.id})">Sil</button>
                    </div>
                `;
            });
        }

        function updateChart() {
            const selectedEx = document.getElementById("chartExerciseSelect").value;
            const filtered = workoutData.filter(item => item.exercise === selectedEx);

            const labels = filtered.map(item => item.date);
            const weights = filtered.map(item => item.weight);
            const reps = filtered.map(item => item.reps);

            const ctx = document.getElementById('progressionChart').getContext('2d');

            if (chartInstance) {
                chartInstance.destroy();
            }

            chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Ağırlık (kg)',
                            data: weights,
                            borderColor: '#00f2fe',
                            backgroundColor: 'rgba(0, 242, 254, 0.1)',
                            borderWidth: 2,
                            yAxisID: 'y',
                            tension: 0.3,
                            fill: true,
                            pointRadius: 4
                        },
                        {
                            label: 'Tekrar',
                            data: reps,
                            borderColor: '#f59e0b',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            borderDash: [4, 4],
                            yAxisID: 'y1',
                            tension: 0.3,
                            pointRadius: 4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            grid: { color: '#1c212b' },
                            ticks: { color: '#9ca3af', font: { size: 10 } }
                        },
                        y: {
                            type: 'linear',
                            position: 'left',
                            grid: { color: '#1c212b' },
                            ticks: { color: '#00f2fe', font: { size: 10 } },
                            title: { display: true, text: 'kg', color: '#00f2fe' }
                        },
                        y1: {
                            type: 'linear',
                            position: 'right',
                            grid: { display: false },
                            ticks: { color: '#f59e0b', font: { size: 10 } },
                            title: { display: true, text: 'tekrar', color: '#f59e0b' }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#e5e7eb', font: { size: 11 } }
                        }
                    }
                }
            });
        }

        // Başlangıçta render et
        saveAndRender();

        // ----------------- CHAT & VISION MANTIĞI -----------------
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
            if (selectedBase64Image) {
                userHtml += `<img src="${selectedBase64Image}" class="preview-img" />`;
            }
            userHtml += `<span>${text || "Fotoğraf analizi talebi"}</span>`;

            chatBox.innerHTML += `<div class="msg user">${userHtml}</div>`;
            const currentImg = selectedBase64Image;
            const currentText = text || "Lütfen bu fotoğrafı detaylı analiz et.";

            input.value = "";
            clearImage();
            btn.disabled = true;
            chatBox.scrollTop = chatBox.scrollHeight;

            const loadingId = "load-" + Date.now();
            chatBox.innerHTML += `<div class="msg coach" id="${loadingId}"><i>Analiz ediliyor...</i></div>`;
            chatBox.scrollTop = chatBox.scrollHeight;

            // Son kaydedilen antrenman verilerini de bota bağlayalım
            const lastLogsSummary = workoutData.slice(-5).map(w => `${w.exercise}: ${w.weight}kg x ${w.reps}`).join(", ");
            const enrichedMessage = lastLogsSummary ? `[Kullanıcının Son Set Kayıtları: ${lastLogsSummary}] - Soru: ${currentText}` : currentText;

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        user_message: enrichedMessage,
                        image_base64: currentImg,
                        history: conversationHistory
                    })
                });
                
                if (!response.ok) throw new Error("Server error: " + response.status);

                const data = await response.json();
                let replyFormatted = data.coach_reply.replace(/\\n/g, "<br>").replace(/\\*\\*(.*?)\\*\\*/g, "<b>$1</b>");
                document.getElementById(loadingId).innerHTML = replyFormatted;

                conversationHistory.push({ role: "user", content: currentText });
                conversationHistory.push({ role: "assistant", content: data.coach_reply });

                if (conversationHistory.length > 8) conversationHistory = conversationHistory.slice(-8);
            } catch (err) {
                console.error(err);
                document.getElementById(loadingId).innerText = "Hata oluştu. Lütfen tekrar dene.";
            } finally {
                btn.disabled = false;
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        }

        function handleKey(e) {
            if (e.key === "Enter") sendMessage();
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTML_INTERFACE

@app.post("/chat")
def coach_dialogue(data: ChatInput):
    start_time = time.time()
    
    context_text = ""
    try:
        results = collection.query(query_texts=[data.user_message], n_results=2)
        if results and results.get("documents") and len(results["documents"]) > 0:
            context_text = "\n".join(results["documents"][0])
    except Exception:
        context_text = "Hipertrofi, progressive overload ve beslenme prensipleri."

    system_prompt = f"""
Sen elit seviyede bir 'Looksmaxxing, Hipertrofi & Fizik Koçu'sun.
Kullanıcının girdiği kilo, tekrar ve progressive overload verilerini titizlikle değerlendir.

1. EGZERSİZ / SET ANALİZİ:
- Kullanıcının son setlerine ve gelişimine göre sonraki antrenman için hedef ağırlık/tekrar öner.
- RPE, tükenişe yakınlık (RIR) ve form ipuçları ver.

2. FOTOĞRAF ANALİZİ (Varsa):
- Yemekse: Gramaj, kalori ve makro ayrımı yap.
- Fizikse: Tahmini yağ oranı ve eksik kas gruplarını belirt.

KAYNAK DOKÜMAN:
{context_text}
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
    print(f"--> [LOG] İşlem tamam: {elapsed}sn")

    return {
        "user_message": data.user_message,
        "coach_reply": reply_text
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)