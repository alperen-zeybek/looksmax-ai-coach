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

app = FastAPI(title="Looksmaxxing & Physique AI Coach with Vision")

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
    <title>Looksmax AI Coach & Vision</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background-color: #0d0f12; color: #e5e7eb; display: flex; justify-content: center; height: 100vh; }
        .chat-container { width: 100%; max-width: 850px; display: flex; flex-direction: column; height: 100vh; border-left: 1px solid #1f242d; border-right: 1px solid #1f242d; background: #13161c; }
        .chat-header { padding: 18px 24px; border-bottom: 1px solid #1f242d; background: #0f1217; display: flex; align-items: center; justify-content: space-between; }
        .chat-header h1 { font-size: 1.15rem; font-weight: 700; color: #00f2fe; letter-spacing: 0.5px; }
        .chat-header span { font-size: 0.8rem; background: #1f2937; padding: 4px 10px; border-radius: 12px; color: #10b981; font-weight: 600; }
        .messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
        .msg { max-width: 80%; padding: 14px 18px; border-radius: 14px; font-size: 0.95rem; line-height: 1.5; word-wrap: break-word; }
        .msg.user { align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 4px; }
        .msg.coach { align-self: flex-start; background: #1a1f29; border: 1px solid #283040; border-bottom-left-radius: 4px; }
        .msg img.preview-img { max-width: 220px; border-radius: 8px; margin-bottom: 8px; display: block; border: 1px solid #3b82f6; }
        
        .preview-container { display: none; padding: 10px 20px 0; background: #0f1217; align-items: center; gap: 10px; }
        .preview-container img { height: 60px; border-radius: 8px; border: 1px solid #00f2fe; }
        .preview-container button { background: #ef4444; color: white; border: none; border-radius: 50%; width: 22px; height: 22px; cursor: pointer; font-size: 12px; }

        .input-area { padding: 16px 20px; border-top: 1px solid #1f242d; background: #0f1217; display: flex; gap: 10px; align-items: center; }
        input[type="text"] { flex: 1; background: #1a1f29; border: 1px solid #283040; color: #fff; padding: 14px 18px; border-radius: 10px; font-size: 0.95rem; outline: none; transition: 0.2s; }
        input[type="text"]:focus { border-color: #00f2fe; }
        
        .file-label { background: #1a1f29; border: 1px solid #283040; color: #00f2fe; padding: 13px 16px; border-radius: 10px; cursor: pointer; font-weight: bold; font-size: 1.1rem; display: flex; align-items: center; justify-content: center; }
        .file-label:hover { border-color: #00f2fe; background: #202634; }
        input[type="file"] { display: none; }

        .send-btn { background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); color: #000; border: none; font-weight: 700; padding: 14px 24px; border-radius: 10px; cursor: pointer; transition: 0.2s; white-space: nowrap; }
        .send-btn:hover { opacity: 0.9; transform: scale(1.02); }
        .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>⚡ LOOKSMAX COACH & VISION</h1>
            <span>● Görsel & Hafıza Aktif</span>
        </div>
        <div class="messages" id="chatBox">
            <div class="msg coach">Selam kral! Antrenman, progressive overload, beslenme ve fizik optimizasyonunu takip ediyorum.<br><br>📸 <b>Yemek fotoğrafı atarsan:</b> Kalori ve makrolarını (Protein/Karb/Yağ) hesaplarım.<br>💪 <b>Fizik fotoğrafı atarsan:</b> Tahmini yağ oranı, eksik kas grupları ve simetri analizi yaparım.</div>
        </div>

        <div class="preview-container" id="previewContainer">
            <img id="imagePreview" src="" alt="Seçilen Görsel" />
            <button onclick="clearImage()">✕</button>
            <span style="font-size:0.8rem; color:#9ca3af;">Görsel eklendi</span>
        </div>

        <div class="input-area">
            <label class="file-label" for="imageInput" title="Fotoğraf Yükle">📷</label>
            <input type="file" id="imageInput" accept="image/*" onchange="handleImageSelect(event)" />
            <input type="text" id="userInput" placeholder="Mesajını veya sorunu yaz... (örn: Bu tabak kaç kalori?)" onkeypress="handleKey(event)" />
            <button class="send-btn" id="sendBtn" onclick="sendMessage()">Gönder</button>
        </div>
    </div>

    <script>
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
            const currentText = text || "Lütfen bu fotoğrafı (yemekse kalori/makro, fizikse yağ oranı ve eksik bölge) detaylı analiz et.";

            input.value = "";
            clearImage();
            btn.disabled = true;
            chatBox.scrollTop = chatBox.scrollHeight;

            const loadingId = "load-" + Date.now();
            chatBox.innerHTML += `<div class="msg coach" id="${loadingId}"><i>Fotoğraf ve veriler taranıyor...</i></div>`;
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        user_message: currentText,
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
                document.getElementById(loadingId).innerText = "Analiz sırasında bir hata oluştu. Lütfen tekrar dene.";
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
    
    # 1. RAG Arama
    context_text = ""
    try:
        results = collection.query(query_texts=[data.user_message], n_results=2)
        if results and results.get("documents") and len(results["documents"]) > 0:
            context_text = "\n".join(results["documents"][0])
    except Exception:
        context_text = "Hipertrofi, beslenme, kalori ve progressive overload kuralları."

    # 2. Sistem Prompt'u
    system_prompt = f"""
Sen elit seviyede bir 'Looksmaxxing, Hipertrofi & Beslenme Koçu'sun.
Kullanıcı sana metin veya fotoğraf gönderebilir:

1. YEMEK/BESLENME FOTOĞRAFI GELİRSE:
- Tabaktaki yiyecekleri tek tek tespit et (örn: ~150g tavuk göğsü, ~200g pirinç pilavı).
- Tahmini Kalori, Protein, Karbonhidrat ve Yağ değerlerini net maddeler halinde yaz.
- Günlük hipertrofi hedefine uygun olup olmadığını değerlendir.

2. FİZİK / FORM FOTOĞRAFI GELİRSE:
- Tahmini Vücut Yağ Oranını (% aralığı) belirt.
- Güçlü ve eksik kalan kas gruplarını listele.
- Vücut simetrisi ve estetiği için doğrudan 2-3 spesifik egzersiz öner.

KAYNAK DOKÜMAN BİLGİSİ:
{context_text}
"""

    is_vision = bool(data.image_base64)
    
    # 3. Mesaj Yapısını Kur
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

    # 4. Canlıdaki Modelleri Doğrudan Sırayla Dene
    reply_text = None
    try:
        all_models = client.models.list()
        # Ses ve güvenlik modellerini ele, gerçek dil modellerini listeye al
        available_models = [
            m.id for m in all_models.data 
            if not any(x in m.id.lower() for x in ["whisper", "guard", "orpheus", "allam"])
        ]
    except Exception:
        available_models = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "groq/compound"]

    print(f"--> [DENENECEK AKTİF MODELLER]: {available_models}")

    for model_name in available_models:
        try:
            print(f"--> [DENENİYOR]: {model_name}")
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0.4,
                max_tokens=500,
            )
            reply_text = chat_completion.choices[0].message.content
            print(f"--> [BAŞARILI]: {model_name} yanıt üretti!")
            break
        except Exception as e:
            print(f"--> [HATA - {model_name}]: {e}")
            continue

    if not reply_text:
        reply_text = "Görsel veya metin analiz edilirken API yanıt veremedi kral."

    elapsed = round(time.time() - start_time, 2)
    print(f"--> [LOG] Soru: '{data.user_message[:30]}' | Süre: {elapsed}sn")

    return {
        "user_message": data.user_message,
        "coach_reply": reply_text
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)