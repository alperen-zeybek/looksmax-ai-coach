import os
import time
from typing import List, Dict
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chromadb
from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6KIYed6AADdAS02s2I9uTV6O_WpLz8-3A5ys5GlCkTKWw")
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Looksmaxxing & Physique AI Coach with Memory")

# ChromaDB Bağlantısı
chroma_client = chromadb.PersistentClient(path="./looksmax_db")
collection = chroma_client.get_or_create_collection(name="looksmax_knowledge")

# Veri Modeli: Mesaj + Konuşma Geçmişi
class ChatMessage(BaseModel):
    role: str
    text: str

class ChatInput(BaseModel):
    user_message: str
    history: List[ChatMessage] = []

def get_embedding(text: str):
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return response.embeddings[0].values

HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Looksmax AI Coach</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background-color: #0d0f12; color: #e5e7eb; display: flex; justify-content: center; height: 100vh; }
        .chat-container { width: 100%; max-width: 800px; display: flex; flex-direction: column; height: 100vh; border-left: 1px solid #1f242d; border-right: 1px solid #1f242d; background: #13161c; }
        .chat-header { padding: 18px 24px; border-bottom: 1px solid #1f242d; background: #0f1217; display: flex; align-items: center; justify-content: space-between; }
        .chat-header h1 { font-size: 1.15rem; font-weight: 700; color: #00f2fe; letter-spacing: 0.5px; }
        .chat-header span { font-size: 0.8rem; background: #1f2937; padding: 4px 10px; border-radius: 12px; color: #10b981; font-weight: 600; }
        .messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
        .msg { max-width: 80%; padding: 14px 18px; border-radius: 14px; font-size: 0.95rem; line-height: 1.5; word-wrap: break-word; }
        .msg.user { align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 4px; }
        .msg.coach { align-self: flex-start; background: #1a1f29; border: 1px solid #283040; border-bottom-left-radius: 4px; }
        .input-area { padding: 16px 20px; border-top: 1px solid #1f242d; background: #0f1217; display: flex; gap: 12px; }
        input[type="text"] { flex: 1; background: #1a1f29; border: 1px solid #283040; color: #fff; padding: 14px 18px; border-radius: 10px; font-size: 0.95rem; outline: none; transition: 0.2s; }
        input[type="text"]:focus { border-color: #00f2fe; }
        button { background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); color: #000; border: none; font-weight: 700; padding: 0 24px; border-radius: 10px; cursor: pointer; transition: 0.2s; }
        button:hover { opacity: 0.9; transform: scale(1.02); }
        button:disabled { opacity: 0.4; cursor: not-allowed; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>⚡ LOOKSMAX & PHYSIQUE AI COACH</h1>
            <span>● Hafıza Aktif</span>
        </div>
        <div class="messages" id="chatBox">
            <div class="msg coach">Selam kral! Antrenman, progressive overload, beslenme veya vücut optimizasyonunu takip ediyorum. Ağırlıklarını veya hedeflerini yaz, konuşmayı aklımda tutarak koçluk yapayım.</div>
        </div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Mesajını yaz... (örn: Bugün benchte 90x8 attım)" onkeypress="handleKey(event)" />
            <button id="sendBtn" onclick="sendMessage()">Gönder</button>
        </div>
    </div>

    <script>
        // Konuşma geçmişini tarayıcı tarafında tutuyoruz
        let conversationHistory = [];

        async function sendMessage() {
            const input = document.getElementById("userInput");
            const btn = document.getElementById("sendBtn");
            const chatBox = document.getElementById("chatBox");
            const text = input.value.trim();
            if (!text) return;

            // Kullanıcı mesajını ekrana bas ve listeye ekle
            chatBox.innerHTML += `<div class="msg user">${text}</div>`;
            input.value = "";
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
                        user_message: text,
                        history: conversationHistory
                    })
                });
                const data = await response.json();
                
                let replyFormatted = data.coach_reply.replace(/\\n/g, "<br>").replace(/\\*\\*(.*?)\\*\\*/g, "<b>$1</b>");
                document.getElementById(loadingId).innerHTML = replyFormatted;

                // Konuşma geçmişine ekle (Hem kullanıcıyı hem koçu)
                conversationHistory.push({ role: "user", text: text });
                conversationHistory.push({ role: "model", text: data.coach_reply });

                // Çok uzarsa hafızayı son 10 mesajla sınırla
                if (conversationHistory.length > 10) {
                    conversationHistory = conversationHistory.slice(-10);
                }
            } catch (err) {
                document.getElementById(loadingId).innerText = "Hata oluştu, tekrar dene.";
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
    
    # 1. Vektör Tabanında Arama Yap
    context_text = ""
    try:
        query_vector = get_embedding(data.user_message)
        results = collection.query(query_embeddings=[query_vector], n_results=2)
        if results and results.get("documents"):
            context_text = "\n".join(results["documents"][0])
    except Exception:
        context_text = "Hipertrofi ve progressive overload prensipleri."

    # 2. Geçmiş Konuşmaları Metne Çevir
    history_formatted = ""
    for msg in data.history:
        role_label = "Kullanıcı" if msg.role == "user" else "Koç"
        history_formatted += f"{role_label}: {msg.text}\n"

    # 3. Prompt Hazırla (Rol + Hafıza + PDF Kaynağı)
    system_instruction = f"""
Sen elit seviyede, doğrudan bilime ve hipertrofi prensiplerine dayalı koçluk yapan 'Looksmaxxing & Hipertrofi Koçu'sun.
Yalnızca antrenman, progressive overload, beslenme, hipertrofi ve fiziksel gelişim konularında konuş.

ÖNEMLİ KURAL: Kullanıcının önceki mesajlardaki ağırlıklarını, setlerini ve hedeflerini kesinlikle hatırla. Ona göre progressive overload veya program önerisi sun. Bro-science yapma, net maddelerle konuş.

KAYNAK DOKÜMAN BİLGİSİ:
{context_text}

GEÇMİŞ KONUŞMA AKIŞI:
{history_formatted if history_formatted else "Henüz önceki mesaj yok (Yeni sohbet)."}
"""

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"{system_instruction}\n\nKULLANICININ YENİ MESAJI: {data.user_message}"
    )

    elapsed = round(time.time() - start_time, 2)
    print(f"--> [LOG] Soru: '{data.user_message}' | Süre: {elapsed}sn | Hafıza Boyutu: {len(data.history)} mesaj")

    return {
        "user_message": data.user_message,
        "coach_reply": response.text
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)