import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from google import genai

# API anahtarını ortam değişkeninden veya doğrudan değişkenden al
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6KIYed6AADdAS02s2I9uTV6O_WpLz8-3A5ys5GlCkTKWw")
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Looksmaxxing & Physique AI Coach")

# Vektör hafızası
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(
    persist_directory="./looksmax_db", 
    embedding_function=embedding_model
)

class ChatInput(BaseModel):
    user_message: str

# Dark Mode Chat Arayüzü
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
        .chat-header span { font-size: 0.8rem; background: #1f2937; padding: 4px 10px; border-radius: 12px; color: #9ca3af; }
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
            <span>RAG + Bilimsel Tabanlı</span>
        </div>
        <div class="messages" id="chatBox">
            <div class="msg coach">Selam kral! V-Taper fiziği, hipertrofi, çene hattı veya cilt/bakım optimizasyonu hakkında neyi öğrenmek istiyorsun? Sor, bilimsel protokollere göre yanıtlayayım.</div>
        </div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Sorunu sor... (örn: Omuz genişliği için en iyi hareket ne?)" onkeypress="handleKey(event)" />
            <button id="sendBtn" onclick="sendMessage()">Gönder</button>
        </div>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById("userInput");
            const btn = document.getElementById("sendBtn");
            const chatBox = document.getElementById("chatBox");
            const text = input.value.trim();
            if (!text) return;

            chatBox.innerHTML += `<div class="msg user">${text}</div>`;
            input.value = "";
            btn.disabled = true;
            chatBox.scrollTop = chatBox.scrollHeight;

            const loadingId = "load-" + Date.now();
            chatBox.innerHTML += `<div class="msg coach" id="${loadingId}"><i>Bilimsel veritabanı taranıyor & analiz ediliyor...</i></div>`;
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ user_message: text })
                });
                const data = await response.json();
                let replyFormatted = data.coach_reply.replace(/\\n/g, "<br>").replace(/\\*\\*(.*?)\\*\\*/g, "<b>$1</b>");
                document.getElementById(loadingId).innerHTML = replyFormatted;
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
    docs = vector_db.similarity_search(data.user_message, k=2)
    context_text = "\n".join([doc.page_content for doc in docs])
    
    system_instruction = f"""
    Sen elit seviyede, doğrudan kanıta ve bilime dayalı tavsiye veren bir 'Looksmaxxing & Hipertrofi Koçu'sun.
    Aşağıdaki bilimsel protokol metnini referans alarak kullanıcının sorusuna net, motive edici ve doğrudan uygulanabilir maddelerle cevap ver.
    Cevaplarında bro-science kullanma, bilimsel temele dayan. Samimi, enerjik ve koç edasıyla Türkçe cevap ver.
    
    BİLGİ BANKASI VERİSİ:
    {context_text}
    """
    
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=f"{system_instruction}\n\nKULLANICI SORUSU: {data.user_message}"
    )
    
    return {
        "user_message": data.user_message,
        "coach_reply": response.text,
        "referenced_context": [doc.page_content for doc in docs]
    }