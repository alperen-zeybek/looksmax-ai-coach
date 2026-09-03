FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Knowledge base'i (varsa ./knowledge_base icindeki .txt/.pdf) build aninda
# vektor veritabanina (./looksmax_db) gomer. Render'da persistent disk
# olmadigi icin bu adim, deploy edilen her image'in kendi RAG hafizasiyla
# gelmesini saglar. knowledge_base/ icine yeni PDF eklendiginde, image'in
# yeniden build edilmesi (yeni bir git push / manuel deploy) gerekir.
RUN python ingest.py || echo "Knowledge base bos veya ingest atlandi, RAG devre disi kalacak."

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
