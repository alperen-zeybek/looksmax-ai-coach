# 1. Temel işletim sistemi ve Python ortamı olarak hafif Linux imajı seçiyoruz
FROM python:3.10-slim

# 2. Konteyner içinde çalışacağımız klasörü belirliyoruz
WORKDIR /app

# 3. Bağımlılık listesini kopyalayıp paketleri yüklüyoruz
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Proje kodlarımızı konteyner içine kopyalıyoruz
COPY main.py .

# 5. Dış dünyaya açılacak portu belirtiyoruz
EXPOSE 8000

# 6. Konteyner çalıştığında API sunucusunu ayağa kaldıran komut
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]