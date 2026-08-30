import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("1. Knowledge Base taranıyor (.txt ve .pdf dosyaları)...")

# TXT dosyalarını yükle
txt_loader = DirectoryLoader('./knowledge_base', glob="*.txt", loader_cls=TextLoader)
txt_docs = txt_loader.load()

# PDF dosyalarını yükle
pdf_loader = DirectoryLoader('./knowledge_base', glob="*.pdf", loader_cls=PyPDFLoader)
pdf_docs = pdf_loader.load()

all_documents = txt_docs + pdf_docs

print(f"Toplam {len(all_documents)} doküman sayfası/metni bulundu. Parçalanıyor...")

# Metinleri bilimsel bağlamı koparmayacak parçalara bölüyoruz
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = text_splitter.split_documents(all_documents)

print(f"Toplam {len(chunks)} adet vektör parçası (chunk) oluşturuldu.")
print("2. Hafıza güncelleniyor ve ChromaDB'ye kaydediliyor...")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Veritabanını sıfırdan en güncel dokümanlarla oluşturuyoruz
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./looksmax_db"
)

print("✅ EĞİTİM TAMAMLANDI! Tüm PDF ve metinler başarıyla hafızaya gömüldü.")