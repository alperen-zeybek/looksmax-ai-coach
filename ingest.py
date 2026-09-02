import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

KNOWLEDGE_DIR = "./knowledge_base"
DB_DIR = "./looksmax_db"

# knowledge_base klasoru yoksa olustur (Docker build'de patlamasin diye)
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

print("1. Knowledge Base taranıyor (.txt ve .pdf dosyaları)...")

all_documents = []

try:
    txt_loader = DirectoryLoader(KNOWLEDGE_DIR, glob="*.txt", loader_cls=TextLoader)
    all_documents += txt_loader.load()
except Exception as e:
    print(f"  ! TXT yukleme hatasi (atlaniyor): {e}")

try:
    pdf_loader = DirectoryLoader(KNOWLEDGE_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
    all_documents += pdf_loader.load()
except Exception as e:
    print(f"  ! PDF yukleme hatasi (atlaniyor): {e}")

if not all_documents:
    print(
        f"⚠️  '{KNOWLEDGE_DIR}' klasorunde hicbir .txt/.pdf bulunamadi. "
        f"Bos bir vektor veritabani olusturulmayacak, RAG devre disi kalacak. "
        f"Dosya ekleyip bu script'i tekrar calistirin (veya image'i yeniden build edin)."
    )
    raise SystemExit(0)

print(f"Toplam {len(all_documents)} doküman sayfası/metni bulundu. Parçalanıyor...")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = text_splitter.split_documents(all_documents)
print(f"Toplam {len(chunks)} adet vektör parçası (chunk) oluşturuldu.")

print("2. Hafıza güncelleniyor ve ChromaDB'ye kaydediliyor...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=DB_DIR
)

print("✅ EĞİTİM TAMAMLANDI! Tüm PDF ve metinler başarıyla hafızaya gömüldü.")
