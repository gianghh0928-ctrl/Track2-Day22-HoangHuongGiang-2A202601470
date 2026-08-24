"""
Tiện ích để tải và xử lý dữ liệu cho RAG pipeline.

Cách dùng:
    from utils.data_loader import load_knowledge_base, split_text, build_vectorstore

    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
    vectorstore = build_vectorstore(chunks, embeddings)
"""
from pathlib import Path


def load_knowledge_base(path: str = None) -> str:
    """
    Đọc file knowledge base và trả về nội dung dạng chuỗi.

    Args:
        path: đường dẫn tới file text.
              Mặc định: data/knowledge_base.txt (thư mục gốc của project)

    Returns:
        Nội dung file dưới dạng str
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.txt"
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Chia văn bản thành các đoạn nhỏ (chunks) để index.

    Dùng RecursiveCharacterTextSplitter — tách ưu tiên theo đoạn văn, câu, rồi ký tự.

    Args:
        text         : văn bản cần chia
        chunk_size   : số ký tự tối đa mỗi chunk (mặc định: 500)
        chunk_overlap: số ký tự chồng lên nhau giữa 2 chunks liên tiếp (mặc định: 50)

    Returns:
        list[str] — danh sách các chuỗi chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks: list, embeddings):
    """
    Tạo FAISS vectorstore từ danh sách chunks và embeddings.
    Có tích hợp lưu cache đĩa và chia batch để tránh dính rate limit Google Gemini.
    """
    import time
    from langchain_community.vectorstores import FAISS

    index_dir = Path(__file__).parent.parent.parent / "data" / "faiss_index"
    if (index_dir / "index.faiss").exists():
        print(f"📦 Đang nạp FAISS index từ cache tại {index_dir} ...")
        try:
            vectorstore = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
            print("✅ Đã nạp FAISS index thành công từ cache.")
            return vectorstore
        except Exception as e:
            print(f"⚠️ Nạp cache thất bại ({e}), tiến hành tạo lại index...")

    print(f"🔨 Đang tạo FAISS index từ {len(chunks)} chunks...")
    batch_size = 30
    vectorstore = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"   Indexing chunks {i+1} đến {min(i+batch_size, len(chunks))} ...")
        success = False
        for attempt in range(6):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_texts(batch, embeddings)
                else:
                    vectorstore.add_texts(batch)
                success = True
                break
            except Exception as e:
                print(f"   ⚠️ Rate limit / Lỗi API: {e}. Đang chờ 15 giây (Thử lại {attempt+1}/6)...")
                time.sleep(15)
        if not success:
            raise RuntimeError("Không thể tạo vectorstore do lỗi API kéo dài.")
        if i + batch_size < len(chunks):
            time.sleep(3)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"✅ FAISS vectorstore đã sẵn sàng và được lưu tại {index_dir}.")
    return vectorstore
