"""Retrieval-Augmented Generation pipeline for the DataSense AI knowledge base.

Pipeline: PDF -> Document Loader -> Text Splitter -> Embeddings ->
Vector Database (FAISS) -> Retriever -> relevant context chunks.

FAISS is used instead of Chroma because it is a pure file-based index with
no background sqlite server, which is more stable for Streamlit deployment.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "data_science_knowledge.pdf"
INDEX_DIR = BASE_DIR / "vector_store" / "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RAGError(Exception):
    """Raised when the RAG pipeline cannot be built or queried."""


_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def is_knowledge_base_available() -> bool:
    return PDF_PATH.exists()


def build_vectorstore(pdf_path: Path = PDF_PATH, index_dir: Path = INDEX_DIR):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    if not pdf_path.exists():
        raise RAGError(f"Knowledge base PDF tidak ditemukan: {pdf_path.name}")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    if not documents:
        raise RAGError("Knowledge base PDF tidak berisi teks yang dapat diproses.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    return vectorstore


def load_or_build_vectorstore(pdf_path: Path = PDF_PATH, index_dir: Path = INDEX_DIR):
    from langchain_community.vectorstores import FAISS

    if index_dir.exists():
        try:
            return FAISS.load_local(
                str(index_dir), get_embeddings(), allow_dangerous_deserialization=True
            )
        except Exception:
            pass  # fall through and rebuild a fresh index
    return build_vectorstore(pdf_path, index_dir)


def retrieve_context(vectorstore, query: str, k: int = 3) -> str:
    """Return the top-k relevant knowledge base chunks formatted as context text."""
    if vectorstore is None or not query.strip():
        return ""
    try:
        docs = vectorstore.similarity_search(query, k=k)
    except Exception as exc:
        raise RAGError(f"Gagal melakukan pencarian pada knowledge base: {exc}") from exc

    if not docs:
        return ""

    chunks = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page")
        page_label = f" (halaman {page + 1})" if isinstance(page, int) else ""
        chunks.append(f"[Referensi {i}{page_label}]\n{doc.page_content.strip()}")
    return "\n\n".join(chunks)
