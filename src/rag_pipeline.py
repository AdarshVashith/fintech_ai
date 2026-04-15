from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_DIR = BASE_DIR / "data" / "policies"
DEFAULT_INDEX_DIR = BASE_DIR / "models" / "policy_faiss_index"


def _normalize_path(path: Optional[str | Path], fallback: Path) -> Path:
    return Path(path).expanduser().resolve() if path else fallback


def _load_pdf_documents(pdf_dir: Path) -> List[Any]:
    from langchain_community.document_loaders import PyPDFLoader

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {pdf_dir}. Add lending-policy PDFs before building the index."
        )

    documents: List[Any] = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = pdf_path.name
        documents.extend(pages)

    return documents


def _build_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest_policy_documents(
    pdf_dir: Optional[str | Path] = None,
    index_dir: Optional[str | Path] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Path:
    """
    Ingest lending-policy PDFs into a persisted FAISS index.
    """
    source_dir = _normalize_path(pdf_dir, DEFAULT_POLICY_DIR)
    vector_index_dir = _normalize_path(index_dir, DEFAULT_INDEX_DIR)
    vector_index_dir.mkdir(parents=True, exist_ok=True)

    documents = _load_pdf_documents(source_dir)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    embeddings = _build_embeddings(embedding_model)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(vector_index_dir))
    return vector_index_dir


def load_policy_vector_store(
    index_dir: Optional[str | Path] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Any:
    from langchain_community.vectorstores import FAISS

    vector_index_dir = _normalize_path(index_dir, DEFAULT_INDEX_DIR)
    if not vector_index_dir.exists():
        raise FileNotFoundError(
            f"Policy index not found at {vector_index_dir}. Run ingest_policy_documents() first."
        )

    embeddings = _build_embeddings(embedding_model)
    return FAISS.load_local(
        str(vector_index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_policy_documents(
    query: str,
    pdf_dir: Optional[str | Path] = None,
    index_dir: Optional[str | Path] = None,
    k: int = DEFAULT_TOP_K,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> List[Any]:
    """
    Retrieve the most relevant policy chunks for a borrower risk scenario.
    """
    vector_index_dir = _normalize_path(index_dir, DEFAULT_INDEX_DIR)
    if not vector_index_dir.exists():
        ingest_policy_documents(
            pdf_dir=pdf_dir,
            index_dir=vector_index_dir,
            embedding_model=embedding_model,
        )

    vector_store = load_policy_vector_store(
        index_dir=vector_index_dir,
        embedding_model=embedding_model,
    )
    return vector_store.similarity_search(query, k=k)


def get_policy_context(
    query: str,
    pdf_dir: Optional[str | Path] = None,
    index_dir: Optional[str | Path] = None,
    k: int = DEFAULT_TOP_K,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> str:
    """
    Return a formatted context block containing lending guidance relevant to the query.
    """
    documents = retrieve_policy_documents(
        query=query,
        pdf_dir=pdf_dir,
        index_dir=index_dir,
        k=k,
        embedding_model=embedding_model,
    )

    formatted_chunks: List[str] = []
    for doc in documents:
        source = doc.metadata.get("source", "Unknown Policy")
        page = doc.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        content = " ".join(doc.page_content.split())
        formatted_chunks.append(f"[Source: {source}{page_label}] {content}")

    return "\n\n".join(formatted_chunks)


def build_policy_query(
    borrower_profile: Dict[str, Any],
    risk_score: Optional[float] = None,
) -> str:
    """
    Turn a borrower profile into a natural-language retrieval query.
    """
    signals: List[str] = []

    credit_amount = borrower_profile.get("credit_amount")
    duration = borrower_profile.get("duration")
    savings = borrower_profile.get("saving_accounts")
    checking = borrower_profile.get("checking_account")
    housing = borrower_profile.get("housing")
    job = borrower_profile.get("job")
    purpose = borrower_profile.get("purpose")
    age = borrower_profile.get("age")
    dti = borrower_profile.get("dti")

    if risk_score is not None:
        risk_band = "high risk" if risk_score >= 0.5 else "lower risk"
        signals.append(f"borrower classified as {risk_band} with default probability {risk_score:.2f}")

    if dti is not None:
        signals.append(f"debt-to-income ratio is {dti}")
        try:
            if float(dti) >= 0.43:
                signals.append("high DTI mitigation policy and compensating factors")
        except (TypeError, ValueError):
            pass

    if credit_amount is not None:
        signals.append(f"requested credit amount is {credit_amount}")
        try:
            if float(credit_amount) >= 10000:
                signals.append("large exposure approval thresholds")
        except (TypeError, ValueError):
            pass

    if duration is not None:
        signals.append(f"loan duration is {duration} months")
        try:
            if float(duration) >= 36:
                signals.append("long-tenor loan policy exceptions")
        except (TypeError, ValueError):
            pass

    if savings in {"NA", "little"}:
        signals.append("limited savings balance")
    if checking in {"NA", "little"}:
        signals.append("weak checking account history")
    if housing == "rent":
        signals.append("borrower is renting")
    if job in {0, 1}:
        signals.append("lower employment stability")
    if purpose:
        signals.append(f"loan purpose is {purpose}")
    if age is not None:
        signals.append(f"borrower age is {age}")

    joined_signals = "; ".join(signals) if signals else "general underwriting and credit risk policy"
    return (
        "Retrieve lending policy guidance, underwriting rules, risk mitigation requirements, "
        f"and approval conditions for this borrower scenario: {joined_signals}."
    )
