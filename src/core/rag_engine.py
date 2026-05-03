import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faulthandler
faulthandler.enable()

import json
import hashlib
import math
import re
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, BSHTMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from core.remote_llm import RemoteLLM


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "merged_chunks.jsonl"

STOPWORDS = {
    "luật", "đất", "đai", "năm", "quy", "định", "như", "thế", "nào",
    "của", "về", "và", "các", "cho", "được", "trong", "theo", "này",
}


class SimpleHashEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 1024):
        self.dimensions = dimensions

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class RAGEngine:
    def __init__(self, config: dict):
        self.config = config
        rag_config = config.get("rag", {})
        persist_dir = self.resolve_path(rag_config.get("persist_dir", "db/chroma"))
        collection_name = rag_config.get("collection_name", "land_law_rag")
        manifest = self.load_index_manifest(persist_dir)

        # ========================
        # 🔹 EMBEDDING
        # ========================
        model_name = os.getenv(
            "EMBEDDING_MODEL",
            rag_config.get("embedding_model", "all-MiniLM-L6-v2")
        )
        embedding_backend = os.getenv(
            "EMBEDDING_BACKEND",
            rag_config.get("embedding_backend", "auto")
        )
        if embedding_backend == "auto":
            embedding_backend = manifest.get("embedding_backend_used", "huggingface")

        if embedding_backend == "simple":
            print("Load embedding: simple hash embeddings")
            self.embeddings = SimpleHashEmbeddings()
        else:
            print(f"Load embedding: {model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'}
            )

        self.vector_db = None
        self.bm25_retriever = None
        self.chunk_documents = self.load_chunk_documents(
            self.resolve_path(str(rag_config.get("chunks_path", DEFAULT_CHUNKS_PATH)))
        )

        if persist_dir.exists():
            self.vector_db = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=self.embeddings,
                collection_name=collection_name,
            )
            print(f"Loaded vector DB: {persist_dir} ({collection_name})")
        else:
            legacy_dir = ROOT_DIR / "db"
            if legacy_dir.exists():
                self.vector_db = Chroma(
                    persist_directory=str(legacy_dir),
                    embedding_function=self.embeddings
                )
                print(f"Loaded legacy vector DB: {legacy_dir}")

        if self.chunk_documents:
            self.bm25_retriever = BM25Retriever.from_documents(
                self.chunk_documents,
                preprocess_func=self.tokenize
            )
            self.bm25_retriever.k = int(rag_config.get("bm25_k", 8))
            print(f"Loaded BM25 index from {len(self.chunk_documents)} chunks")

        # ========================
        # 🔹 LLM
        # ========================
        self.llm = RemoteLLM(
            api_url=os.getenv("COLAB_API_URL", ""),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", 512)),
            temperature=float(os.getenv("TEMPERATURE", 0.3))
        )

        # ========================
        # 🔥 PROMPT (ANTI-LAN-MAN)
        # ========================
        self.rag_prompt = PromptTemplate(
            template="""Bạn là trợ lý pháp lý Việt Nam. Trả lời dựa đúng vào TÀI LIỆU.

YÊU CẦU:
- Chỉ dùng thông tin có trong TÀI LIỆU.
- Ưu tiên điều luật liên quan trực tiếp nhất.
- Nếu TÀI LIỆU không đủ để trả lời, nói rõ là chưa đủ căn cứ.
- Không bịa thêm chủ thể, thủ tục hoặc điều kiện ngoài tài liệu.

CÁCH TRẢ LỜI:
- Mở đầu bằng điều luật chính nếu có.
- Trả lời dạng gạch đầu dòng, ngắn gọn.
- Với câu hỏi pháp luật rộng, tóm tắt các ý chính thay vì chép dài.

TÀI LIỆU:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:""",
            input_variables=["context", "question"]
        )

        self.no_rag_prompt = PromptTemplate(
            template="Câu hỏi: {question}\nTrả lời:",
            input_variables=["question"]
        )

    def resolve_path(self, path: str) -> Path:
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return ROOT_DIR / path_obj

    def load_index_manifest(self, persist_dir: Path) -> dict:
        manifest_path = persist_dir / "index_manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            print(f"Cannot read index manifest: {exc}")
            return {}

    def load_chunk_documents(self, path: Path) -> List[Document]:
        if not path.exists():
            print(f"Missing processed chunks: {path}")
            return []

        docs: List[Document] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                metadata = dict(record.get("metadata", {}))
                metadata["chunk_id"] = record.get("chunk_id")
                metadata["doc_id"] = record.get("doc_id")
                metadata["chunk_index"] = record.get("chunk_index")
                docs.append(Document(page_content=record["text"], metadata=metadata))
        return docs

    def is_legal_query(self, question: str) -> bool:
        # 1. Kiểm tra qua từ khóa (Cách nhẹ nhất)
        legal_keywords = ["luật", "đất đai", "quy định", "thủ tục", "nghị định", "thông tư", "quyền", "nghĩa vụ", "tranh chấp"]
        question_lower = question.lower()
        
        # Nếu có từ khóa pháp lý rõ ràng, cho qua luôn
        if any(kw in question_lower for kw in legal_keywords):
            return True
        
        # 2. (Nâng cao) Dùng LLM để phân loại nhanh
        # Bạn có thể dùng một prompt cực ngắn để LLM xác định True/False
        # Nhưng để tiết kiệm, bước 1 là đủ dùng cho mức độ cơ bản.
        return False

    def calculate_top_score(self, doc: Document, query: str) -> float:
        # Tái sử dụng logic chấm điểm của bạn để kiểm tra ngưỡng
        keywords = self.extract_keywords(query)
        text = doc.page_content.lower()
        # Tính toán điểm dựa trên keywords, tương tự như trong rerank_documents
        score = sum(2.0 for kw in keywords if kw in text)
        if " ".join(keywords) in text:
            score += 5.0
        return score

    # ========================
    # 🔹 LOAD DATA
    # ========================
    def process_documents(self, file_paths: List[str]):
        docs = []

        for path in file_paths:
            if path.endswith('.pdf'):
                loader = PyPDFLoader(path)
            elif path.endswith('.docx'):
                loader = Docx2txtLoader(path)
            elif path.endswith('.html'):
                loader = BSHTMLLoader(path)
            else:
                loader = TextLoader(path, encoding='utf-8')

            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            separators=["\nĐiều ", "\n\n", "\n"],
            chunk_size=3000,
            chunk_overlap=300
        )

        splits = splitter.split_documents(docs)

        rag_config = self.config.get("rag", {})
        persist_dir = self.resolve_path(rag_config.get("persist_dir", "db/chroma"))
        collection_name = rag_config.get("collection_name", "land_law_rag")

        self.vector_db = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=str(persist_dir),
            collection_name=collection_name,
        )

        self.chunk_documents = splits
        self.bm25_retriever = BM25Retriever.from_documents(
            splits,
            preprocess_func=self.tokenize
        )
        self.bm25_retriever.k = int(rag_config.get("bm25_k", 8))

        return True

    # ========================
    # 🔥 QUERY NORMALIZATION / EXPANSION
    # ========================
    def normalize_query(self, question: str) -> str:
        cleaned = re.sub(r"\s+", " ", question).strip()
        keywords = self.extract_keywords(cleaned)
        expansions = []
        if keywords:
            expansions.append(" ".join(keywords))

        if not expansions:
            return cleaned
        return f"{cleaned} {' '.join(expansions)}"

    # ========================
    # 🔹 KEYWORD EXTRACTION (nhẹ)
    # ========================
    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def extract_keywords(self, query: str):
        words = self.tokenize(query)
        return [w for w in words if len(w) > 2 and w not in STOPWORDS]

    def infer_target_articles(self, query: str) -> List[str]:
        lowered = query.lower()
        articles = re.findall(r"điều\s+(\d+)", lowered)
        deduped = []
        for article in articles:
            label = f"Điều {article}"
            if label not in deduped:
                deduped.append(label)
        return deduped

    def topic_overlap_score(self, topic: str, keywords: List[str]) -> int:
        normalized_topic = topic.replace("_", " ").lower()
        return sum(1 for keyword in keywords if keyword in normalized_topic)

    def doc_key(self, doc: Document) -> Tuple[str, int]:
        metadata = doc.metadata or {}
        return (
            str(metadata.get("doc_id") or hash(doc.page_content)),
            int(metadata.get("chunk_index") or 0),
        )

    def rerank_documents(self, docs: List[Document], query: str) -> List[Document]:
        keywords = self.extract_keywords(query)
        target_articles = self.infer_target_articles(query)
        phrase = " ".join(keywords)

        def score(doc: Document) -> float:
            text = doc.page_content.lower()
            metadata = doc.metadata or {}
            article = str(metadata.get("article") or "")
            topic = str(metadata.get("topic") or "").lower()
            source_type = str(metadata.get("source_type") or "")

            value = 0.0
            value += sum(1.5 for kw in keywords if kw in text)
            if phrase and phrase in text:
                value += 3.0
            value += self.topic_overlap_score(topic, keywords) * 3.0

            for target in target_articles:
                if article == target:
                    value += 12.0
                if target.lower() in text[:220]:
                    value += 6.0

            if source_type == "legal_document":
                value += 2.0
            if "giải_thích_từ_ngữ" in topic:
                value -= 3.0
            if "bài viết liên quan" in text or "thống kê truy cập" in text:
                value -= 10.0

            return value

        return sorted(docs, key=score, reverse=True)

    def add_neighbor_chunks(self, docs: List[Document], query: str) -> List[Document]:
        if not docs or not self.chunk_documents:
            return docs

        selected_keys = {self.doc_key(doc) for doc in docs}
        docs_by_key = {self.doc_key(doc): doc for doc in self.chunk_documents}
        expanded = list(docs)
        target_articles = set(self.infer_target_articles(query))
        top_article = (docs[0].metadata or {}).get("article")
        if top_article:
            target_articles.add(top_article)
        if not target_articles:
            return docs

        for doc in docs[:3]:
            metadata = doc.metadata or {}
            if metadata.get("article") not in target_articles:
                continue

            doc_id = metadata.get("doc_id")
            chunk_index = int(metadata.get("chunk_index") or 0)
            for neighbor_index in (chunk_index - 1, chunk_index + 1):
                key = (str(doc_id), neighbor_index)
                if key in docs_by_key and key not in selected_keys:
                    expanded.append(docs_by_key[key])
                    selected_keys.add(key)

        return expanded

    def is_exact_question_match(self, doc: Document, question: str) -> bool:
        metadata = doc.metadata or {}
        doc_question = str(metadata.get("question") or "")
        if not doc_question:
            return False
        return self.normalize_for_match(doc_question) == self.normalize_for_match(question)

    def normalize_for_match(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def apply_high_confidence_filter(
        self,
        docs: List[Document],
        question: str,
        query: str,
    ) -> List[Document]:
        target_articles = set(self.infer_target_articles(query))
        if not docs:
            return docs
        top_metadata = docs[0].metadata or {}
        top_article = top_metadata.get("article")
        top_source_type = top_metadata.get("source_type")
        keywords = self.extract_keywords(query)

        if top_article and top_source_type == "legal_document":
            top_text = docs[0].page_content.lower()
            overlap = sum(1 for keyword in keywords if keyword in top_text)
            if overlap >= max(2, min(4, len(keywords))):
                target_articles.add(top_article)

        if not target_articles:
            exact_matches = [doc for doc in docs if self.is_exact_question_match(doc, question)]
            if exact_matches:
                articles_from_exact = self.extract_articles_from_docs(exact_matches, first_only=True)
                if articles_from_exact:
                    exact_matches = self.include_article_chunks(exact_matches, articles_from_exact)
                return exact_matches
            return docs

        filtered = []
        for doc in docs:
            metadata = doc.metadata or {}
            article = metadata.get("article")
            if article in target_articles or self.is_exact_question_match(doc, question):
                filtered.append(doc)

        min_docs = int(self.config.get("rag", {}).get("min_high_confidence_docs", 2))
        if len(filtered) >= min_docs:
            return filtered
        return docs

    def extract_articles_from_docs(self, docs: List[Document], first_only: bool = False) -> List[str]:
        articles = []
        for doc in docs:
            for article in re.findall(r"Điều\s+(\d+)", doc.page_content):
                label = f"Điều {article}"
                if label not in articles:
                    articles.append(label)
                    if first_only:
                        return articles
        return articles

    def include_article_chunks(self, docs: List[Document], articles: List[str]) -> List[Document]:
        if not self.chunk_documents:
            return docs

        selected_keys = {self.doc_key(doc) for doc in docs}
        expanded = list(docs)
        for article in articles:
            article_docs = [
                doc for doc in self.chunk_documents
                if (doc.metadata or {}).get("article") == article
            ]
            article_docs.sort(key=lambda item: int((item.metadata or {}).get("chunk_index") or 0))
            for doc in article_docs:
                key = self.doc_key(doc)
                if key not in selected_keys:
                    expanded.append(doc)
                    selected_keys.add(key)
        return expanded

    def retrieve_documents(self, question: str) -> Tuple[str, List[Document]]:
        query = self.normalize_query(question)
        rag_config = self.config.get("rag", {})
        vector_k = int(rag_config.get("vector_k", 8))
        bm25_k = int(rag_config.get("bm25_k", 8))
        final_k = int(rag_config.get("final_k", 5))

        candidates: List[Document] = []
        if self.vector_db:
            candidates.extend(
                self.vector_db.similarity_search(query, k=vector_k)
            )

        if self.bm25_retriever:
            self.bm25_retriever.k = bm25_k
            candidates.extend(self.bm25_retriever.invoke(query))

        seen = set()
        unique_docs = []
        for doc in candidates:
            key = self.doc_key(doc)
            if key not in seen:
                unique_docs.append(doc)
                seen.add(key)

        ranked = self.rerank_documents(unique_docs, query)
        ranked = self.rerank_documents(self.add_neighbor_chunks(ranked, query), query)
        ranked = self.apply_high_confidence_filter(ranked, question, query)

        if not ranked:
                return query, []

        top_score = self.calculate_top_score(ranked[0], query) 
            
        if top_score < 5.0: # Ngưỡng này bạn cần tinh chỉnh dựa trên thực tế
            return query, []

        return query, ranked[:final_k]

    def build_context(self, docs: List[Document]) -> str:
        max_chars = int(self.config.get("rag", {}).get("max_chars_per_doc", 1800))
        blocks = []
        for index, doc in enumerate(docs, start=1):
            metadata = doc.metadata or {}
            article = metadata.get("article") or "Không rõ điều"
            source_type = metadata.get("source_type") or "unknown"
            chunk_id = metadata.get("chunk_id") or "unknown"
            text = doc.page_content.strip()[:max_chars]
            blocks.append(
                f"[Nguồn {index}] {article} | {source_type} | {chunk_id}\n{text}"
            )
        return "\n\n".join(blocks)

    def print_debug_context(self, query: str, docs: List[Document], context: str) -> None:
        print(f"\nRetrieval Query: {query}\n")
        print("========== RETRIEVED DOCS ==========")
        for index, doc in enumerate(docs, start=1):
            metadata = doc.metadata or {}
            snippet = re.sub(r"\s+", " ", doc.page_content[:180]).strip()
            print(
                f"{index}. {metadata.get('article')} | "
                f"{metadata.get('topic')} | {metadata.get('chunk_id')}"
            )
            print(f"   {snippet}")
        print("========== CONTEXT ==========")
        print(context[:1500])
        print("=============================\n")

    def extractive_answer(self, question: str, docs: List[Document]) -> str:
        if not docs:
            return ""

        primary_doc = next(
            (doc for doc in docs if (doc.metadata or {}).get("source_type") == "legal_document"
             and (doc.metadata or {}).get("article")),
            None,
        )
        if not primary_doc and self.is_exact_question_match(docs[0], question):
            faq_answer = self.extract_faq_answer(docs[0].page_content)
            if faq_answer:
                return faq_answer

        if not primary_doc:
            return ""

        first_metadata = primary_doc.metadata or {}
        primary_article = first_metadata.get("article")
        primary_source = first_metadata.get("source_type")
        if primary_source != "legal_document" or not primary_article:
            return ""

        article_docs = [
            doc for doc in docs
            if (doc.metadata or {}).get("article") == primary_article
        ]
        if not article_docs:
            return ""

        lines = []
        for doc in sorted(article_docs, key=lambda item: int((item.metadata or {}).get("chunk_index") or 0)):
            text = self.clean_legal_text(doc.page_content, primary_article)
            for line in self.split_legal_lines(text):
                if line and line not in lines:
                    lines.append(line)

        if not lines:
            return ""

        max_bullets = int(self.config.get("rag", {}).get("max_extractive_bullets", 8))
        bullets = "\n".join(f"- {line}" for line in lines[:max_bullets])
        return f"{primary_article} Luật Đất đai năm 2024 quy định như sau:\n{bullets}"

    def extract_faq_answer(self, text: str) -> str:
        match = re.search(r"Trả lời(?:\s*\(tiếp\))?:\s*(.+)", text, flags=re.DOTALL)
        if not match:
            return ""
        answer = re.sub(r"\s+", " ", match.group(1)).strip()
        parts = self.split_legal_lines(answer)
        if not parts:
            return answer
        max_bullets = int(self.config.get("rag", {}).get("max_extractive_bullets", 8))
        return "\n".join(f"- {part}" for part in parts[:max_bullets])

    def clean_legal_text(self, text: str, article: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        marker = f"{article}."
        if marker in text:
            text = text.split(marker, 1)[1].strip()
        text = re.sub(r"^GIAO ĐẤT, CHO THUÊ ĐẤT, CHUYỂN MỤC ĐÍCH SỬ DỤNG ĐẤT\.\s*", "", text)
        first_clause = re.search(r"\b1\.", text)
        if first_clause and first_clause.start() < 160:
            text = text[first_clause.start():].strip()
        return text

    def split_legal_lines(self, text: str) -> List[str]:
        parts = re.split(r"(?=(?:\d+\.|[a-zđ]\)))", text)
        lines = []
        for part in parts:
            cleaned = part.strip(" ;")
            if len(cleaned) < 12:
                continue
            lines.append(cleaned)
        return lines or [text]

    # ========================
    # 🔹 QUERY
    # ========================
    def query(self, question: str, mode: str = "base"):
        try:
            # if not self.is_legal_query(question):
            #     return "Xin lỗi, tôi là trợ lý chuyên về Luật Đất đai Việt Nam. Tôi không thể trả lời các câu hỏi ngoài lĩnh vực này."

            if not self.llm.api_url:
                return "Chưa cấu hình API"

            llm = self.llm.copy(update={"mode": mode})

            # ========================
            # 🔹 NO RAG
            # ========================
            if mode in ["base", "ft"]:
                chain = self.no_rag_prompt | llm
                return chain.invoke({"question": question})

            # ========================
            # 🔹 RAG
            # ========================
            if not self.is_legal_query(question):
                return "Xin lỗi, ở chế độ tra cứu luật, tôi chỉ có thể hỗ trợ các vấn đề liên quan đến Luật Đất đai Việt Nam."
    
            if not self.vector_db and not self.bm25_retriever:
                return "Chưa load tài liệu"

            retrieval_query, selected_docs = self.retrieve_documents(question)
            if not selected_docs:
                return "Câu hỏi của bạn không nằm trong phạm vi dữ liệu pháp luật hiện có của tôi hoặc không liên quan đến Luật Đất đai."

            context = self.build_context(selected_docs)
            self.print_debug_context(retrieval_query, selected_docs, context)

            if self.config.get("rag", {}).get("use_extractive_answer", True):
                answer = self.extractive_answer(question, selected_docs)
                if answer:
                    return answer

            # ========================
            # 🔹 GENERATE
            # ========================
            chain = self.rag_prompt | llm

            return chain.invoke({
                "context": context,
                "question": question
            })

        except Exception as e:
            return f"Error: {str(e)}"
