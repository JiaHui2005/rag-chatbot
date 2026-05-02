import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from core.remote_llm import RemoteLLM

class RAGEngine:
    def __init__(self, config: dict):
        self.config = config
        
        # Sử dụng HuggingFace Embeddings miễn phí (chạy local)
        model_name = os.getenv("EMBEDDING_MODEL", "YuITC/vietnamese-embedding-vn-legal")
        print(f"Đang load mô hình Embedding: {model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'} # Chạy trên CPU cho nhẹ máy local
        )
        self.vector_db = None
        
        # Load ChromaDB if exists
        if os.path.exists("./db"):
            self.vector_db = Chroma(
                persist_directory="./db", 
                embedding_function=self.embeddings
            )
            print("Đã load Vector DB từ ./db")

        # Khởi tạo Remote LLM (Chạy trên Google Colab)
        colab_url = os.getenv("COLAB_API_URL", "http://your-ngrok-url-placeholder")
        
        print(f"Đang kết nối tới Colab API: {colab_url}")
        
        self.llm = RemoteLLM(
            api_url=colab_url,
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", 512)),
            temperature=float(os.getenv("TEMPERATURE", 0.3))
        )

        # Prompt Template Tiếng Việt cho RAG
# Prompt Template Tiếng Việt ĐƠN GIẢN cho Phi-3
        prompt_template = """Bạn là trợ lý Pháp lý. Hãy đọc kỹ TÀI LIỆU dưới đây và trả lời CÂU HỎI của người dùng.
YÊU CẦU BẮT BUỘC:
1. Chỉ trả lời dựa trên TÀI LIỆU, tuyệt đối không tự sáng tác thêm các Điều luật không có trong tài liệu.
2. Trả lời ngắn gọn, đúng trọng tâm và KẾT THÚC CÂU TRẢ LỜI NGAY LẬP TỨC khi đã đủ ý.

TÀI LIỆU:
{context}

CÂU HỎI:
{question}"""
        
        self.rag_prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        # Prompt Template cho Không RAG
        no_rag_template = """Bạn là một trợ lý ảo am hiểu pháp luật Việt Nam. Hãy trả lời câu hỏi sau của người dùng bằng tiếng Việt một cách rõ ràng, chi tiết và lịch sự.
        
Câu hỏi: {question}
Trả lời:"""
        self.no_rag_prompt = PromptTemplate(
            template=no_rag_template, input_variables=["question"]
        )
    def process_documents(self, file_paths: List[str]):
        """Xử lý tài liệu và lưu vào Vector DB"""
        documents = []
        for path in file_paths:
            if path.endswith('.pdf'):
                loader = PyPDFLoader(path)
            elif path.endswith('.docx'):
                loader = Docx2txtLoader(path)
            elif path.endswith('.html'):
                loader = BSHTMLLoader(path)
            else:
                loader = TextLoader(path, encoding='utf-8')
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\nĐiều ",  # Cắt ưu tiên 1: Tách riêng từng Điều
                "\n\n",     # Cắt ưu tiên 2: Cắt theo đoạn văn lớn nếu Điều quá dài
                "\n",       # Cắt ưu tiên 3: Xuống dòng thông thường
                ". "        # Cắt ưu tiên 4: Dấu chấm câu
            ],
            chunk_size=5000,   # Tăng từ 3000 lên 5000 để chứa trọn vẹn cả những Điều luật cực dài
            chunk_overlap=500, # Tăng overlap để giữ ngữ cảnh liền mạch
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        
        if not splits:
            raise ValueError("Không tìm thấy nội dung văn bản hợp lệ nào trong các file tài liệu. Xin hãy kiểm tra lại file (ví dụ PDF có thể là file ảnh không đọc được chữ).")
            
        self.vector_db = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./db"
        )
        # TẠO THÊM BM25 RETRIEVER TỪ CÁC CHUNKS
        self.bm25_retriever = BM25Retriever.from_documents(splits)
        self.bm25_retriever.k = 2 # Lấy top 2 theo từ khóa
        
        return True

    def query(self, question: str, mode: str = "base"):
        """
        Truy vấn hệ thống với 4 chế độ:
        - mode="base": Base Model Không RAG
        - mode="base_rag": Base Model Có RAG
        - mode="ft": Fine-Tuned Model Không RAG
        - mode="ft_rag": Fine-Tuned Model Có RAG
        """
        try:
            # Kiểm tra URL API trước khi chạy
            if not self.llm.api_url or "your-ngrok-url" in self.llm.api_url:
                return "❌ Model chưa sẵn sàng: Bạn chưa cấu hình COLAB_API_URL trong file .env."

            # Cập nhật mode cho LLM để Server Colab biết có dùng Adapter hay không
            llm_with_mode = self.llm.copy(update={"mode": mode})

            # Thực thi Không RAG
            if mode in ["base", "ft"]:
                chain = self.no_rag_prompt | llm_with_mode
                return chain.invoke({"question": question})

            # Thực thi Có RAG
            elif mode in ["base_rag", "ft_rag"]:
                if not self.vector_db:
                    return "Vui lòng tải lên tài liệu hoặc khởi tạo DB trước."
                
                # Khởi tạo Vector Retriever
                vector_retriever = self.vector_db.as_retriever(search_kwargs={"k": 2})
                
                # Kết hợp thành Hybrid Retriever (NẾU bm25_retriever tồn tại)
                if hasattr(self, 'bm25_retriever') and self.bm25_retriever:
                    final_retriever = EnsembleRetriever(
                        retrievers=[self.bm25_retriever, vector_retriever],
                        weights=[0.5, 0.5] # 50% trọng số cho từ khóa, 50% cho ngữ nghĩa
                    )
                else:
                    final_retriever = vector_retriever
                
                # --- BẮT ĐẦU IN NGỮ CẢNH RA TERMINAL ĐỂ DEBUG ---
                retrieved_docs = final_retriever.invoke(question)
                rag_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                print("\n" + "="*50)
                print("📝 NGỮ CẢNH RAG (CONTEXT) ĐƯỢC LẤY TỪ DB/BM25:")
                print("="*50)
                print(rag_context)
                print("="*50 + "\n")
                # --- KẾT THÚC IN ---
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm_with_mode,
                    chain_type="stuff",
                    retriever=final_retriever, # Sử dụng bộ truy xuất đã nâng cấp
                    chain_type_kwargs={"prompt": self.rag_prompt}
                )
                
                # Thực thi chuỗi RAG (Có thể gặp lỗi API OpenAI Embeddings)
                response = qa_chain.invoke(question)
                return response["result"] if isinstance(response, dict) else response
            
            return "Mode không hợp lệ."

        except Exception as e:
            # Xử lý lỗi API (OpenAI quota, Colab connection, v.v.)
            error_msg = str(e)
            if "insufficient_quota" in error_msg:
                return "❌ Lỗi: Tài khoản OpenAI của bạn đã hết hạn hoặc hết hạn mức (Quota). Vui lòng kiểm tra lại API Key hoặc nạp thêm tiền."
            elif "RemoteLLM" in error_msg or "connection" in error_msg.lower():
                return f"❌ Lỗi kết nối Server Colab: {error_msg}. Hãy đảm bảo Server Colab đang chạy và link trong .env là chính xác."
            else:
                return f"⚠️ Hệ thống AI gặp sự cố: {error_msg}"