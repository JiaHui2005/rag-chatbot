import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from core.remote_llm import RemoteLLM

class RAGEngine:
    def __init__(self, config: dict):
        self.config = config
        self.embeddings = OpenAIEmbeddings()
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
        prompt_template = """Sử dụng các thông tin ngữ cảnh sau đây để trả lời câu hỏi của người dùng. 
Nếu bạn không tìm thấy câu trả lời trong ngữ cảnh, hãy nói rằng bạn không biết, đừng cố bịa ra câu trả lời. 
Luôn trả lời bằng tiếng Việt một cách rõ ràng, chi tiết và lịch sự.

Ngữ cảnh:
{context}

Câu hỏi: {question}
Trả lời:"""
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
            else:
                loader = TextLoader(path)
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get('rag', {}).get('chunk_size', 1000),
            chunk_overlap=self.config.get('rag', {}).get('chunk_overlap', 200)
        )
        splits = text_splitter.split_documents(documents)
        
        self.vector_db = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./db"
        )
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
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm_with_mode,
                    chain_type="stuff",
                    retriever=self.vector_db.as_retriever(search_kwargs={"k": 5}),
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