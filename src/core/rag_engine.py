import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

class RAGEngine:
    def __init__(self, config: dict):
        self.config = config
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model_name=config['llm']['model'],
            temperature=config['llm']['temperature']
        )
        self.vector_db = None

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
            chunk_size=self.config['rag']['chunk_size'],
            chunk_overlap=self.config['rag']['chunk_overlap']
        )
        splits = text_splitter.split_documents(documents)
        
        self.vector_db = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./db"
        )
        return True

    def query(self, question: str):
        """Truy vấn hệ thống RAG"""
        if not self.vector_db:
            return "Vui lòng tải lên tài liệu trước."
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever()
        )
        return qa_chain.run(question)
