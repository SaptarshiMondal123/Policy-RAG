import os
import logging
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class PolicyResponse(BaseModel):
    answer: str = Field(description="The direct answer to the user's question.")
    confidence: str = Field(description="Confidence level: 'High', 'Medium', or 'Low'.")
    context_used: bool = Field(description="True if the answer used the retrieved context, False otherwise.")

class RAGSystem:
    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")

        logger.info("Initializing RAG System...")
        
        # 1. Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            temperature=0.1,
            google_api_key=self.api_key
        )
        
        self.parser = JsonOutputParser(pydantic_object=PolicyResponse)

    def ingest_data(self, data_path="data"):
        logger.info(f"Loading data from {data_path}...")
        try:
            loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return
        
        if not documents:
            logger.warning("No documents found.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        logger.info(f"Split into {len(chunks)} chunks. Creating Vector Store...")
        self.vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings, 
            persist_directory=self.db_path
        )
        logger.info("Ingestion complete!")

    def load_vector_store(self):
        self.vector_store = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)

    def get_prompt(self, version="advanced"):
        if version == "basic":
            template = """You are a helpful assistant.
            Answer the user's question based on the context below.
            Context: {context}
            Question: {question}
            {format_instructions}"""
        else:
            template = """You are a Senior Research Analyst.
            INSTRUCTIONS:
            1. **Direct Answer:** Start with a clear, direct answer.
            2. **Evidence:** Support answer with specific details or quotes.
            3. **Structure:** Use bullet points and bold text.
            4. **Negative Constraint:** If answer is NOT in context, set 'answer' to "I cannot find this information".
            
            CONTEXT: {context}
            QUESTION: {question}
            {format_instructions}"""
        return ChatPromptTemplate.from_template(template)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, question, version="advanced"):
        if not hasattr(self, 'vector_store'):
            self.load_vector_store()
        
        logger.info(f"Received query: {question}")
        
        results = self.vector_store.similarity_search_with_score(question, k=10)
        filtered_docs = [doc for doc, score in results if score < 1.5][:5]
        
        if not filtered_docs:
            return {"answer": "I cannot find this information.", "confidence": "High", "context_used": False}
        
        rag_chain = (
            {"context": lambda x: self.format_docs(filtered_docs), 
             "question": RunnablePassthrough(),
             "format_instructions": lambda x: self.parser.get_format_instructions()}
            | self.get_prompt(version)
            | self.llm
            | self.parser
        )
        
        try:
            response = rag_chain.invoke(question)

            response['source_documents'] = [doc.page_content for doc in filtered_docs]

            return response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {"answer": "Error", "confidence": "Low", "context_used": False}