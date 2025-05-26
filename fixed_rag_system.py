from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.prompts import PromptTemplate
import os
import pickle
from datetime import datetime
from typing import List, Dict, Any
from langchain.schema import Document

class DocumentIngestion:
    def __init__(self, document_path, chunk_size=1000, chunk_overlap=200):
        self.document_path = document_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
    def load_document(self) -> List[Document]:
        """
        Load document based on file extension and return list of Document objects
        ready for embedding
        """
        _, file_extension = os.path.splitext(self.document_path)
        
        try:
            if file_extension.lower() == '.pdf':
                # Load PDF document using PyPDFLoader
                loader = PyPDFLoader(self.document_path)
                documents = loader.load()
                
            elif file_extension.lower() == '.txt':
                # Load text document using TextLoader
                loader = TextLoader(self.document_path)
                documents = loader.load()
                
            elif file_extension.lower() == '.csv':
                # Load CSV document - properly specify the CSV column names to use
                loader = CSVLoader(
                    file_path=self.document_path,
                    csv_args={
                        'delimiter': ',',
                        'quotechar': '"',
                    },
                    # By default, it creates a metadata column with 'source' file path
                )
                documents = loader.load()
                
            elif file_extension.lower() == '.json':
                # Load JSON document
                loader = JSONLoader(
                    file_path=self.document_path,
                    jq_schema=".",  # Extract data at root level
                    content_key=None,  # We'll use a custom extraction function
                    text_content=False,
                    json_lines=False
                )
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Split documents into chunks for better context management
            split_documents = self.text_splitter.split_documents(documents)
            
            print(f"Loaded {len(documents)} document(s) and split into {len(split_documents)} chunks")
            return split_documents
            
        except Exception as e:
            print(f"Error loading document: {str(e)}")
            raise
    
    def process_documents_for_embedding(self) -> List[Document]:
        """
        Process documents to prepare them for embedding
        """
        # Load and chunk the document
        chunks = self.load_document()
        return chunks

class VectorStore:
    def __init__(self, embedding_model_name="text-embedding-ada-002", 
                 persist_directory="./vector_store"):
        
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        
        # Initialize the embedding model with HuggingFaceEmbeddings
        self.embeddings = AzureOpenAIEmbeddings(azure_deployment=embedding_model_name)
        
        # Make sure the persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def create_vector_store(self, documents):
        """
        Create a FAISS vector store from documents
        """
        vector_store = FAISS.from_documents(documents, self.embeddings)
        return vector_store
    
    def save_vector_store(self, vector_store, name=None):
        """
        Save the vector store to disk
        """
        if name is None:
            name = f"vectorstore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        save_path = os.path.join(self.persist_directory, name)
        vector_store.save_local(save_path)
        print(f"Vector store saved to {save_path}")
        return save_path
    
    def load_vector_store(self, path):
        """
        Load a vector store from disk
        """
        if os.path.exists(path):
            vector_store = FAISS.load_local(path, self.embeddings)
            print(f"Vector store loaded from {path}")
            return vector_store
        else:
            raise FileNotFoundError(f"No vector store at {path}")
        

class RAGSystem:
    def __init__(self, model_name="gpt-35-turbo", temperature=0.0, embedding_model="text-embedding-ada-002"):
        self.model_name = model_name
        self.temperature = temperature
        self.embedding_model = embedding_model
        self.vector_store = None
        self.qa_chain = None
        self.conversation_history = []
        
    def ingest_document(self, document_path, chunk_size=1000, chunk_overlap=200):
        """
        Ingest a document, chunk it, and create vector store
        """
        # Create document ingestion instance with explicit chunk_size and chunk_overlap
        ingestion = DocumentIngestion(document_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = ingestion.process_documents_for_embedding()
        
        # Create vector store
        vector_store_handler = VectorStore(embedding_model_name=self.embedding_model)
        self.vector_store = vector_store_handler.create_vector_store(documents)
        vector_store_path = vector_store_handler.save_vector_store(self.vector_store)
        
        return {
            "documents": len(documents),
            "vector_store_path": vector_store_path
        }
    
    def load_existing_vector_store(self, vector_store_path):
        """
        Load an existing vector store
        """
        vector_store_handler = VectorStore(embedding_model_name=self.embedding_model)
        self.vector_store = vector_store_handler.load_vector_store(vector_store_path)
    
    def setup_qa_chain(self):
        """
        Set up the QA chain with appropriate templates for context-aware responses
        """
        if self.vector_store is None:
            raise ValueError("Vector store must be initialized before setting up QA chain")
            
        # Create a standard retrieval chain without custom prompt
        # This simplifies the chain and avoids the input key errors
        model = AzureChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4})
        )
    
    def ask(self, question):
        """
        Ask a question to the RAG system
        """
        if self.qa_chain is None:
            self.setup_qa_chain()
        try:
            # Get answer - simplified to just pass the question
            result = self.qa_chain({"query": question})
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": result["result"]})
            
            return result["result"]
        except Exception as e:
            error_message = f"Error in asking question: {str(e)}"
            print(error_message)
            return error_message

# Example usage
if __name__ == "__main__":
    document_path = 'students.csv'
    rag_system = RAGSystem()
    
    # Ingest the document and create vector store
    ingest_result = rag_system.ingest_document(document_path, chunk_size=1000, chunk_overlap=200)
    print(f"Document ingestion result: {ingest_result}")
    
    # Set up QA chain
    rag_system.setup_qa_chain()
    
    # Ask a question
    question = "What is the total marks of Student_12?"
    answer = rag_system.ask(question)
    print(f"Answer: {answer}")