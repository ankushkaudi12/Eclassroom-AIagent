# Import necessary libraries
import os
import asyncio
from typing import List, Dict, Any, Optional
import pickle
import json
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
import logging

# Load environment variables
load_dotenv()

# Initialize global variables
llm = None
vector_store = None
graph = None

# Add paths for stored data
PROCESSED_DATA_DIR = "processed_data"
VECTOR_STORE_PATH = os.path.join(PROCESSED_DATA_DIR, "vector_store.pkl")
METADATA_PATH = os.path.join(PROCESSED_DATA_DIR, "metadata.json")

# Create processed_data directory if it doesn't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,       # Use DEBUG or INFO as needed
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Logging setup complete, stdout is active.")

def save_vector_store(vector_store: 'SimpleVectorStore') -> None:
    """Save the vector store and its metadata to disk."""
    try:
        # Save vector store
        with open(VECTOR_STORE_PATH, 'wb') as f:
            pickle.dump(vector_store, f)
        
        # Save metadata
        metadata = {
            'last_processed': time.time(),
            'num_documents': len(vector_store.documents),
            'embedding_model': Config.EMBEDDING_MODEL
        }
        with open(METADATA_PATH, 'w') as f:
            json.dump(metadata, f)
        
        logging.info("Vector store and metadata saved successfully")
    except Exception as e:
        logging.error(f"Error saving vector store: {str(e)}")
        raise

def load_vector_store() -> Optional['SimpleVectorStore']:
    """Load the vector store from disk if it exists."""
    try:
        if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(METADATA_PATH):
            # Load metadata first to verify compatibility
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
                
            if metadata.get('embedding_model') != Config.EMBEDDING_MODEL:
                logging.warning("Embedding model mismatch, need to reprocess documents")
                return None
                
            # Load vector store
            with open(VECTOR_STORE_PATH, 'rb') as f:
                vector_store = pickle.load(f)
                logging.info(f"Loaded vector store with {len(vector_store.documents)} documents")
                return vector_store
    except Exception as e:
        logging.error(f"Error loading vector store: {str(e)}")
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the AI agent components when the API starts."""
    global llm, vector_store, graph
    try:
        print("Initializing AI agent...")
        llm = initialize_llm()
        
        # Try to load existing vector store
        vector_store = load_vector_store()
        
        if vector_store is None:
            print("No existing vector store found or it's outdated. Processing PDFs...")
            # Load and process PDFs
            pdf_dir = "data"
            print(f"Loading PDFs from {pdf_dir}")
            documents = load_and_process_pdfs(pdf_dir)
            
            if not documents:
                raise ValueError("No documents available for processing")

            # Create vector store
            print("Creating vector store...")
            vector_store = create_vector_store(documents)
            
            # Save the processed data
            print("Saving processed data...")
            save_vector_store(vector_store)
        else:
            print("Loaded existing vector store successfully")

        # Create the conversation graph
        graph_builder = StateGraph(State)
        graph_builder.add_node("process_query", process_user_query)
        graph_builder.add_edge(START, "process_query")
        graph_builder.add_edge("process_query", END)
        graph = graph_builder.compile()

        print("AI agent is ready!")
        yield
    except asyncio.CancelledError:
        print("Lifespan cancelled, shutting down...")
        return
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="AI Agent API",
    description="API for the AI Agent with PDF processing capabilities",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the service is running and PDFs are processed"""
    global vector_store
    return {
        "status": "healthy",
        "pdf_store_ready": vector_store is not None,
        "documents_processed": len(vector_store.documents) if vector_store else 0
    }

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    processing_time: float
    sources: List[Dict[str, Any]]

class Config:
    """Configuration class for the AI agent."""
    GROQ_AGENT_API = os.getenv('GROQ_AGENT_API')
    MODEL_NAME = "Gemma2-9b-It"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 30
    TEMPERATURE = 0.5

# Debug: Check if API key is loaded
groq_agent_api = os.getenv('GROQ_AGENT_API')
print(f"GROQ_AGENT_KEY loaded: {'Yes' if groq_agent_api else 'No'}")
if not groq_agent_api:
    raise ValueError("Please set GROQ_AGENT_KEY in your .env file")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

class SimpleVectorStore:
    """Efficient vector store implementation with caching."""
    def __init__(self):
        self.documents: List[Any] = []
        self.embeddings: np.ndarray = np.array([])
        self.metadata: List[Dict[str, Any]] = []
        self.sentence_transformer = SentenceTransformer(Config.EMBEDDING_MODEL)
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def __getstate__(self):
        """Custom state handling for pickle serialization."""
        state = self.__dict__.copy()
        # Don't pickle the transformer model - it will be reloaded
        del state['sentence_transformer']
        return state

    def __setstate__(self, state):
        """Custom state handling for pickle deserialization."""
        self.__dict__.update(state)
        # Reload the transformer model
        self.sentence_transformer = SentenceTransformer(Config.EMBEDDING_MODEL)

    def add_documents(self, documents: List[Dict[str, Any]], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add documents to the vector store with efficient batch processing."""
        if not documents:
            return

        self.documents.extend(documents)
        self.metadata.extend(metadatas if metadatas else [{}] * len(documents))

        # Batch process embeddings for efficiency
        texts = [doc.page_content for doc in documents]
        new_embeddings = self.sentence_transformer.encode(texts, show_progress_bar=False)
        
        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Efficient similarity search with caching."""
        if not self.documents:
            return []

        # Check cache first
        if query in self._embedding_cache:
            query_embedding = self._embedding_cache[query]
        else:
            query_embedding = self.sentence_transformer.encode(query)
            self._embedding_cache[query] = query_embedding

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return [{
            'document': self.documents[idx],
            'metadata': self.metadata[idx],
            'similarity': float(similarities[idx])
        } for idx in top_k_indices]

def load_and_process_pdfs(pdf_dir: str) -> List[Dict[str, Any]]:
    """Load and process PDFs with proper error handling."""
    documents = []
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in directory: {pdf_dir}")
        return documents

    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        try:
            print(f"Processing {os.path.basename(pdf_file)}")
            loader = PyPDFLoader(pdf_file)
            pages = loader.load()
            documents.extend(pages)
            print(f"Successfully processed {os.path.basename(pdf_file)} - {len(pages)} pages")
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue

    return documents

def create_vector_store(documents: List[Dict[str, Any]]) -> SimpleVectorStore:
    """Create and initialize vector store with optimized text splitting."""
    if not documents:
        print("No documents provided for vector store creation")
        raise ValueError("No documents provided")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )

    texts = text_splitter.split_documents(documents)
    vector_store = SimpleVectorStore()
    vector_store.add_documents(texts)
    return vector_store

# Define the state for the graph
class State(TypedDict):
    """State management for the conversation graph."""
    messages: List[Dict[str, str]]
    context: str
    conversation_history: List[Dict[str, str]]

def create_prompt_template() -> PromptTemplate:
    """Create an optimized prompt template for the LLM."""
    prompt_template = """You are a helpful AI assistant. Use the following context to answer the question.
    If the context doesn't contain relevant information, say so and provide a general answer.

    Context: {context}

    Conversation History:
    {conversation_history}

    Question: {question}

    Answer:"""

    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "conversation_history"]
    )

def process_user_query(state: State) -> Dict[str, Any]:
    """Process user query with error handling and efficient context retrieval."""
    global llm, vector_store
    try:
        if llm is None:
            llm = initialize_llm()
            
        query = state["messages"][-1]["content"]
        conversation_history = state.get("conversation_history", [])

        # Get relevant context
        results = vector_store.similarity_search(query, k=3)
        
        if not results:
            print("No relevant context found for query")
            response = llm.invoke([{"role": "user", "content": query}])
            return {"messages": [{"role": "assistant", "content": response.content}]}

        # Format context and conversation history
        context = "\n\n".join([
            f"Source: {result['metadata'].get('source', 'Unknown')} "
            f"(Page {result['metadata'].get('page', 'N/A')})\n"
            f"Content: {result['document'].page_content}"
            for result in results
        ])

        formatted_history = "\n".join([
            f"User: {msg['user']}\nAssistant: {msg['assistant']}"
            for msg in conversation_history[-3:]
        ])

        # Create and format prompt
        prompt = create_prompt_template()
        formatted_prompt = prompt.format(
            context=context,
            question=query,
            conversation_history=formatted_history
        )

        # Get response from LLM
        response = llm.invoke([{"role": "user", "content": formatted_prompt}])
        
        # Update conversation history
        conversation_history.append({
            "user": query,
            "assistant": response.content
        })

        return {
            "messages": [{"role": "assistant", "content": response.content}],
            "conversation_history": conversation_history
        }

    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return {
            "messages": [{"role": "assistant", "content": "I apologize, but I encountered an error while processing your query. Please try again."}],
            "conversation_history": state.get("conversation_history", [])
        }

def initialize_llm() -> ChatGroq:
    """Initialize the LLM with proper configuration."""
    if not Config.GROQ_AGENT_API:
        raise ValueError("GROQ_AGENT_KEY not found in environment variables")

    return ChatGroq(
        api_key=Config.GROQ_AGENT_API,
        model_name=Config.MODEL_NAME,
        temperature=Config.TEMPERATURE,
        max_retries=Config.MAX_RETRIES,
        request_timeout=Config.REQUEST_TIMEOUT
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query and return the response with source information"""
    global graph
    if graph is None:
        raise HTTPException(status_code=503, detail="AI agent is not initialized yet")
    
    try:
        start_time = time.time()
        response_content = ""
        source_documents = []
        
        # Get the query response
        for event in graph.stream({
            'messages': [{"role": "user", "content": request.query}],
            'conversation_history': []
        }):
            for value in event.values():
                if "messages" in value:
                    for msg in value["messages"]:
                        response_content = msg["content"]
        
        # Get relevant sources
        if vector_store:
            results = vector_store.similarity_search(request.query, k=3)
            source_documents = [
                {
                    "source": result["metadata"].get("source", "Unknown"),
                    "page": result["metadata"].get("page", "N/A"),
                    "similarity": float(result["similarity"])
                }
                for result in results
            ]
        
        processing_time = time.time() - start_time
        
        return {
            "response": response_content,
            "processing_time": processing_time,
            "sources": source_documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add endpoint to get PDF processing status
@app.get("/status")
async def get_processing_status():
    """Get the current status of PDF processing"""
    global vector_store
    if vector_store is None:
        return {"status": "not_initialized", "document_count": 0}
    
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        return {
            "status": "ready",
            "document_count": len(vector_store.documents),
            "last_processed": metadata.get('last_processed'),
            "embedding_model": metadata.get('embedding_model')
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    try:
        print("Starting the server...")
        import uvicorn
        import socket
        
        # Test if we can create a socket on the port
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.bind(('127.0.0.1', 8000))  # Try port 8000 instead
        test_socket.close()
        
        print("Port 8000 is available, starting server...")
        
        # Configure uvicorn with specific settings
        config = uvicorn.Config(
            app="main:app",
            host="127.0.0.1",
            port=8000,  # Changed to port 8000
            reload=True,
            log_level="info",
            access_log=True,
            workers=1
        )
        
        server = uvicorn.Server(config)
        server.run()
    except socket.error as e:
        print(f"Port error: {e}")
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        import traceback
        traceback.print_exc()


