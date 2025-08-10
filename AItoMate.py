import streamlit as st
import voyageai
import faiss
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import io
import requests
from neo4j import GraphDatabase
from typing import List, Dict, Union, Optional
import os
import json
import re
from datetime import datetime, timedelta
import logging
import tempfile
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import sqlite3
from uuid import uuid4
import pytesseract 

# Page configuration
st.set_page_config(
    page_title="AItoMate",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
    font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #3f4b8c 0%, #4a3d6a 50%, #6a4b78 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(50, 60, 100, 0.4);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.05) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
    }

    .main-header p {
        font-size: 1.2rem;
        opacity: 0.85;
        font-weight: 300;
    }

    .metric-card {
        background: linear-gradient(135deg, #3f4b8c 0%, #4a3d6a 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(50, 60, 100, 0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(50, 60, 100, 0.5);
    }

    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        animation: fadeInUp 0.3s ease-out;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .user-message {
        background: linear-gradient(135deg, #4f6355 0%, #435f69 100%);
        border-left: 5px solid #384c7a;
        margin-left: 2rem;
        color: #e6e6e6;
    }

    .assistant-message {
        background: linear-gradient(135deg, #4a5f5d 0%, #6a5b58 100%);
        border-left: 5px solid #3c2750;
        margin-right: 2rem;
        color: #e6e6e6;
    }

    .system-message {
        background: linear-gradient(135deg, #4a5f5d 0%, #6a5b58 100%);
        border-left: 5px solid #b28c2b;
        text-align: center;
        font-style: italic;
        color: #e6e6e6;
    }

    .sidebar-section {
        background: linear-gradient(135deg, #2d2f31 0%, #3a3d3f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        border: 1px solid #444;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }

    .sidebar-section h3 {
        color: #e0e0e0;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .success-box {
        background: linear-gradient(135deg, #3f615c 0%, #5c6f6d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1e5a34;
        margin: 1rem 0;
        color: #d0e2d4;
        box-shadow: 0 5px 15px rgba(30, 90, 52, 0.3);
    }

    .warning-box {
        background: linear-gradient(135deg, #7a6a3f 0%, #8c5f48 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #b28c2b;
        margin: 1rem 0;
        color: #f3e6b1;
        box-shadow: 0 5px 15px rgba(178, 140, 43, 0.3);
    }

    .error-box {
        background: linear-gradient(135deg, #7a3f3f 0%, #7a2f4f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #a12d3b;
        margin: 1rem 0;
        color: #f5c3c6;
        box-shadow: 0 5px 15px rgba(161, 45, 59, 0.3);
    }

    .stButton > button {
        background: linear-gradient(135deg, #3f4b8c 0%, #4a3d6a 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(50, 60, 100, 0.4);
        position: relative;
        overflow: hidden;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(50, 60, 100, 0.5);
    }

    .stButton > button:active {
        transform: translateY(-1px);
    }

    .document-card {
        background: linear-gradient(135deg, #2f3133 0%, #36383a 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #444;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .document-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(135deg, #3f4b8c 0%, #4a3d6a 100%);
    }

    .document-card:hover {
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        transform: translateY(-5px);
    }

    .chat-tab {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }

    .chat-tab:hover {
        background: linear-gradient(135deg, #2f3133 0%, #3a3d3f 100%);
        border-color: #3f4b8c;
    }

    .chat-tab.active {
        background: linear-gradient(135deg, #3f4b8c 0%, #4a3d6a 100%);
        color: white;
        box-shadow: 0 5px 15px rgba(50, 60, 100, 0.4);
    }

    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }

    .status-connected {
        background: linear-gradient(135deg, #355e4a 0%, #3b5c62 100%);
        color: #b8d8c5;
    }

    .status-disconnected {
        background: linear-gradient(135deg, #6a3f3f 0%, #703a52 100%);
        color: #f3c6ce;
    }

    .hybrid-score {
        background: linear-gradient(135deg, #3f4b8c 0%, #4a3d6a 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin-left: 0.5rem;
    }

    .knowledge-node {
        background: linear-gradient(135deg, #5a6c6b 0%, #6a5b5a 100%);
        padding: 0.75rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 3px solid #4a3d6a;
        font-size: 0.9rem;
        color: #ddd;
    }

    .user-profile-card {
        background: linear-gradient(135deg, #3f4b8c 0%, #4a3d6a 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(50, 60, 100, 0.4);
    }

    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #1e1e1e;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3f4b8c 0%, #4a3d6a 100%);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #4a3d6a 0%, #3f4b8c 100%);
    }

    .stTextInput > div > div > input {
        border-radius: 15px;
        border: 2px solid #444;
        padding: 0.75rem 1rem;
        font-family: 'Inter', sans-serif;
        background: #1f1f1f;
        color: #e6e6e6;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #3f4b8c;
        box-shadow: 0 0 0 3px rgba(63, 75, 140, 0.3);
    }

</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.system = None
    st.session_state.uploaded_files = {}
    st.session_state.processing_status = {}
    st.session_state.chat_sessions = {}
    st.session_state.active_chat_id = None
    st.session_state.new_chat_name = ""
    st.session_state.user_profile = None
    st.session_state.rename_mode = False

# Configuration class
class Config:
    VOYAGE_API_KEY = "pa-tDh9PAJmIfaPahq1-GkuSk8uVNGrI69sq3uxpiGK8Y7"
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
    OLLAMA_MODEL = "llama3.2:1b"
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_AUTH = ("neo4j", "omarnasser")
    CHAT_DB = "chat_history.db"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserProfileManager:
    """Manage user profiles and persistent memory"""
    
    def __init__(self, neo4j_driver):
        self.neo4j_driver = neo4j_driver
        self.current_user = None
    
    def extract_user_info(self, message: str) -> Dict:
        """Extract user information from message using simple patterns"""
        user_info = {}
        
        name_patterns = [
            r"my name is ([A-Za-z\s]+)",
            r"i'm ([A-Za-z\s]+)",
            r"i am ([A-Za-z\s]+)",
            r"call me ([A-Za-z\s]+)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, message.lower())
            if match:
                user_info['name'] = match.group(1).strip().title()
                break
        
        if 'work at' in message.lower() or 'work for' in message.lower():
            work_match = re.search(r"work (?:at|for) ([A-Za-z0-9\s&]+)", message.lower())
            if work_match:
                user_info['company'] = work_match.group(1).strip().title()
        
        if 'interested in' in message.lower():
            interest_match = re.search(r"interested in ([A-Za-z0-9\s,]+)", message.lower())
            if interest_match:
                user_info['interests'] = interest_match.group(1).strip()
        
        return user_info
    
    def create_or_update_user_node(self, user_info: Dict, session_id: str):
        """Create or update user node in Neo4j"""
        if not self.neo4j_driver or not user_info:
            return None
        
        try:
            with self.neo4j_driver.session() as session:
                user_id = user_info.get('name', f'user_{session_id[:8]}')
                
                session.run("""
                MERGE (u:User {id: $user_id})
                SET u.name = coalesce($name, u.name),
                    u.company = coalesce($company, u.company),
                    u.interests = coalesce($interests, u.interests),
                    u.last_active = datetime(),
                    u.session_count = coalesce(u.session_count, 0) + 1
                """, {
                    'user_id': user_id,
                    'name': user_info.get('name'),
                    'company': user_info.get('company'),
                    'interests': user_info.get('interests')
                })
                
                session.run("""
                MATCH (u:User {id: $user_id})
                MERGE (s:ChatSession {id: $session_id})
                MERGE (u)-[:HAS_SESSION]->(s)
                SET s.created_at = coalesce(s.created_at, datetime())
                """, {'user_id': user_id, 'session_id': session_id})
                
                self.current_user = user_id
                return user_id
                
        except Exception as e:
            logger.error(f"Failed to create user node: {str(e)}")
            return None
    
    def get_user_context(self, user_id: str) -> str:
        """Get user context for personalization"""
        if not self.neo4j_driver:
            return ""
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("""
                MATCH (u:User {id: $user_id})
                OPTIONAL MATCH (u)-[:INTERESTED_IN]->(concept:Concept)
                OPTIONAL MATCH (u)-[:HAS_SESSION]->(s:ChatSession)
                RETURN u.name as name, u.company as company, u.interests as interests,
                       count(DISTINCT s) as session_count,
                       collect(DISTINCT concept.name)[..5] as related_concepts
                """, {'user_id': user_id}).single()
                
                if result:
                    context = f"User: {result['name'] or 'Unknown'}"
                    if result['company']:
                        context += f" from {result['company']}"
                    if result['interests']:
                        context += f", interested in: {result['interests']}"
                    if result['related_concepts']:
                        context += f". Previously discussed: {', '.join(result['related_concepts'])}"
                    return context
                    
        except Exception as e:
            logger.error(f"Failed to get user context: {str(e)}")
        
        return ""
    
    def connect_user_to_concepts(self, user_id: str, concepts: List[str]):
        """Connect user to concepts they've interacted with"""
        if not self.neo4j_driver or not concepts:
            return
        
        try:
            with self.neo4j_driver.session() as session:
                for concept in concepts:
                    session.run("""
                    MATCH (u:User {id: $user_id})
                    MATCH (c:Concept {name: $concept})
                    MERGE (u)-[r:INTERESTED_IN]->(c)
                    SET r.interaction_count = coalesce(r.interaction_count, 0) + 1,
                        r.last_interaction = datetime()
                    """, {'user_id': user_id, 'concept': concept})
                    
        except Exception as e:
            logger.error(f"Failed to connect user to concepts: {str(e)}")

class OllamaLLM:
    """Ollama LLM handler optimized for Streamlit"""
    
    def __init__(self, base_url: str = Config.OLLAMA_URL, model: str = Config.OLLAMA_MODEL):
        self.base_url = base_url
        self.chat_url = Config.OLLAMA_CHAT_URL
        self.model = model
        self.is_connected = self.check_connection()
    
    def check_connection(self):
        """Check Ollama connection with progress feedback"""
        try:
            response = requests.get(f"{self.base_url.replace('/api/generate', '')}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model not in model_names and model_names:
                    self.model = model_names[0]
                
                return True
            return False
        except Exception:
            return False
    
    def simple_generate(self, prompt: str, **kwargs) -> str:
        """Generate text with progress indicator"""
        if not self.is_connected:
            return "❌ Ollama not connected. Please ensure Ollama is running."
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.3),
                    "top_p": kwargs.get("top_p", 0.9),
                    "num_ctx": kwargs.get("num_ctx", 2048),
                    "repeat_penalty": kwargs.get("repeat_penalty", 1.1)
                }
            }
            
            with st.spinner("🤖 Generating response..."):
                response = requests.post(self.base_url, json=payload, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "No response from LLM")
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def chat_generate(self, messages: List[Dict], **kwargs) -> str:
        """Chat generation with progress indicator"""
        if not self.is_connected:
            return "❌ Ollama not connected. Please ensure Ollama is running."
        
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "num_ctx": kwargs.get("num_ctx", 4096),
                    "repeat_penalty": kwargs.get("repeat_penalty", 1.1)
                }
            }
            
            with st.spinner("💭 Thinking..."):
                response = requests.post(self.chat_url, json=payload, timeout=90)
                response.raise_for_status()
                
                result = response.json()
                return result.get("message", {}).get("content", "No response from LLM")
                
        except Exception as e:
            return f"Error: {str(e)}"

class ChatDatabase:
    """SQLite database for chat history with user profiles"""
    import os
    print(">>> Using DB at:", os.path.abspath(Config.CHAT_DB))

    def __init__(self, db_path=Config.CHAT_DB):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id TEXT PRIMARY KEY,
                name TEXT,
                company TEXT,
                interests TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                name TEXT,
                user_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES user_profiles(id)
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                user_info_extracted TEXT,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
            )
            """)
            conn.commit()
    
    def create_or_update_user(self, user_info: Dict) -> str:
        """Create or update user profile"""
        user_id = user_info.get('name', f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO user_profiles (id, name, company, interests, last_active)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (user_id, user_info.get('name'), user_info.get('company'), user_info.get('interests')))
            conn.commit()
        
        return user_id
    
    def create_chat_session(self, session_id, name="New Chat", user_id=None):
        """Create a new chat session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO chat_sessions (id, name, user_id)
            VALUES (?, ?, ?)
            """, (session_id, name, user_id))
            conn.commit()
    
    def add_message(self, session_id, role, content, metadata=None, user_info_extracted=None):
        """Add a message to a chat session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO chat_messages (session_id, role, content, metadata, user_info_extracted)
            VALUES (?, ?, ?, ?, ?)
            """, (session_id, role, content, 
                 json.dumps(metadata) if metadata else None,
                 json.dumps(user_info_extracted) if user_info_extracted else None))
            
            cursor.execute("""
            UPDATE chat_sessions
            SET last_used = CURRENT_TIMESTAMP
            WHERE id = ?
            """, (session_id,))
            conn.commit()
    
    def rename_chat_session(self, session_id, new_name):
        """Rename a chat session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE chat_sessions
            SET name = ?
            WHERE id = ?
            """, (new_name, session_id))
            conn.commit()
    
    def delete_chat_session(self, session_id):
        """Delete a chat session and its messages"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            conn.commit()
    
    def get_chat_sessions(self):
        """Get all chat sessions ordered by last_used"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT cs.id, cs.name, cs.created_at, cs.last_used, up.name as user_name
            FROM chat_sessions cs
            LEFT JOIN user_profiles up ON cs.user_id = up.id
            ORDER BY cs.last_used DESC
            """)
            return cursor.fetchall()
    
    def get_chat_messages(self, session_id):
        """Get all messages for a chat session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT role, content, timestamp, metadata, user_info_extracted
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
            """, (session_id,))
            return cursor.fetchall()

class HybridSearchEngine:
    """Enhanced hybrid search combining vector similarity and graph traversal"""
    
    def __init__(self, faiss_index, metadata, neo4j_driver, voyage_client):
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.neo4j_driver = neo4j_driver
        self.voyage_client = voyage_client
        self.alpha = 0.7
    
    def search(self, query: str, top_k: int = 5, alpha: float = None) -> List[Dict]:
        """Hybrid search with configurable weighting"""
        if alpha is not None:
            self.alpha = alpha
        
        vector_results = self._vector_search(query, top_k * 2)
        graph_results = self._graph_search(query, top_k * 2)
        combined_results = self._combine_results(vector_results, graph_results, self.alpha)
        
        return combined_results[:top_k]
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform vector similarity search"""
        if not self.faiss_index or not self.voyage_client:
            return []
        
        try:
            query_embedding = self.voyage_client.multimodal_embed(
                inputs=[[query]],
                model="voyage-multimodal-3",
                input_type="document"
            ).embeddings[0]
            
            query_embedding = np.array([query_embedding]).astype("float32")
            distances, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.metadata)))
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata):
                    similarity = 1 / (1 + distance)
                    results.append({
                        "text": self.metadata[idx],
                        "vector_score": float(similarity),
                        "rank": i + 1,
                        "source": "vector",
                        "method": "FAISS"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []
    
    def _graph_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform graph-based search"""
        if not self.neo4j_driver:
            return []
        
        try:
            query_embedding = self.voyage_client.multimodal_embed(
                inputs=[[query]],
                model="voyage-multimodal-3", 
                input_type="document"
            ).embeddings[0]
            
            with self.neo4j_driver.session() as session:
                results = session.run("""
                MATCH (c:Concept)
                WHERE c.embedding IS NOT NULL
                WITH c, gds.similarity.cosine(c.embedding, $query_embedding) AS concept_similarity
                WHERE concept_similarity > 0.3
                ORDER BY concept_similarity DESC
                LIMIT 10
                
                MATCH (c)-[:RELATES*1..2]-(related)
                MATCH (doc:Document)
                WHERE doc.text CONTAINS c.name OR doc.text CONTAINS related.name
                
                WITH doc, c, concept_similarity, related,
                     gds.similarity.cosine(doc.embedding, $query_embedding) AS doc_similarity
                WHERE doc_similarity > 0.2
                
                RETURN DISTINCT doc.text as text, 
                       doc_similarity,
                       concept_similarity,
                       doc.filename as source,
                       c.name as matched_concept,
                       collect(DISTINCT related.name)[..3] as related_concepts,
                       (doc_similarity * 0.6 + concept_similarity * 0.4) as graph_score
                ORDER BY graph_score DESC
                LIMIT $top_k
                """, {
                    "query_embedding": query_embedding,
                    "top_k": top_k
                }).data()
                
                graph_results = []
                for i, record in enumerate(results):
                    graph_results.append({
                        "text": record["text"],
                        "graph_score": float(record["graph_score"]),
                        "concept_match": record["matched_concept"],
                        "related_concepts": record["related_concepts"],
                        "rank": i + 1,
                        "source": record["source"],
                        "method": "Graph"
                    })
                
                return graph_results
                
        except Exception as e:
            logger.error(f"Graph search failed: {str(e)}")
            return []
    
    def _combine_results(self, vector_results: List[Dict], graph_results: List[Dict], alpha: float) -> List[Dict]:
        """Combine vector and graph search results with hybrid scoring"""
        combined = {}
        
        for result in vector_results:
            text = result["text"]
            combined[text] = {
                **result,
                "vector_score": result.get("vector_score", 0),
                "graph_score": 0,
                "hybrid_score": 0,
                "methods": ["Vector"],
                "source": result.get("source", "Unknown"),
                "concept_match": result.get("concept_match", ""),
                "related_concepts": result.get("related_concepts", [])
            }
        
        for result in graph_results:
            text = result["text"]
            if text in combined:
                combined[text]["graph_score"] = result.get("graph_score", 0)
                combined[text]["methods"].append("Graph")
                combined[text]["concept_match"] = result.get("concept_match", combined[text]["concept_match"])
                combined[text]["related_concepts"] = list(set(combined[text]["related_concepts"] + result.get("related_concepts", [])))
            else:
                combined[text] = {
                    **result,
                    "vector_score": 0,
                    "graph_score": result.get("graph_score", 0),
                    "hybrid_score": 0,
                    "methods": ["Graph"],
                    "source": result.get("source", "Unknown"),
                    "concept_match": result.get("concept_match", ""),
                    "related_concepts": result.get("related_concepts", [])
                }
        
        for text, result in combined.items():
            vector_score = min(1.0, max(0.0, result["vector_score"]))
            graph_score = min(1.0, max(0.0, result["graph_score"]))
            hybrid_score = (alpha * vector_score) + ((1 - alpha) * graph_score)
            if len(result["methods"]) > 1:
                hybrid_score *= 1.2
            result["hybrid_score"] = hybrid_score
        
        sorted_results = sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)
        for i, result in enumerate(sorted_results):
            result["rank"] = i + 1
            
        return sorted_results

class StreamlitGraphRAGSystem:
    """Enhanced GraphRAG system with hybrid search and user memory"""
    
    def __init__(self):
        self.faiss_index = None
        self.metadata = []
        self.neo4j_driver = None
        self.vector_index_created = False
        self.knowledge_graph_created = False
        self.conversation_history = []
        self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.llm = OllamaLLM()
        self.client = None
        self.chat_db = ChatDatabase()
        self.user_manager = None
        self.hybrid_search = None
        self.setup_voyage_client()
    
    def setup_voyage_client(self):
        """Setup Voyage AI client"""
        try:
            voyageai.api_key = Config.VOYAGE_API_KEY
            self.client = voyageai.Client()
            return True
        except Exception as e:
            st.error(f"Failed to setup Voyage AI: {str(e)}")
            return False
    
    def connect_neo4j(self):
        """Connect to Neo4j with progress feedback"""
        try:
            with st.spinner("🔗 Connecting to Neo4j..."):
                self.neo4j_driver = GraphDatabase.driver(Config.NEO4J_URI, auth=Config.NEO4J_AUTH)
                with self.neo4j_driver.session() as session:
                    result = session.run("RETURN 1 AS test")
                    if result.single()["test"] == 1:
                        self.create_neo4j_schema()
                        self.user_manager = UserProfileManager(self.neo4j_driver)
                        self._setup_hybrid_search()
                        return True
            return False
        except Exception as e:
            st.error(f"Neo4j connection failed: {str(e)}")
            return False
    
    def _setup_hybrid_search(self):
        """Setup hybrid search engine"""
        if self.client and self.neo4j_driver:
            self.hybrid_search = HybridSearchEngine(
                self.faiss_index, self.metadata, self.neo4j_driver, self.client
            )
    
    def create_neo4j_schema(self):
        """Create Neo4j schema with enhanced indexes"""
        try:
            with self.neo4j_driver.session() as session:
                vector_indexes = [
                    """
                    CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
                    FOR (d:Document) ON (d.embedding)
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 1024,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                    """,
                    """
                    CREATE VECTOR INDEX concept_embeddings IF NOT EXISTS
                    FOR (c:Concept) ON (c.embedding)
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 1024,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                    """
                ]
                
                for index in vector_indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        logger.error(f"Failed to create vector index: {str(e)}")
                
                self.vector_index_created = True
                
                constraints_and_indexes = [
                    "CREATE CONSTRAINT unique_concept IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
                    "CREATE CONSTRAINT unique_entity IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
                    "CREATE CONSTRAINT unique_document IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                    "CREATE CONSTRAINT unique_user IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
                    "CREATE INDEX concept_importance IF NOT EXISTS FOR (c:Concept) ON (c.importance)",
                    "CREATE INDEX document_filename IF NOT EXISTS FOR (d:Document) ON (d.filename)",
                    "CREATE INDEX user_last_active IF NOT EXISTS FOR (u:User) ON (u.last_active)"
                ]
                
                for constraint in constraints_and_indexes:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        logger.error(f"Failed to create constraint/index: {str(e)}")
                
                self.knowledge_graph_created = True
                
        except Exception as e:
            logger.error(f"Schema creation failed: {str(e)}")
    
    def get_file_hash(self, file_content: bytes) -> str:
        """Generate hash for file content"""
        return hashlib.md5(file_content).hexdigest()
    


    def process_pdf(self, file_content: bytes, filename: str) -> List[Union[str, Image.Image]]:
        """Process PDF with OCR for tables and structured table extraction"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            doc = fitz.open(tmp_file_path)
            pdf_text = ""
            pdf_images = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_pages = len(doc)
            
            for i, page in enumerate(doc):
                progress = (i + 1) / total_pages
                progress_bar.progress(progress)
                status_text.text(f"Processing page {i + 1} of {total_pages}")
                
                # Try text extraction first
                text = page.get_text("text").lower()
                if not text.strip() or "function code" in text.lower():
                    logger.warning(f"Attempting OCR for page {i + 1} of {filename} (possible table)")
                    try:
                        pix = page.get_pixmap(dpi=300)  # Higher DPI for table clarity
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        text = pytesseract.image_to_string(img, config='--psm 6').lower()  # PSM 6 for block text/tables
                        # Preprocess table content
                        text = self._preprocess_table_text(text)
                    except Exception as e:
                        logger.error(f"OCR failed for page {i + 1} of {filename}: {str(e)}")
                        text = ""
                
                pdf_text += text + f"\n[Page {i + 1}]\n"
                
                # Extract images (limit to 5 total)
                if len(pdf_images) < 5:
                    image_list = page.get_images(full=True)
                    for img in image_list[:2]:
                        if len(pdf_images) >= 5:
                            break
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            pdf_images.append(Image.open(io.BytesIO(image_bytes)).resize((256, 256)))
                        except Exception as e:
                            logger.warning(f"Image extraction failed for page {i + 1}: {str(e)}")
                            continue
            
            doc.close()
            os.unlink(tmp_file_path)
            
            progress_bar.progress(1.0)
            status_text.text("✅ PDF processing complete!")
            
            return [pdf_text, *pdf_images] if pdf_images else [pdf_text]
            
        except Exception as e:
            st.error(f"Failed to process PDF {filename}: {str(e)}")
            logger.error(f"PDF processing failed: {str(e)}")
            return []

    def _preprocess_table_text(self, text: str) -> str:
        """Preprocess table text to structure function code data"""
        # Normalize whitespace and split lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        structured_text = ""
        
        # Detect function code pattern (e.g., P00.13)
        func_code_pattern = r"p\d{2}\.\d{2}"
        current_code = None
        
        for line in lines:
            # Match function code
            code_match = re.search(func_code_pattern, line, re.IGNORECASE)
            if code_match:
                current_code = code_match.group(0).upper()
                structured_text += f"\nFunction Code {current_code}:\n"
            # Add line content under the current function code
            if current_code:
                structured_text += f"{line}\n"
            else:
                structured_text += f"{line}\n"
        
        # Normalize function code references
        structured_text = re.sub(func_code_pattern, lambda m: m.group(0).upper(), structured_text, flags=re.IGNORECASE)
        structured_text = re.sub(r"\s+", " ", structured_text).strip()
        return structured_text
    
    def chunk_document(self, text: str, chunk_size: int = 200) -> List[str]:
        """Split document into smaller chunks for technical content"""
        words = text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        return [chunk for chunk in chunks if len(chunk.strip()) >= 50]  # Skip short chunks
    
    def extract_entities_and_concepts(self, text: str) -> Dict:
        """Extract function codes and relationships from VFD data sheet"""
        prompt = f"""Extract key concepts, entities, and relationships from the text, focusing on VFD function codes (e.g., P00.13), technical terms, and their descriptions.

    Text: {text[:2000]}...

    Return ONLY valid JSON:
    {{
        "concepts": [{{"name": "concept", "description": "desc", "importance": 0.8, "type": "technical"}}],
        "entities": [{{"name": "entity", "description": "desc", "type": "function_code"}}],
        "relationships": [{{"source": "source", "target": "target", "type": "defines", "strength": 0.7}}]
    }}"""
        
        try:
            response = self.llm.simple_generate(prompt, temperature=0.1, num_ctx=3000)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                parsed_data.setdefault('concepts', [])
                parsed_data.setdefault('entities', [])
                parsed_data.setdefault('relationships', [])
                
                # Add function code entities
                func_code_pattern = r"P\d{2}\.\d{2}"
                for match in re.finditer(func_code_pattern, text, re.IGNORECASE):
                    code = match.group(0).upper()
                    # Find description (next few lines after the code)
                    start_idx = match.start()
                    desc = text[start_idx:start_idx+200].split('\n')[1:3]
                    desc = ' '.join(line.strip() for line in desc if line.strip())
                    parsed_data['entities'].append({
                        "name": code,
                        "description": desc or f"Function code {code}",
                        "type": "function_code"
                    })
                    parsed_data['relationships'].append({
                        "source": code,
                        "target": "VFD Configuration",
                        "type": "defines",
                        "strength": 0.8
                    })
                
                return parsed_data
            return {"concepts": [], "entities": [], "relationships": []}
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return {"concepts": [], "entities": [], "relationships": []}
        
    def create_faiss_index(self, documents: List[List[Union[str, Image.Image]]], filename: str):
        """Create FAISS index for documents with table content"""
        if not self.client:
            st.error("Voyage AI client not initialized")
            return None, []
        
        try:
            processed_docs = []
            all_text = ""
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, doc in enumerate(documents):
                status_text.text(f"Processing document {i + 1}/{len(documents)}...")
                
                text = doc[0]
                images = doc[1:] if len(doc) > 1 else []
                all_text += text + " "
                
                chunks = self.chunk_document(text, chunk_size=200)
                for j, chunk in enumerate(chunks):
                    # Include images for the first 5 chunks
                    if j < 5 and images:
                        processed_docs.append([chunk, *images[:2]])
                    else:
                        processed_docs.append([chunk])
                
                progress_bar.progress((i + 1) / len(documents))
            
            status_text.text("🧠 Extracting knowledge graph...")
            
            knowledge = self.extract_entities_and_concepts(all_text)
            self.store_knowledge_graph(knowledge, all_text, filename)
            
            status_text.text("🔄 Creating embeddings...")
            
            response = self.client.multimodal_embed(
                inputs=processed_docs,
                model="voyage-multimodal-3",
                input_type="document"
            )
            
            embeddings = np.array(response.embeddings).astype("float32")
            dimension = embeddings.shape[1]
            
            if self.faiss_index is None:
                self.faiss_index = faiss.IndexFlatL2(dimension)
                self.metadata = []
            
            self.faiss_index.add(embeddings)
            self.metadata.extend([f"[{filename}] {doc[0]}" for doc in processed_docs])
            
            self.store_documents_neo4j(processed_docs, response.embeddings, filename)
            self._setup_hybrid_search()
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Indexed {len(processed_docs)} chunks from {filename}!")
            logger.info(f"Added {len(embeddings)} vectors to FAISS index, total: {self.faiss_index.ntotal}")
            
            return response, processed_docs
            
        except Exception as e:
            st.error(f"Failed to create index for {filename}: {str(e)}")
            logger.error(f"FAISS index creation failed: {str(e)}")
            return None, []
                
        except Exception as e:
            st.error(f"Failed to create index for {filename}: {str(e)}")
            logger.error(f"FAISS index creation failed: {str(e)}")
            return None, []
    
    def store_knowledge_graph(self, knowledge: Dict, source_text: str, filename: str):
        """Store knowledge graph in Neo4j with enhanced relationships"""
        if not self.neo4j_driver or not self.knowledge_graph_created:
            return
            
        try:
            with self.neo4j_driver.session() as session:
                for concept in knowledge.get('concepts', []):
                    try:
                        concept_text = concept['name'] + " " + concept.get('description', '')
                        concept_embedding = self.client.multimodal_embed(
                            inputs=[[concept_text]],
                            model="voyage-multimodal-3",
                            input_type="document"
                        ).embeddings[0]
                        
                        session.run("""
                        MERGE (c:Concept {name: $name})
                        SET c.description = $description,
                            c.importance = $importance,
                            c.type = $type,
                            c.embedding = $embedding,
                            c.source_document = $source,
                            c.last_mentioned = datetime(),
                            c.mention_count = coalesce(c.mention_count, 0) + 1,
                            c.avg_importance = (coalesce(c.avg_importance, 0) * coalesce(c.mention_count, 1) + $importance) / (coalesce(c.mention_count, 0) + 1)
                        """, {**concept, 'embedding': concept_embedding, 'source': filename})
                        
                    except Exception as e:
                        logger.error(f"Failed to store concept {concept.get('name', 'Unknown')}: {str(e)}")
                        continue
                
                for entity in knowledge.get('entities', []):
                    session.run("""
                    MERGE (e:Entity {name: $name})
                    SET e.description = $description,
                        e.type = $type,
                        e.source_document = $source,
                        e.last_mentioned = datetime(),
                        e.mention_count = coalesce(e.mention_count, 0) + 1
                    """, {**entity, 'source': filename})
                
                for rel in knowledge.get('relationships', []):
                    session.run("""
                    MATCH (source) WHERE source.name = $source
                    MATCH (target) WHERE target.name = $target
                    MERGE (source)-[r:RELATES {type: $type}]->(target)
                    SET r.strength = $strength,
                        r.source_document = $source_doc,
                        r.last_used = datetime(),
                        r.usage_count = coalesce(r.usage_count, 0) + 1,
                        r.avg_strength = (coalesce(r.avg_strength, 0) * coalesce(r.usage_count, 1) + $strength) / (coalesce(r.usage_count, 0) + 1)
                    """, {**rel, 'source_doc': filename})
                    
                    session.run("""
                    MATCH (source) WHERE source.name = $target
                    MATCH (target) WHERE target.name = $source
                    MERGE (source)-[r:RELATES {type: $reverse_type}]->(target)
                    SET r.strength = $strength,
                        r.source_document = $source_doc,
                        r.last_used = datetime(),
                        r.usage_count = coalesce(r.usage_count, 0) + 1
                    """, {
                        'source': rel['target'],
                        'target': rel['source'],
                        'reverse_type': f"reverse_{rel['type']}",
                        'strength': rel['strength'],
                        'source_doc': filename
                    })
                
        except Exception as e:
            logger.error(f"Failed to store knowledge graph: {str(e)}")
    
    def store_documents_neo4j(self, documents: List, embeddings: List, filename: str):
        """Store documents in Neo4j"""
        if not self.neo4j_driver or not self.vector_index_created:
            return
            
        try:
            with self.neo4j_driver.session() as session:
                for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                    doc_id = f"{self.get_file_hash(filename.encode())}_{i}"
                    session.run("""
                    MERGE (d:Document {id: $id})
                    SET d.text = $text,
                        d.embedding = $embedding,
                        d.has_image = $has_image,
                        d.source = $source,
                        d.filename = $filename,
                        d.chunk_index = $chunk_index,
                        d.timestamp = datetime(),
                        d.word_count = size(split($text, ' '))
                    """, {
                        "id": doc_id,
                        "text": doc[0],
                        "embedding": embedding,
                        "has_image": len(doc) > 1,
                        "source": filename,
                        "filename": filename,
                        "chunk_index": i
                    })
        except Exception as e:
            logger.error(f"Failed to store documents: {str(e)}")
    
    def search_documents(self, query: str, top_k: int = 3, user_id: str = None) -> Dict:
        """Enhanced search with user context and hybrid approach"""
        if not self.client:
            return {"results": [], "graph_context": [], "user_context": ""}
        
        try:
            user_context = ""
            if user_id and self.user_manager:
                user_context = self.user_manager.get_user_context(user_id)
            
            if self.hybrid_search:
                enhanced_query = f"{query} {user_context}".strip()
                results = self.hybrid_search.search(enhanced_query, top_k)
                
                if user_id and self.user_manager:
                    concepts = []
                    for result in results[:3]:
                        if result.get("concept_match"):
                            concepts.append(result["concept_match"])
                    self.user_manager.connect_user_to_concepts(user_id, concepts)
                
                return {
                    "results": results,
                    "graph_context": self.get_enhanced_graph_context(query, top_k),
                    "user_context": user_context,
                    "search_method": "Hybrid"
                }
            
            graph_context = self.get_graph_context(query, top_k)
            context_terms = []
            for context in graph_context:
                context_terms.append(context['concept'])
                for item in context.get('related_items', []):
                    context_terms.append(item['name'])
            
            enhanced_query = f"{query} {' '.join(context_terms[:10])} {user_context}".strip()
            
            question_embedding = self.client.multimodal_embed(
                inputs=[[enhanced_query]],
                model="voyage-multimodal-3",
                input_type="document"
            ).embeddings[0]
            
            faiss_results = []
            if self.faiss_index:
                query_embedding = np.array([question_embedding]).astype("float32")
                distances, indices = self.faiss_index.search(query_embedding, top_k)
                
                faiss_results = [
                    {
                        "text": self.metadata[idx],
                        "hybrid_score": float(1 / (1 + distances[0][i])),
                        "source": "FAISS",
                        "method": "Vector",
                        "methods": ["Vector"]
                    }
                    for i, idx in enumerate(indices[0]) if idx < len(self.metadata)
                ]
            
            return {
                "results": faiss_results,
                "graph_context": graph_context,
                "user_context": user_context,
                "enhanced_query": enhanced_query,
                "search_method": "Fallback"
            }
            
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            return {"results": [], "graph_context": [], "user_context": ""}
    
    def get_enhanced_graph_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Get enhanced graph context with multi-hop traversal"""
        if not self.neo4j_driver or not self.knowledge_graph_created:
            return []
            
        try:
            query_embedding = self.client.multimodal_embed(
                inputs=[[query]],
                model="voyage-multimodal-3",
                input_type="document"
            ).embeddings[0]
            
            with self.neo4j_driver.session() as session:
                results = session.run("""
                CALL db.index.vector.queryNodes('concept_embeddings', $top_k, $embedding)
                YIELD node as concept, score as concept_score
                
                MATCH (concept)-[r1:RELATES*1..3]-(related)
                WHERE related:Concept OR related:Entity
                
                WITH concept, concept_score, related, 
                     reduce(strength = 1.0, rel in r1 | strength * rel.strength) as path_strength,
                     size(r1) as hop_distance
                
                WITH concept, concept_score, related, path_strength,hop_distance, 
                     path_strength * (1.0 / hop_distance) as weighted_strength
                
                RETURN concept.name as main_concept,
                       concept.description as main_description,
                       concept_score,
                       collect(DISTINCT {
                           name: related.name, 
                           type: labels(related)[0],
                           strength: weighted_strength,
                           hop_distance: hop_distance
                       }) as related_items
                ORDER BY concept_score DESC
                LIMIT $top_k
                """, {"top_k": top_k, "embedding": query_embedding}).data()
                
                enhanced_results = []
                for record in results:
                    related_items = sorted(
                        record["related_items"], 
                        key=lambda x: x["strength"], 
                        reverse=True
                    )[:10]
                    
                    enhanced_results.append({
                        "concept": record["main_concept"],
                        "description": record["main_description"],
                        "score": float(record["concept_score"]),
                        "related_items": related_items
                    })
                
                return enhanced_results
                    
        except Exception as e:
            logger.error(f"Enhanced graph context failed: {str(e)}")
            return []
    
    def get_graph_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Get graph context for query (fallback method)"""
        if not self.neo4j_driver or not self.knowledge_graph_created:
            return []
            
        try:
            query_embedding = self.client.multimodal_embed(
                inputs=[[query]],
                model="voyage-multimodal-3",
                input_type="document"
            ).embeddings[0]
            
            with self.neo4j_driver.session() as session:
                results = session.run("""
                MATCH (c:Concept)
                WHERE c.embedding IS NOT NULL
                WITH c, gds.similarity.cosine(c.embedding, $embedding) AS similarity
                WHERE similarity > 0.3
                
                MATCH (c)-[r:RELATES]-(related)
                RETURN c.name as concept, 
                       c.description as description,
                       similarity as score,
                       collect(DISTINCT {name: related.name, type: labels(related)[0], relationship: r.type}) as related_items
                ORDER BY similarity DESC
                LIMIT $top_k
                """, {"top_k": top_k, "embedding": query_embedding}).data()
                
                return results
                    
        except Exception as e:
            logger.error(f"Graph context failed: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context: List[Dict], graph_context: List[Dict] = None, user_context: str = "") -> str:
        """Generate personalized answer using LLM"""
        try:
            user_info = f"\n### User Context:\n{user_context}\n" if user_context else ""
            
            graph_info = ""
            if graph_context:
                graph_info = "\n### Knowledge Graph Context:\n"
                for ctx in graph_context[:3]:
                    graph_info += f"- **{ctx['concept']}**: {ctx.get('description', 'No description')}\n"
                    if ctx.get('related_items'):
                        related = [f"{item['name']} (strength: {item.get('strength', 0):.2f})" 
                                 for item in ctx['related_items'][:3]]
                        graph_info += f"  Related: {', '.join(related)}\n"
            
            context_text = ""
            if context:
                context_text = "\n### Document Context:\n"
                for i, res in enumerate(context[:3], 1):
                    methods = ' + '.join(res.get('methods', ['Unknown']))
                    hybrid_score = res.get('hybrid_score', res.get('vector_score', 0))
                    context_text += f"{i}. [Hybrid Score: {hybrid_score:.3f}] [Methods: {methods}] [Source: {res.get('source', 'Unknown')}]\n"
                    context_text += f"{res['text'][:500]}...\n\n"
                    if res.get('concept_match'):
                        context_text += f"   🧠 Concept Match: {res['concept_match']}\n"
                    if res.get('related_concepts'):
                        context_text += f"   🔗 Related: {', '.join(res['related_concepts'])}\n\n"
            
            system_message = f"""You are a helpful technical assistant with access to document content and knowledge graph. 
Use the provided context to give comprehensive, accurate answers that are personalized for the user.
Be specific and reference technical details when relevant. If you know the user's name or background, acknowledge it naturally.

{user_info}
{graph_info}
{context_text}

Instructions:
- Provide detailed, technical answers when appropriate
- Reference specific document sections and concepts when relevant
- If you recognize the user, personalize your response accordingly
- Highlight connections between different concepts and documents
- Use the hybrid search scores to prioritize information"""
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
            
            return self.llm.chat_generate(messages, temperature=0.7, num_ctx=4096)
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def process_user_message(self, message: str, session_id: str) -> Dict:
        """Process user message and extract information"""
        user_info = {}
        
        if self.user_manager:
            user_info = self.user_manager.extract_user_info(message)
            
            if user_info:
                user_id = self.user_manager.create_or_update_user_node(user_info, session_id)
                self.chat_db.create_or_update_user(user_info)
                
                return {
                    "user_info": user_info,
                    "user_id": user_id,
                    "has_user_info": True
                }
        
        return {
            "user_info": {},
            "user_id": None,
            "has_user_info": False
        }
    
    def get_system_stats(self) -> Dict:
        """Get enhanced system statistics"""
        stats = {
            "documents": 0,
            "concepts": 0,
            "entities": 0,
            "relationships": 0,
            "users": 0,
            "faiss_vectors": len(self.metadata) if self.metadata else 0,
            "avg_concept_importance": 0,
            "most_connected_concept": "None"
        }
        
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    stats["documents"] = session.run("MATCH (d:Document) RETURN count(d) as count").single()["count"]
                    stats["concepts"] = session.run("MATCH (c:Concept) RETURN count(c) as count").single()["count"]
                    stats["entities"] = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
                    stats["relationships"] = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) as count").single()["count"]
                    stats["users"] = session.run("MATCH (u:User) RETURN count(u) as count").single()["count"]
                    
                    try:
                        avg_importance = session.run("""
                        MATCH (c:Concept)
                        WHERE c.avg_importance IS NOT NULL
                        RETURN avg(c.avg_importance) as avg_importance
                        """).single()
                        stats["avg_concept_importance"] = float(avg_importance["avg_importance"] or 0)
                    except:
                        pass
                    
                    try:
                        most_connected = session.run("""
                        MATCH (c:Concept)-[r:RELATES]-()
                        RETURN c.name as concept, count(r) as connections
                        ORDER BY connections DESC
                        LIMIT 1
                        """).single()
                        if most_connected:
                            stats["most_connected_concept"] = f"{most_connected['concept']} ({most_connected['connections']} connections)"
                    except:
                        pass
                        
            except Exception as e:
                logger.error(f"Failed to get system stats: {str(e)}")
        
        return stats
    
    def cleanup_old_data(self, days_threshold: int = 30):
        """Clean up old data from Neo4j and SQLite"""
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    session.run("""
                    MATCH (u:User)-[r:HAS_SESSION]->(s:ChatSession)
                    WHERE s.created_at < datetime() - duration({days: $days})
                    DELETE r, s
                    """, {"days": days_threshold})
                    
                    session.run("""
                    MATCH (u:User)
                    WHERE NOT (u)-[:HAS_SESSION]->()
                    AND u.last_active < datetime() - duration({days: $days})
                    DELETE u
                    """, {"days": days_threshold})
                    
            except Exception as e:
                logger.error(f"Failed to cleanup Neo4j data: {str(e)}")
        
        with sqlite3.connect(self.chat_db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            DELETE FROM chat_messages
            WHERE session_id IN (
                SELECT id FROM chat_sessions 
                WHERE created_at < datetime('now', '-' || ? || ' days')
            )
            """, (days_threshold,))
            cursor.execute("""
            DELETE FROM chat_sessions
            WHERE created_at < datetime('now', '-' || ? || ' days')
            """, (days_threshold,))
            conn.commit()
    
    def export_knowledge_graph(self, format: str = "cypher") -> str:
        """Export knowledge graph in specified format"""
        if not self.neo4j_driver:
            return ""
            
        try:
            with self.neo4j_driver.session() as session:
                if format.lower() == "cypher":
                    result = session.run("""
                    MATCH (n)
                    OPTIONAL MATCH (n)-[r]->(m)
                    RETURN n, r, m
                    LIMIT 1000
                    """)
                    
                    cypher_export = []
                    nodes_seen = set()
                    
                    for record in result:
                        for node in [record["n"], record["m"]]:
                            if node and node.id not in nodes_seen:
                                labels = ":".join(node.labels)
                                props = ", ".join([f"{k}: ${k}" for k in node.keys()])
                                cypher_export.append(f"CREATE (:{labels} {{{props}}})")
                                nodes_seen.add(node.id)
                        
                        if record["r"]:
                            rel_type = record["r"].type
                            start_id = record["n"].id
                            end_id = record["m"].id
                            props = ", ".join([f"{k}: ${k}" for k in record["r"].keys()])
                            cypher_export.append(
                                f"MATCH (a), (b) WHERE id(a) = {start_id} AND id(b) = {end_id} "
                                f"CREATE (a)-[:{rel_type} {{{props}}}]->(b)"
                            )
                    
                    return "\n".join(cypher_export)
                
                elif format.lower() == "json":
                    result = session.run("""
                    MATCH (n)
                    OPTIONAL MATCH (n)-[r]->(m)
                    RETURN n, collect({relationship: r, target: m}) as relationships
                    LIMIT 1000
                    """)
                    
                    json_export = {"nodes": [], "relationships": []}
                    nodes_seen = set()
                    
                    for record in result:
                        node = record["n"]
                        if node.id not in nodes_seen:
                            json_export["nodes"].append({
                                "id": node.id,
                                "labels": list(node.labels),
                                "properties": dict(node)
                            })
                            nodes_seen.add(node.id)
                        
                        for rel in record["relationships"]:
                            if rel["relationship"] and rel["target"]:
                                json_export["relationships"].append({
                                    "source": node.id,
                                    "target": rel["target"].id,
                                    "type": rel["relationship"].type,
                                    "properties": dict(rel["relationship"])
                                })
                    
                    return json.dumps(json_export, indent=2)
                
                else:
                    return "Unsupported export format"
                    
        except Exception as e:
            logger.error(f"Failed to export knowledge graph: {str(e)}")
            return ""
    
    def get_document_stats(self, filename: str) -> Dict:
        """Get detailed statistics for a specific document"""
        stats = {
            "filename": filename,
            "chunks": 0,
            "word_count": 0,
            "concepts": 0,
            "entities": 0,
            "last_accessed": None,
            "created_at": None
        }
        
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    result = session.run("""
                    MATCH (d:Document {filename: $filename})
                    OPTIONAL MATCH (d)-[:CONTAINS]->(c:Concept)
                    OPTIONAL MATCH (d)-[:MENTIONS]->(e:Entity)
                    RETURN count(d) as chunks,
                           sum(d.word_count) as word_count,
                           count(DISTINCT c) as concepts,
                           count(DISTINCT e) as entities,
                           max(d.timestamp) as last_accessed,
                           min(d.timestamp) as created_at
                    """, {"filename": filename}).single()
                    
                    if result:
                        stats.update({
                            "chunks": result["chunks"],
                            "word_count": result["word_count"],
                            "concepts": result["concepts"],
                            "entities": result["entities"],
                            "last_accessed": result["last_accessed"],
                            "created_at": result["created_at"]
                        })
            except Exception as e:
                logger.error(f"Failed to get document stats: {str(e)}")
        
        return stats

def render_header():
    """Render enhanced main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🧠 AItoMate</h1>
        <p>Intelligent document analysis with hybrid search, knowledge graphs, and persistent user memory</p>
    </div>
    """, unsafe_allow_html=True)

def render_user_profile():
    """Render user profile section"""
    if st.session_state.system and st.session_state.system.user_manager and st.session_state.system.user_manager.current_user:
        user_context = st.session_state.system.user_manager.get_user_context(st.session_state.system.user_manager.current_user)
        if user_context:
            st.sidebar.markdown(f"""
            <div class="user-profile-card">
                <h4>👤 User Profile</h4>
                <p>{user_context}</p>
            </div>
            """, unsafe_allow_html=True)

def render_sidebar():
    """Render enhanced sidebar with user profile and system stats"""
    render_user_profile()
    
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h3>🚀 System Status</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.system:
        ollama_class = "status-connected" if st.session_state.system.llm.is_connected else "status-disconnected"
        ollama_icon = "✅" if st.session_state.system.llm.is_connected else "❌"
        st.sidebar.markdown(f"""
        <div class="status-indicator {ollama_class}">
            {ollama_icon} Ollama LLM
        </div>
        """, unsafe_allow_html=True)
        
        neo4j_class = "status-connected" if st.session_state.system.neo4j_driver else "status-disconnected"
        neo4j_icon = "✅" if st.session_state.system.neo4j_driver else "❌"
        st.sidebar.markdown(f"""
        <div class="status-indicator {neo4j_class}">
            {neo4j_icon} Neo4j Graph DB
        </div>
        """, unsafe_allow_html=True)
        
        voyage_class = "status-connected" if st.session_state.system.client else "status-disconnected"
        voyage_icon = "✅" if st.session_state.system.client else "❌"
        st.sidebar.markdown(f"""
        <div class="status-indicator {voyage_class}">
            {voyage_icon} Voyage AI
        </div>
        """, unsafe_allow_html=True)
        
        hybrid_class = "status-connected" if st.session_state.system.hybrid_search else "status-disconnected"
        hybrid_icon = "✅" if st.session_state.system.hybrid_search else "❌"
        st.sidebar.markdown(f"""
        <div class="status-indicator {hybrid_class}">
            {hybrid_icon} Hybrid Search
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h3>📊 Enhanced Statistics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.system:
        stats = st.session_state.system.get_system_stats()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Documents", stats["documents"])
            st.metric("Concepts", stats["concepts"])
            st.metric("Users", stats["users"])
        with col2:
            st.metric("Entities", stats["entities"])
            st.metric("Relations", stats["relationships"])
            st.metric("FAISS Vectors", stats["faiss_vectors"])
        
        if stats["avg_concept_importance"] > 0:
            st.sidebar.metric("Avg Concept Importance", f"{stats['avg_concept_importance']:.2f}")
        
        if stats["most_connected_concept"] != "None":
            st.sidebar.markdown(f"**🔗 Most Connected:** {stats['most_connected_concept']}")
    
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h3>📁 Document Management</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple PDF documents to expand the knowledge base"
    )
    
    if uploaded_files:
        process_uploaded_files(uploaded_files)
    
    if st.session_state.uploaded_files:
        st.sidebar.markdown("### 📚 Processed Documents")
        for filename, info in st.session_state.uploaded_files.items():
            status_icon = "✅" if info["processed"] else "⏳"
            st.sidebar.markdown(f"""
            <div class="document-card">
                {status_icon} <strong>{filename}</strong><br>
                <small>Uploaded: {info.get('upload_time', 'Unknown')[:16]}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h3>💬 Chat Sessions</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.session_state.new_chat_name = st.text_input(
            "New chat name", 
            value="",
            placeholder="Enter chat name",
            key="new_chat_input"
        )
    with col2:
        if st.button("➕", help="Create new chat"):
            if st.session_state.system:
                new_chat_id = str(uuid4())
                name = st.session_state.new_chat_name or f"Chat {len(st.session_state.chat_sessions) + 1}"
                st.session_state.system.chat_db.create_chat_session(new_chat_id, name)
                st.session_state.active_chat_id = new_chat_id
                st.session_state.new_chat_name = ""
                st.rerun()
    
    if st.session_state.system:
        chat_sessions = st.session_state.system.chat_db.get_chat_sessions()
        
        for session in chat_sessions:
            session_id, name, created_at, last_used, user_name = session
            is_active = st.session_state.active_chat_id == session_id
            
            col1, col2 = st.sidebar.columns([4, 1])
            
            with col1:
                button_class = "chat-tab active" if is_active else "chat-tab"
                user_indicator = f" 👤 {user_name}" if user_name else ""
                if st.button(
                    f"{'➤ ' if is_active else ''}{name}{user_indicator}",
                    key=f"chat_tab_{session_id}",
                    help=f"Last used: {last_used}",
                    use_container_width=True
                ):
                    st.session_state.active_chat_id = session_id
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"delete_{session_id}", help="Delete this chat"):
                    if st.session_state.active_chat_id == session_id:
                        st.session_state.active_chat_id = None
                    st.session_state.system.chat_db.delete_chat_session(session_id)
                    st.rerun()
    
    if st.sidebar.button("🗑️ Clear All Data", help="Clear all processed documents and conversation history"):
        clear_all_data()

def process_uploaded_files(uploaded_files):
    """Process newly uploaded files with enhanced feedback"""
    if not st.session_state.system:
        st.sidebar.error("Please initialize the system first!")
        return
    
    for uploaded_file in uploaded_files:
        file_hash = st.session_state.system.get_file_hash(uploaded_file.read())
        uploaded_file.seek(0)
        
        if uploaded_file.name not in st.session_state.uploaded_files:
            st.session_state.uploaded_files[uploaded_file.name] = {
                "hash": file_hash,
                "processed": False,
                "upload_time": datetime.now().isoformat()
            }
            
            with st.sidebar:
                st.markdown(f"""
                <div class="warning-box">
                    ⏳ Processing {uploaded_file.name}...
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    file_content = uploaded_file.read()
                    document = st.session_state.system.process_pdf(file_content, uploaded_file.name)
                    
                    if document:
                        embed_response, processed_docs = st.session_state.system.create_faiss_index([document], uploaded_file.name)
                        
                        if embed_response:
                            st.session_state.uploaded_files[uploaded_file.name]["processed"] = True
                            st.markdown(f"""
                            <div class="success-box">
                                ✅ {uploaded_file.name} processed successfully!
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="error-box">
                                ❌ Failed to process {uploaded_file.name}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="error-box">
                            ❌ Failed to extract content from {uploaded_file.name}
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-box">
                        ❌ Error processing {uploaded_file.name}: {str(e)}
                    </div>
                    """, unsafe_allow_html=True)

def clear_all_data():
    """Clear all data and reset system"""
    if st.session_state.system and st.session_state.system.neo4j_driver:
        try:
            with st.session_state.system.neo4j_driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
        except:
            pass
    
    st.session_state.uploaded_files = {}
    st.session_state.chat_sessions = {}
    st.session_state.active_chat_id = None
    st.session_state.processing_status = {}
    st.session_state.user_profile = None
    
    if st.session_state.system:
        st.session_state.system.faiss_index = None
        st.session_state.system.metadata = []
        st.session_state.system.conversation_history = []
        st.session_state.system.hybrid_search = None
        if st.session_state.system.user_manager:
            st.session_state.system.user_manager.current_user = None
    
    st.sidebar.success("🧹 All data cleared!")
    st.rerun()

def render_analytics_dashboard():
    """Render enhanced analytics dashboard with Plotly visualizations"""
    if not st.session_state.system or not st.session_state.system.neo4j_driver:
        st.info("📊 Analytics dashboard requires Neo4j connection.")
        return

    st.markdown("## 📊 Advanced Knowledge Graph Analytics")

    try:
        with st.session_state.system.neo4j_driver.session() as session:
            concept_analysis = session.run("""
            MATCH (c:Concept)
            WHERE c.avg_importance IS NOT NULL
            RETURN c.type as type, 
                   count(c) as count,
                   avg(c.avg_importance) as avg_importance,
                   max(c.mention_count) as max_mentions
            ORDER BY count DESC
            """).data()

            if concept_analysis:
                col1, col2 = st.columns(2)

                with col1:
                    df_concepts = pd.DataFrame(concept_analysis)
                    fig_concepts = px.pie(
                        df_concepts,
                        values='count',
                        names='type',
                        title="📊 Concept Distribution by Type",
                        labels={'count': 'Concept Count'},
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_concepts.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_concepts, use_container_width=True)

                with col2:
                    fig_scatter = px.scatter(
                        df_concepts,
                        x='count',
                        y='avg_importance',
                        size='max_mentions',
                        color='type',
                        title="🔍 Concept Importance vs. Count",
                        labels={'count': 'Concept Count', 'avg_importance': 'Average Importance'},
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_scatter.update_traces(marker=dict(line=dict(color='#FFFFFF', width=1)))
                    st.plotly_chart(fig_scatter, use_container_width=True)

                st.markdown("### 📈 Top Concepts by Connectivity")
                top_concepts = session.run("""
                MATCH (c:Concept)-[r:RELATES]-()
                WHERE c.avg_importance IS NOT NULL
                RETURN c.name as concept,
                       c.description as description,
                       count(r) as connections,
                       c.avg_importance as importance
                ORDER BY connections DESC
                LIMIT 5
                """).data()

                for concept in top_concepts:
                    # Handle None values for importance and description
                    importance = concept['importance'] if concept['importance'] is not None else 0.0
                    description = concept['description'] if concept['description'] is not None else "No description available"
                    st.markdown(f"""
                    <div class="knowledge-node">
                        <strong>{concept['concept']}</strong> ({concept['connections']} connections)<br>
                        <small>{description[:100]}...</small><br>
                        <small>Importance: {importance:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.warning("No concept data available for analysis.")

    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            ❌ Failed to load analytics: {str(e)}
        </div>
        """, unsafe_allow_html=True)
        logging.error(f"Analytics dashboard error: {str(e)}")

def render_search_demo():
    """Render enhanced search demo with Plotly visualization"""
    st.markdown("## 🔍 Hybrid Search Demo")

    if not st.session_state.system or not st.session_state.uploaded_files:
        st.info("📝 Please upload documents to enable search demo.")
        return

    demo_query = st.text_input(
        "Test query for search comparison:",
        placeholder="Enter a query to compare different search methods",
        key="search_demo_input"
    )

    if demo_query and st.button("🔬 Run Search Comparison"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🎯 Hybrid Search Results")
            if st.session_state.system.hybrid_search:
                results = st.session_state.system.hybrid_search.search(demo_query, top_k=3)
                for i, result in enumerate(results, 1):
                    concept_match = f"<small>🧠 Concept: {result['concept_match']}</small><br>" if result.get('concept_match') else ""
                    related_concepts = f"<small>🔗 Related: {', '.join(result['related_concepts'][:3])}</small>" if result.get('related_concepts') else ""
                    st.markdown(f"""
                    <div class="document-card">
                        <strong>Result {i}</strong> (Hybrid Score: <span class="hybrid-score">{result['hybrid_score']:.3f}</span>)<br>
                        <small>Source: {result.get('source', 'Unknown')}</small><br>
                        <small>Methods: {', '.join(result.get('methods', ['Unknown']))}</small><br>
                        {result['text'][:200]}...<br>
                        {concept_match}
                        {related_concepts}
                    </div>
                    """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 📊 Search Score Comparison")
            if results:
                df = pd.DataFrame([
                    {
                        'Result': f"Result {i+1}",
                        'Vector Score': result.get('vector_score', 0),
                        'Graph Score': result.get('graph_score', 0),
                        'Hybrid Score': result.get('hybrid_score', 0)
                    } for i, result in enumerate(results)
                ])
                fig = go.Figure(data=[
                    go.Bar(name='Vector Score', x=df['Result'], y=df['Vector Score'], marker_color='#667eea'),
                    go.Bar(name='Graph Score', x=df['Result'], y=df['Graph Score'], marker_color='#764ba2'),
                    go.Bar(name='Hybrid Score', x=df['Result'], y=df['Hybrid Score'], marker_color='#f093fb')
                ])
                fig.update_layout(
                    title="Search Score Comparison",
                    xaxis_title="Results",
                    yaxis_title="Score",
                    barmode='group',
                    template='plotly_white',
                    margin=dict(t=50, b=50)
                )
                st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.markdown("### 🧠 Graph Context")
            graph_context = st.session_state.system.get_enhanced_graph_context(demo_query, top_k=3)
            for ctx in graph_context:
                st.markdown(f"""
                <div class="knowledge-node">
                    <strong>{ctx['concept']}</strong> (Score: {ctx['score']:.2f})<br>
                    <small>{ctx.get('description', 'No description')[:100]}...</small><br>
                    <small>Related: {', '.join([item['name'] for item in ctx.get('related_items', [])[:3]])}</small>
                </div>
                """, unsafe_allow_html=True)

def render_chat_interface():
    """Render enhanced chat interface with session management"""
    if not st.session_state.system:
        st.info("🛠️ System not initialized. Please wait or check connections.")
        return

    if not st.session_state.active_chat_id:
        st.info("💬 Select or create a chat session from the sidebar.")
        return

    # Display chat session name
    chat_sessions = st.session_state.system.chat_db.get_chat_sessions()
    chat_name = next((s[1] for s in chat_sessions if s[0] == st.session_state.active_chat_id), "New Chat")
    st.markdown(f"## 💬 Chat Session: {chat_name}")

    chat_container = st.container()
    with chat_container:
        messages = st.session_state.system.chat_db.get_chat_messages(st.session_state.active_chat_id)
        for msg in messages:
            role, content, timestamp, metadata, user_info = msg
            msg_class = {
                "user": "user-message",
                "assistant": "assistant-message",
                "system": "system-message"
            }.get(role, "user-message")
            st.markdown(f"""
            <div class="chat-message {msg_class}">
                <strong>{role.title()}</strong> <small>({timestamp})</small><br>
                {content}<br>
                {f'<small>User Info: {json.loads(user_info)}</small>' if user_info else ''}
            </div>
            """, unsafe_allow_html=True)

    user_input = st.text_input(
        "Your message:",
        placeholder="Ask a question about your documents or anything else...",
        key=f"chat_input_{st.session_state.active_chat_id}"
    )

    if user_input and st.button("🚀 Send", key=f"send_{st.session_state.active_chat_id}"):
        with st.spinner("🤖 Processing your message..."):
            user_info_result = st.session_state.system.process_user_message(user_input, st.session_state.active_chat_id)
            
            st.session_state.system.chat_db.add_message(
                st.session_state.active_chat_id,
                "user",
                user_input,
                metadata={"timestamp": datetime.now().isoformat()},
                user_info_extracted=user_info_result.get("user_info")
            )

            search_results = st.session_state.system.search_documents(
                user_input,
                top_k=3,
                user_id=user_info_result.get("user_id")
            )

            answer = st.session_state.system.generate_answer(
                user_input,
                search_results.get("results", []),
                search_results.get("graph_context", []),
                search_results.get("user_context", "")
            )

            st.session_state.system.chat_db.add_message(
                st.session_state.active_chat_id,
                "assistant",
                answer,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "search_method": search_results.get("search_method", "Unknown"),
                    "result_count": len(search_results.get("results", []))
                }
            )

            st.rerun()

def main():
    """Main application entry point"""
    render_header()

    if not st.session_state.initialized:
        with st.spinner("🛠️ Initializing GraphRAG system..."):
            st.session_state.system = StreamlitGraphRAGSystem()
            if st.session_state.system.setup_voyage_client():
                if st.session_state.system.connect_neo4j():
                    st.session_state.initialized = True
                    st.success("✅ System initialized successfully!")
                else:
                    st.error("❌ Failed to connect to Neo4j. Some features will be limited.")
            else:
                st.error("❌ Failed to initialize Voyage AI client. Search functionality will be limited.")

    render_sidebar()

    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Search Demo", "📊 Analytics"])

    with tab1:
        render_chat_interface()
    with tab2:
        render_search_demo()
    with tab3:
        render_analytics_dashboard()

if __name__ == "__main__":
    main()
                       
