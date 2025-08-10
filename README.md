
# AItoMate: Intelligent Document Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AItoMate is an advanced AI-powered platform for document analysis, knowledge extraction, and interactive querying. It combines hybrid search (vector + graph), knowledge graphs, and persistent user memory to provide personalized, context-aware responses. Built with Streamlit for the frontend, it supports PDF processing with OCR for tables, multimodal embeddings, and local LLM integration.

Key capabilities include:
- Processing technical PDFs (e.g., VFD data sheets with function codes like P00.13).
- Hybrid search using FAISS for vectors and Neo4j for graph-based traversal.
- User profile management with persistent memory across sessions.
- Chat interface with history, analytics dashboard, and search demos.

## Features

- **Document Processing**: Upload PDFs, extract text/images/tables with OCR (using PyTesseract), chunk content, and generate embeddings via Voyage AI.
- **Hybrid Search Engine**: Combines vector similarity (FAISS) and graph traversal (Neo4j) for enhanced retrieval accuracy.
- **Knowledge Graph**: Extracts entities, concepts, and relationships (e.g., function codes in technical docs) and stores them in Neo4j for multi-hop queries.
- **Personalized Chat**: Multi-session chat with user profile extraction, context-aware responses using Ollama LLM.
- **Analytics Dashboard**: Visualize knowledge graph stats (concept distribution, importance, connectivity) with Plotly.
- **Search Demo**: Compare hybrid, vector, and graph search results with interactive scoring.
- **Persistent Storage**: SQLite for chat history and user profiles; Neo4j for graph data.
- **Modern UI**: Custom CSS for a sleek, responsive interface with animations and gradients.

## Technologies Used

- **Frontend**: Streamlit
- **AI/ML**: Voyage AI (multimodal embeddings), FAISS (vector search), Ollama (local LLM), PyTesseract (OCR)
- **Databases**: Neo4j (graph DB), SQLite (chat history)
- **Libraries**: PyMuPDF (PDF processing), Neo4j Python Driver, Pandas, Plotly, NumPy, Requests, etc.
- **Other**: Logging, Tempfile, UUID, Hashlib for utilities.

## Installation

1. **Clone the Repository**:
   ```
   git clone https://github.com/30omarnasser/AItoMate/blob/main/AItoMate.py
   cd aitomate
   ```

2. **Install Dependencies**:
   Create a virtual environment and install requirements:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
   (Create a `requirements.txt` with: `streamlit voyageai faiss-cpu numpy fitz pillow requests neo4j typing-extensions pytesseract pandas plotly sqlite3 uuid` and others from imports.)

3. **Setup External Services**:
   - **Voyage AI**: Get an API key from [Voyage AI](https://www.voyageai.com/) and set it in the code (`Config.VOYAGE_API_KEY`).
   - **Ollama**: Install [Ollama](https://ollama.ai/) and run it locally (default model: `llama3.2:1b`). Ensure it's accessible at `http://localhost:11434`.
   - **Neo4j**: Install [Neo4j Desktop](https://neo4j.com/download/) or use Aura. Set URI and credentials in `Config.NEO4J_URI` and `Config.NEO4J_AUTH`.
   - **Tesseract OCR**: Install Tesseract (e.g., `brew install tesseract` on macOS, or via apt on Linux) and ensure it's in your PATH.

4. **Environment Variables** (Optional):
   For security, move API keys to `.env` and load with `dotenv`.

## Usage

1. **Run the App**:
   ```
   streamlit run app.py  # Assuming the code is in app.py
   ```
   Open `http://localhost:8501` in your browser.

2. **Key Interactions**:
   - **Sidebar**: Upload PDFs, manage chat sessions, view system status and stats.
   - **Chat Tab**: Create/select sessions, ask questions (e.g., "What is function code P00.13?"). The system extracts user info (e.g., name, interests) for personalization.
   - **Search Demo Tab**: Test queries to compare hybrid search results with visualizations.
   - **Analytics Tab**: View graphs on concepts, importance, and connectivity.
   - **Clear Data**: Use the sidebar button to reset everything.

3. **Example Workflow**:
   - Upload a PDF (e.g., VFD manual).
   - Wait for processing (embeddings + graph creation).
   - Chat: "Tell me about P00.13 in the manual." – Gets personalized, context-aware response.

## Configuration

- **Config Class**: Customize URLs, models, and credentials in the code.
- **Logging**: Set to INFO level; outputs to console.
- **Cleanup**: Automatically cleans old data (>30 days) in Neo4j/SQLite.
- **Export**: Use `export_knowledge_graph()` to export graph as Cypher or JSON.

## Screenshots

*(Add screenshots here or link to images in repo)*

- **Main Interface**  
  ![Main Header](https://github.com/user-attachments/assets/fefa80c0-8b4e-4a6d-9bf8-deb4bb0678c2)

- **Chat Session**  
  ![Chat](https://github.com/user-attachments/assets/f2225a2d-c5e5-42f9-8ff4-7bc17fc5a2da)

- **Analytics Dashboard**  
  ![Analytics](https://github.com/user-attachments/assets/302d5472-d306-41a4-b7d5-05b1449ebae0)


## Contributing

Contributions welcome! Fork the repo, create a branch, and submit a PR. For issues, use GitHub Issues.

## Acknowledgments

- Built with inspiration from GraphRAG systems and multimodal AI.
- Thanks to Voyage AI, Ollama, and Neo4j communities.

For questions, contact [3omarnassereldin@gmail.com].
