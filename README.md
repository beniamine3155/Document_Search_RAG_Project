# Document Search RAG Project

An intelligent document search and question-answering system powered by Retrieval-Augmented Generation (RAG) and ReAct agents. This project combines document processing, vector search, and large language models to provide accurate answers from your document corpus.

## Features

- **Agentic RAG System**: Advanced RAG implementation with ReAct agents
- **Smart Document Search**: FAISS-powered vector similarity search
- **Multi-Source Ingestion**: Support for URLs, PDFs, and text files
- **Streamlit Web Interface**: User-friendly web application
- **Tool Integration**: Wikipedia search for general knowledge
- **Interactive Chat**: Real-time question answering with history
- **Performance Optimized**: Cached initialization and efficient retrieval

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Document       │    │   Vector Store   │    │   ReAct Agent   │
│  Processor      │───▶│   (FAISS)        │───▶│   (LangGraph)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  URLs, PDFs,    │    │  Embeddings &    │    │  Generated      │
│  Text Files     │    │  Similarity      │    │  Answers        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

1. **Document Ingestion** (`src/document_ingestion/`)
   - Web scraping with BeautifulSoup
   - PDF processing with PyPDF
   - Text chunking with overlap

2. **Vector Store** (`src/vectorstore/`)
   - OpenAI embeddings
   - FAISS similarity search
   - Retrieval interface

3. **ReAct Agent** (`src/nodes/`)
   - Document retriever tool
   - Wikipedia search tool
   - LangGraph workflow orchestration

4. **State Management** (`src/state/`)
   - Pydantic models for type safety
   - Workflow state tracking

## Quick Start

### Prerequisites

- Python 3.13+
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Document_Search_RAG_Project
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Usage

#### 1. Streamlit Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

Access the application at `http://localhost:8501`

#### 2. Command Line Interface

```bash
python main.py
```

#### 3. Python API

```python
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# Initialize components
llm = Config.get_llm()
doc_processor = DocumentProcessor()
vector_store = VectorStore()

# Process documents
documents = doc_processor.process_urls(Config.DEFAULT_URLS)
vector_store.create_vectorstore(documents)

# Build RAG system
graph_builder = GraphBuilder(
    retriever=vector_store.get_retriever(),
    llm=llm
)
graph_builder.build()

# Ask questions
result = graph_builder.run("What are the key components of LLM agents?")
print(result['answer'])
```

## Configuration

### Default Settings (`src/config/config.py`)

```python
# Model Configuration
LLM_MODEL = "openai:gpt-4o"

# Document Processing
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Default URLs
DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
]
```

### Custom Data Sources

Add your URLs to `data/url.txt`:
```
https://example.com/document1
https://example.com/document2
```

Or modify the configuration directly in your code.

## Project Structure

```
Document_Search_RAG_Project/
├── src/
│   ├── config/
│   │   └── config.py                 # Configuration settings
│   ├── document_ingestion/
│   │   └── document_processor.py     # Document loading & processing
│   ├── vectorstore/
│   │   └── vectorstore.py            # FAISS vector operations
│   ├── nodes/
│   │   ├── nodes.py                  # Basic RAG nodes
│   │   └── reactnode.py              # ReAct agent implementation
│   ├── state/
│   │   └── rag_state.py              # State management
│   └── graph_builder/
│       └── graph_builder.py          # LangGraph workflow
├── data/
│   ├── attention.pdf                 # Sample PDF document
│   └── url.txt                       # URL list for processing
├── streamlit_app.py                  # Web interface
├── main.py                          # CLI interface
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## Key Technologies

- **[LangChain](https://langchain.com/)**: LLM application framework
- **[LangGraph](https://langchain-ai.github.io/langgraph/)**: Workflow orchestration
- **[FAISS](https://faiss.ai/)**: Vector similarity search
- **[Streamlit](https://streamlit.io/)**: Web application framework
- **[OpenAI](https://openai.com/)**: Language models and embeddings
- **[Pydantic](https://pydantic.dev/)**: Data validation

## Use Cases

- **Research Assistant**: Query academic papers and research documents
- **Documentation Search**: Find information in technical documentation
- **Knowledge Base**: Search company internal documents
- **Educational Tool**: Learn from textbooks and course materials

## 🔧 Customization

### Adding New Document Types

Extend `DocumentProcessor` in `src/document_ingestion/document_processor.py`:

```python
def load_custom_format(self, file_path: str) -> List[Document]:
    # Implement custom document loader
    pass
```

### Custom Tools for ReAct Agent

Add tools in `src/nodes/reactnode.py`:

```python
def _build_tools(self) -> List[Tool]:
    tools = [retriever_tool, wikipedia_tool]
    
    # Add custom tool
    custom_tool = Tool(
        name="custom_search",
        description="Custom search functionality",
        func=self._custom_search_function,
    )
    tools.append(custom_tool)
    
    return tools
```

### Different LLM Models

Modify `src/config/config.py`:

```python
# Use different models
LLM_MODEL = "anthropic:claude-3-sonnet"  # Anthropic
LLM_MODEL = "openai:gpt-3.5-turbo"      # Cheaper OpenAI
```

## Example Queries

- "What is the concept of agent loop in autonomous agents?"
- "Explain the key components of LLM-powered agents"
- "How do diffusion models work for video generation?"
- "What are the main challenges in building autonomous agents?"

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   ```
   Solution: Ensure OPENAI_API_KEY is set in .env file
   ```

2. **Module Import Errors**
   ```
   Solution: Activate virtual environment and install requirements
   ```

3. **Memory Issues with Large Documents**
   ```
   Solution: Reduce CHUNK_SIZE in config or process fewer documents
   ```
