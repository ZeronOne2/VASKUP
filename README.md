# VASKUP

**Advanced Agentic RAG System for Patent Analysis using LangGraph**

VASKUP is an advanced patent analysis system that leverages LangGraph and multiple AI agents to provide comprehensive patent research and analysis capabilities.

## Features

- 📄 **Excel File Processing**: Upload and process patent numbers from Excel files
- 🔍 **Patent Search**: Automated patent information retrieval using SerpAPI
- 🧠 **Advanced RAG**: LangGraph-powered agentic RAG system
- 📊 **Vector Store**: Efficient document chunking and embedding with ChromaDB
- 🔄 **Adaptive Workflows**: Self-improving analysis with feedback loops
- 🌐 **Web Interface**: Korean-language Streamlit interface
- 📈 **Rich Reports**: HTML analysis reports with visualizations

## Technology Stack

- **Python 3.11+**
- **LangChain & LangGraph**: Agentic workflows
- **ChromaDB**: Vector storage
- **Streamlit**: Web interface
- **OpenAI API**: Language models
- **SerpAPI**: Patent search
- **uv**: Dependency management

## Quick Start

1. **Clone and Setup**
   ```bash
   git clone <repository>
   cd VASKUP
   ```

2. **Install Dependencies**
   ```bash
   uv sync
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run Application**
   ```bash
   uv run streamlit run main.py
   ```

## Project Structure

```
VASKUP/
├── src/              # Source code
├── data/             # Data files
├── vectorstore/      # Vector database
├── reports/          # Generated reports
├── temp/             # Temporary files
├── pyproject.toml    # Project configuration
└── .env.example      # Environment template
```

## Development

This project follows the Task Master development workflow with LangGraph node-based development approach.

## License

[License information to be added] 