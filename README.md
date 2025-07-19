# ARC: Augmented Recall & Context

An intelligent knowledge management system that transforms markdown documents into a searchable graph database with entity extraction, relationship mapping, and semantic search capabilities.

## 🚀 Overview

ARC processes your markdown documents to:
- **Extract entities** (people, organizations, dates, technologies) using spaCy NLP
- **Infer relationships** between entities based on context and proximity  
- **Build a knowledge graph** stored in Neo4j for complex queries
- **Enable semantic search** through ChromaDB vector embeddings
- **Provide entity disambiguation** to handle name variations

## 🏗️ Architecture

- **Neo4j**: Graph database for entities and relationships
- **ChromaDB**: Vector database for semantic document search
- **spaCy**: Natural language processing for entity extraction
- **Sentence Transformers**: Text embeddings for semantic similarity

## 📋 Prerequisites

- Python 3.8+
- Neo4j (via Homebrew or Docker)
- Git

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/dcurlewis/arc.git
cd arc
```

### 2. Set Up Python Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Neo4j
```bash
# Using Homebrew (recommended for macOS)
brew install neo4j
brew services start neo4j

# Or using Docker
docker run -d \
  --name neo4j-arc \
  -p 7474:7474 -p 7687:7687 \
  -v neo4j_data:/data \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:latest
```

### 4. Download spaCy Model
```bash
python -m spacy download en_core_web_lg
```

### 5. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env with your credentials (see Configuration section)
```

## ⚙️ Configuration

Create a `.env` file in the project root:

```bash
# Neo4j Database Credentials
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# Optional: Override other settings if needed
# SPACY_MODEL=en_core_web_lg
# CHROMADB_PATH=./data/chromadb
# IMPORT_SOURCE_DIR=./import
```

The system uses sensible defaults defined in `config.yaml`. You can customize:
- Entity disambiguation rules
- Relationship extraction keywords  
- Performance settings
- Logging configuration

## 📂 Project Structure

```
arc/
├── tools/                  # Core ARC modules
│   ├── arc_core.py        # Database managers and configuration
│   ├── arc_import.py      # Import pipeline for processing files
│   └── arc_query.py       # Query interface for searching
├── tests/                 # Comprehensive test suite (Star Wars themed data)
├── docs/                  # Project documentation
├── config.yaml           # Configuration with sensible defaults
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables (create from .env.example)
```

## 🔄 Usage

### Import Your Documents

Place markdown files in the `import/` directory, then run:

```bash
cd tools
python arc_import.py --help
```

Options:
- `--limit N`: Process only first N files (useful for testing)
- `--verbose`: Detailed logging output
- `--directory PATH`: Custom import directory

Examples:
```bash
# Test with a few files
python arc_import.py --limit 5 --verbose

# Import all files
python arc_import.py

# Import from custom directory  
python arc_import.py --directory /path/to/your/docs
```

### Query Your Knowledge

```python
from arc_core import get_db_manager
from arc_query import ARCQueryEngine

# Initialize
db_manager = get_db_manager()
query_engine = ARCQueryEngine(db_manager)

# Semantic search
results = query_engine.semantic_search("machine learning projects")

# Entity queries
people = query_engine.find_entities(entity_type="PERSON")
orgs = query_engine.find_entities(entity_type="ORG")

# Relationship queries
relationships = query_engine.find_relationships("Alice", "ORG")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --verbose
```

The test suite uses synthetic Star Wars-themed data to avoid exposing real personal information on GitHub.

## 🔒 Security & Privacy

- **Environment Variables**: All sensitive credentials stored in `.env` (excluded from Git)
- **Data Exclusion**: `import/`, `data/`, and `logs/` directories excluded from version control
- **Synthetic Test Data**: Tests use Star Wars universe data to prevent real data exposure
- **Local Processing**: All data processing happens locally on your machine

## 📈 Performance

- **Incremental Processing**: Files are hashed to avoid reprocessing unchanged content
- **Batch Operations**: Entities and relationships created in batches for efficiency
- **Configurable Limits**: Adjust batch sizes and processing limits in `config.yaml`
- **Progress Tracking**: Detailed logging of import progress and statistics

## 🐛 Troubleshooting

### Neo4j Connection Issues
```bash
# Check Neo4j status
brew services list | grep neo4j

# Restart Neo4j
brew services restart neo4j

# View Neo4j logs
tail -f /usr/local/var/log/neo4j.log
```

### spaCy Model Issues
```bash
# Reinstall language model
python -m spacy download en_core_web_lg --force
```

### ChromaDB Issues
```bash
# Reset ChromaDB (deletes all vector data)
rm -rf data/chromadb
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python run_tests.py`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **spaCy** for excellent NLP capabilities
- **Neo4j** for powerful graph database functionality  
- **ChromaDB** for efficient vector storage and similarity search
- **Sentence Transformers** for high-quality text embeddings

## 📞 Support

For questions or issues:
1. Check the [documentation](docs/)
2. Search [existing issues](https://github.com/dcurlewis/arc/issues)
3. Create a [new issue](https://github.com/dcurlewis/arc/issues/new) if needed

---

**ARC**: Turning your documents into an intelligent, searchable knowledge graph. 🧠✨ 