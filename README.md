# ARC: Augmented Recall & Context
## Enhanced Knowledge Management System with Claude Desktop Integration

An **production-ready** intelligent knowledge management system that transforms markdown documents into a sophisticated searchable graph database with enhanced AI-powered entity extraction, relationship mapping, and Claude Desktop integration.

## 🚀 Overview

**ARC (Enhanced)** processes your markdown documents to:
- **Extract entities** (people, organizations, projects, concepts) using enhanced spaCy NLP with custom disambiguation
- **Infer sophisticated relationships** through context-aware analysis and proximity detection
- **Build an intelligent knowledge graph** in Neo4j with temporal awareness
- **Enable multi-modal semantic search** through enhanced ChromaDB embeddings
- **Provide Claude Desktop integration** with 12 specialized MCP tools
- **Support natural language queries** through sophisticated AI assistance

## 🎯 **Phase 3 Complete: Production-Ready System**

### **📊 System Capabilities**
- **Enhanced entity extraction** with AI-powered disambiguation and alias handling
- **Sophisticated relationship inference** with context awareness and temporal tracking  
- **Multi-modal document indexing** with semantic understanding and metadata preservation
- **12 specialized MCP tools** for comprehensive knowledge access through Claude Desktop
- **4 embedding types** (documents, entities, relationships, hybrid) for multi-dimensional search

### **🛠️ Available Claude Desktop Tools**
- `search_entities` - Find people, organizations, projects
- `semantic_search` - Document search by meaning  
- `enhanced_hybrid_search` - **Most powerful** - search everything at once
- `get_entity_relationships` - Entity relationship mapping
- `find_connection_path` - How entities connect
- `entity_context_search` - Entity + related documents
- `entity_centric_search` - Enhanced entity-focused search
- `relationship_search` - Specific relationship patterns
- `temporal_search` - Time-range filtered search
- `enhanced_temporal_search` - Advanced temporal search
- `meeting_preparation` - Automated meeting context
- `get_document` - Retrieve specific documents

## 🏗️ Enhanced Architecture

- **Neo4j**: Graph database with sophisticated entity relationships
- **ChromaDB**: Vector database with multi-modal enhanced embeddings
- **Enhanced spaCy**: Custom pipeline with disambiguation and custom patterns
- **Advanced Query Interface**: Hybrid search across all content types
- **Claude Desktop Integration**: Natural language access through MCP protocol
- **Sentence Transformers**: Multiple embedding models for semantic understanding

## 📋 Prerequisites

- Python 3.8+
- Neo4j (via Homebrew or Docker)
- Claude Desktop app (for Phase 3 integration)
- Git

## 🚀 **Quick Start (Phase 3 - Production Ready)**

### **For Claude Desktop Integration**
```bash
# 1. Navigate to your ARC directory
cd /Users/dbdave/work/arc

# 2. Copy Claude Desktop configuration
cp claude_desktop/claude_desktop_config.json ~/.config/claude-desktop/

# 3. Update paths in config for your system
nano ~/.config/claude-desktop/claude_desktop_config.json

# 4. Restart Claude Desktop completely
# 5. Test with: "Search for entities in the ARC system"
```

**📖 Complete Phase 3 Guide**: See [`claude_desktop/installation_guide.md`](claude_desktop/installation_guide.md) for detailed setup instructions.

## 🛠️ Installation (Development Setup)

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
NEO4J_PASSWORD=neo4j # update to your own password on first usage

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

## 📂 **Enhanced Project Structure**

```
arc/
├── tools/                          # Enhanced ARC modules
│   ├── arc_core.py                # Database managers and configuration
│   ├── arc_import.py              # Enhanced import pipeline with --clear option
│   ├── arc_query.py               # Advanced query interface
│   ├── arc_mcp_server.py          # ⭐ MCP server with 12 specialized tools
│   ├── enhanced_entity_extractor.py # ⭐ AI-powered entity extraction
│   ├── enhanced_embeddings.py     # ⭐ Multi-modal embedding system
│   └── run_comprehensive_tests.py # Comprehensive test automation
├── claude_desktop/                 # ⭐ Phase 3 Integration Files
│   ├── README.md                  # Integration overview and quick start
│   ├── installation_guide.md      # Step-by-step setup instructions
│   ├── arc_usage_guide.md         # Detailed examples and best practices  
│   ├── arc_system_prompt.md       # Comprehensive AI system prompt
│   └── claude_desktop_config.json # Claude Desktop MCP configuration
├── tests/                         # ⭐ Enhanced test suite with fixtures
│   ├── unit/                      # Unit tests for enhanced components
│   ├── integration/               # Integration tests for MCP server
│   └── conftest.py               # Enhanced test fixtures
├── config/                        # ⭐ Enhanced configuration system
│   ├── enhanced_entity_config.template.yaml # Anonymized config template
│   └── README.md                 # Configuration documentation
├── config.yaml                   # Enhanced configuration with new capabilities
├── requirements.txt               # Updated dependencies
└── .env                          # Environment variables (enhanced)
```

## 🔄 **Enhanced Usage**

### **Phase 3: Natural Language Queries (Recommended)**

Once Claude Desktop is configured, simply ask natural language questions:

```
"Tell me everything about [Person Name]"
"Prepare me for my meeting with [Team] about [Topic]"  
"How are [Person A] and [Person B] connected?"
"Find recent discussions about [Project/Initiative]"
"What are the key relationships around [Topic]?"
```

**📖 Usage Examples**: See [`claude_desktop/arc_usage_guide.md`](claude_desktop/arc_usage_guide.md) for comprehensive examples.

### **Development: Enhanced Import Pipeline**

Place markdown files in the `import/` directory, then run:

```bash
cd tools
python arc_import.py --help
```

**Enhanced Options:**
- `--clear`: Reset databases before import (fresh start)
- `--limit N`: Process only first N files (useful for testing)  
- `--verbose`: Detailed logging output
- `--directory PATH`: Custom import directory

**Examples:**
```bash
# Fresh import with enhanced extraction (recommended)
python arc_import.py --clear

# Test with enhanced extraction
python arc_import.py --limit 5 --verbose

# Import from custom directory with enhancements
python arc_import.py --directory /path/to/your/docs
```

### **Advanced: Query Interface

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

## 📈 **Development Phases**

### **Phase 1: Foundation** ✅ 
- Basic Neo4j + ChromaDB integration
- Initial MCP server with core tools  
- Basic entity extraction and document indexing

### **Phase 2: Enhancement** ✅
- **Enhanced entity extraction** with custom spaCy pipeline
- **Sophisticated disambiguation** rules and alias handling
- **Multi-modal embeddings** with relationship awareness
- **Advanced query interface** with hybrid search

### **Phase 3: Production Integration** ✅ 
- **Comprehensive Claude Desktop** integration
- **12 specialized tools** with optimized prompting
- **Usage guides and best practices**
- **Global shortcuts and workflow templates**

## 🎉 **Production-Ready System**

Your **enhanced ARC system** provides:
- **Comprehensive entity extraction** with sophisticated disambiguation
- **Rich relationship mapping** with context awareness and temporal tracking
- **Multi-modal document understanding** with preserved metadata
- **12 specialized MCP tools** for comprehensive Claude Desktop access
- **Advanced search capabilities** across multiple embedding types

---

**🎯 ARC Enhanced**: Transforming your documents into an intelligent, searchable knowledge graph with Claude Desktop integration. 🧠✨

*Version 3.0 - Production Ready with Enhanced AI Integration* 