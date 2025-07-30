# ARC: Augmented Recall & Context

A knowledge management system that turns your markdown documents into a searchable graph database with AI-powered entity extraction and Claude Desktop integration.

## What it does

ARC processes markdown files to extract **people**, **organisations**, and **projects**, then builds a graph database showing how they're connected. You can search through documents semantically and query relationships between entities.

The system combines:

- **Neo4j** for graph relationships
- **ChromaDB** for vector search  
- **spaCy** for entity extraction
- **Claude Desktop integration** via MCP tools

IMO the main value is finding connections you've forgotten about and preparing context for meetings.

## Installation

### Prerequisites

- Python 3.8+
- Neo4j (via Homebrew recommended)
- Claude Desktop app

### Basic setup

1. **Clone and set up environment**

   ```bash
   git clone https://github.com/dcurlewis/arc.git
   cd arc
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install Neo4j**

   ```bash
   brew install neo4j
   brew services start neo4j
   ```

3. **Download language model**

   ```bash
   python -m spacy download en_core_web_lg
   ```

4. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your Neo4j credentials
   ```

### Claude Desktop integration

1. **Test the MCP server first**

   ```bash
   cd tools
   python test_mcp_server.py
   ```

2. **Copy configuration**

   ```bash
   mkdir -p ~/.config/claude-desktop
   cp claude_desktop/claude_desktop_config.json ~/.config/claude-desktop/
   ```

3. **Update paths in the config file**
   Edit `~/.config/claude-desktop/claude_desktop_config.json` and replace `/path/to/arc` with your actual project path.

4. **Restart Claude Desktop completely**

## Usage

### Import your documents

Place markdown files in the `import/` directory, then:

```bash
cd tools
python arc_import.py --clear  # Fresh start
```

### Query via Claude Desktop

Once configured, you can ask Claude natural language questions:

- 'Tell me everything about [Person Name]'
- 'How are [Person A] and [Person B] connected?'  
- 'Prepare me for my meeting with [Team] about [Topic]'
- 'Find recent discussions about [Project]'

### Available tools

The system provides 12 specialised tools through Claude:

**Search & discovery:**

- `search_entities` - Find people, organisations, projects
- `semantic_search` - Search documents by meaning
- `enhanced_hybrid_search` - Search everything at once

**Relationships:**

- `get_entity_relationships` - Map entity connections  
- `find_connection_path` - Show how entities connect
- `relationship_search` - Find specific relationship patterns

**Context & analysis:**

- `entity_context_search` - Entity info plus related documents
- `entity_centric_search` - Enhanced entity-focused search
- `temporal_search` - Time-range filtered search
- `enhanced_temporal_search` - Advanced temporal queries
- `meeting_preparation` - Context for meeting attendees
- `get_document` - Retrieve specific documents

The way I see it, `enhanced_hybrid_search` is usually your best starting point for complex queries.

## Configuration

The system uses sensible defaults from `config.yaml`. Key environment variables in `.env`:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

You can customise entity extraction rules and relationship patterns in the config file if needed.

## Troubleshooting

**Neo4j connection issues:**

```bash
brew services list | grep neo4j
brew services restart neo4j
```

**Claude Desktop not connecting:**

1. Check the MCP server runs manually: `./tools/arc_mcp_server.py`
2. Verify paths in claude_desktop_config.json
3. Restart Claude Desktop completely

**Import problems:**

```bash
# Reset everything
python arc_import.py --clear
rm -rf data/chromadb  # Nuclear option for vector DB
```

**Permission errors:**

```bash
chmod +x tools/arc_mcp_server.py
```

## What gets extracted

The system identifies:

- **People** (with aliases and variations)
- **Organisations** (companies, teams, departments)  
- **Projects** (initiatives, campaigns, products)
- **Relationships** (who works with whom, project membership)
- **Temporal context** (when things happened)

It's quite good at handling name variations (like 'Dave' vs 'David') and inferring relationships from context.

## Security notes

- All processing happens locally
- Sensitive files (`.env`, `import/`, `data/`) are excluded from git
- Test data uses synthetic examples (not real personal info)

## Project structure

```text
arc/
├── tools/           # Core system modules
├── import/          # Put your markdown files here
├── data/            # Databases (Neo4j exports, ChromaDB)
├── claude_desktop/  # Integration configs (being consolidated)
├── config.yaml      # System configuration
└── .env            # Your credentials
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `python tools/run_comprehensive_tests.py`
4. Submit a pull request

## Licence

MIT - see LICENSE file for details.

---

**Next steps:** Import your documents, configure Claude Desktop, and start exploring your organisational memory through natural language queries.
