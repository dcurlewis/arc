# ARC MCP Server Setup for Claude Desktop

## 🎯 Overview

This guide helps you integrate the custom ARC MCP server with Claude Desktop, enabling intelligent access to your knowledge graph and vector database directly from Claude conversations.

## 🛠️ Available Tools

The ARC MCP server provides 8 powerful tools:

### Graph Database Tools
- **`search_entities`** - Find people, organizations, projects in your knowledge graph
- **`get_entity_relationships`** - Get all relationships for a specific entity  
- **`find_connection_path`** - Find shortest path between two entities

### Vector Search Tools  
- **`semantic_search`** - Search documents by semantic meaning using vector similarity
- **`get_document`** - Retrieve specific documents by ID

### Combined Analysis Tools
- **`entity_context_search`** - Find entity info + related documents
- **`temporal_search`** - Search within time ranges
- **`meeting_preparation`** - Prepare context for meetings with specific attendees

## 📋 Prerequisites

1. ✅ ARC system is installed and working
2. ✅ Neo4j is running (`brew services start neo4j`)  
3. ✅ Data has been imported (your 240 markdown files)
4. ✅ Claude Desktop is installed

**Note:** Throughout this guide, `$ARC_PROJECT_ROOT` refers to your ARC installation directory. Replace `/path/to/arc` with your actual project path (e.g., `/Users/yourname/projects/arc`).

## 🔧 Installation Steps

### Step 1: Verify ARC MCP Server

```bash
# Test the MCP server
cd $ARC_PROJECT_ROOT
python tools/test_mcp_server.py
```

You should see:
```
✅ Configuration loaded successfully
✅ Database manager initialized successfully  
✅ Neo4j connection successful
✅ ChromaDB connection successful - 2 collections found
🎉 ARC MCP Server validation complete - all systems ready!
```

### Step 2: Configure Claude Desktop

1. **Locate Claude Desktop config file:**
   ```bash
   # macOS location
   ~/Library/Application Support/Claude/claude_desktop_config.json
   ```

2. **Create or update the config file:**
   ```bash
   # Create directory if it doesn't exist
   mkdir -p "~/Library/Application Support/Claude"
   
   # Copy our config template
   cp $ARC_PROJECT_ROOT/claude_desktop_config.json "~/Library/Application Support/Claude/claude_desktop_config.json"
   ```

   Or manually add this to your existing config:
   ```json
   {
     "mcpServers": {
       "arc-knowledge-graph": {
         "command": "/path/to/arc/arc_mcp_server.sh",
         "args": [],
         "env": {
           "ARC_LOG_LEVEL": "INFO"
         }
       }
     }
   }
   ```

### Step 3: Restart Claude Desktop

1. **Completely quit Claude Desktop** (Cmd+Q)
2. **Restart Claude Desktop**
3. **Look for the MCP server connection** - you should see indicators in the interface

## 🧪 Testing the Integration

### Basic Entity Search
```
Can you search for entities related to "Canva" in my knowledge graph?
```

### Semantic Document Search  
```
Find documents that discuss "machine learning" or "AI" in my knowledge base.
```

### Meeting Preparation
```
I have a meeting coming up with John Smith and Sarah Johnson. Can you prepare context about our previous interactions and any relevant documents?
```

### Connection Discovery
```
Find the connection path between "Project Alpha" and "Engineering Team" in my knowledge graph.
```

## 🔍 Troubleshooting

### MCP Server Not Connecting

1. **Check Neo4j is running:**
   ```bash
   brew services list | grep neo4j
   ```

2. **Test MCP server manually:**
   ```bash
   cd $ARC_PROJECT_ROOT
   ./arc_mcp_server.sh
   ```

3. **Check Claude Desktop logs:**
   - Look for error messages in Claude Desktop
   - Check system console for MCP-related errors

### Permission Issues

```bash
# Ensure scripts are executable
chmod +x $ARC_PROJECT_ROOT/arc_mcp_server.sh
chmod +x $ARC_PROJECT_ROOT/tools/arc_mcp_server.py
```

### Environment Issues

```bash
# Verify .env file exists
ls -la $ARC_PROJECT_ROOT/.env

# Check virtual environment
source $ARC_PROJECT_ROOT/venv/bin/activate
which python
```

## 📊 Usage Examples

### Entity Research
**Prompt:** "What can you tell me about the entities connected to 'customer support' in my knowledge graph?"

**Expected:** The MCP server will search entities, find relationships, and provide comprehensive context.

### Document Discovery  
**Prompt:** "Find all documents that mention both 'API' and 'authentication' and summarize the key points."

**Expected:** Semantic search will find relevant documents and provide summaries.

### Relationship Analysis
**Prompt:** "Show me how 'Marketing Team' is connected to 'Q4 Campaign' in my knowledge graph."

**Expected:** Path finding will show the relationship chain between entities.

## 🚀 Next Steps

Once the MCP server is working with Claude Desktop, you can:

1. **Create custom prompts** for your specific use cases
2. **Build knowledge exploration workflows** 
3. **Set up automated context preparation** for recurring meetings
4. **Develop domain-specific queries** for your business needs

## 📝 Notes

- The MCP server runs locally and accesses your private knowledge graph
- All data stays on your machine - no external API calls for core functionality  
- Performance scales with your hardware and Neo4j configuration
- You can extend the tools by modifying `tools/arc_mcp_server.py`

---

**Your ARC system is now ready for intelligent knowledge exploration through Claude Desktop! 🧠✨** 