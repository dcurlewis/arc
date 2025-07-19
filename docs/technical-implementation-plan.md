# ARC (Adaptive Retention & Context) Technical Implementation Plan

## Document Information
- **Version**: 1.0
- **Date**: 18 July 2025
- **Purpose**: Technical blueprint for implementing the ARC system
- **Status**: Final Plan

## Executive Summary

This plan outlines the technical implementation of ARC, evolving from the current markdown-based AI Context Management System to a graph database-backed solution with semantic search capabilities. The system will maintain the simplicity of current command-based interactions while adding intelligent relationship tracking and proactive context surfacing.

### Key Technical Decisions
- **Primary Database**: Neo4j (embedded mode) for graph relationships
- **Semantic Search**: ChromaDB for vector embeddings and similarity search
- **Entity Extraction**: spaCy for accurate NER with context-aware disambiguation
- **MCP Architecture**: Leverage existing tools where possible, create custom ARC tools for complex operations
- **Migration**: Phased approach preserving all existing functionality

## 1. System Architecture

### 1.1 Core Components

```
┌─────────────────────────────────────────────────────────┐
│                   Claude Desktop                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │         ARC Project Configuration                │   │
│  │  • System Prompt with command definitions        │   │
│  │  • Memory baseline (current-memory.md)           │   │
│  │  • .cursorrules for development guidance         │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                            │
                            ├──── MCP Tools Interface
                            │
┌───────────────────────────┴─────────────────────────────┐
│                    MCP Tool Layer                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Existing    │  │ Neo4j MCP    │  │ Custom ARC   │  │
│  │ File Tools  │  │ (if available)│  │ MCP Tools    │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────┐
│                   Data Layer                             │
│  ┌─────────────────┐              ┌─────────────────┐  │
│  │     Neo4j       │              │    ChromaDB     │  │
│  │  (Graph Store)  │              │ (Vector Store)  │  │
│  └─────────────────┘              └─────────────────┘  │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Markdown Files (Export/Backup)         │   │
│  └─────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| Graph Database | Neo4j Embedded (headless) | Native graph operations, no GUI overhead, runs in-process |
| Vector Database | ChromaDB | Simple setup, good Python support, local persistence |
| Entity Extraction | spaCy (en_core_web_lg) | Accurate NER, customisable, good performance |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Balanced size/performance, runs locally |
| MCP Framework | Python-based tools | Integrates with existing ecosystem |
| Scripting | Python 3.11+ | Consistency with MCP tools, rich ecosystem |

## 2. Database Design

### 2.1 Neo4j Graph Schema

```cypher
// Core Entity Nodes
(:Person {
  id: String,
  name: String,
  aliases: [String],
  email: String?,
  created_at: DateTime,
  updated_at: DateTime
})

(:Company {
  id: String,
  name: String,
  type: String, // vendor, partner, internal
  created_at: DateTime
})

(:Project {
  id: String,
  name: String,
  status: String, // ACTIVE, COMPLETED, BLOCKED, PLANNED
  priority: String,
  created_at: DateTime
})

(:Topic {
  id: String,
  name: String,
  category: String,
  created_at: DateTime
})

(:Meeting {
  id: String,
  title: String,
  date: DateTime,
  type: String,
  source_file: String
})

(:Document {
  id: String,
  title: String,
  type: String,
  date_created: DateTime,
  source_file: String,
  content_hash: String
})

(:Decision {
  id: String,
  title: String,
  status: String, // IMPLEMENTED, PENDING, BLOCKED
  date_made: DateTime,
  rationale: String
})

(:Task {
  id: String,
  title: String,
  status: String,
  priority: String,
  due_date: DateTime?,
  created_at: DateTime
})

// Relationship Types
(:Person)-[:WORKS_AT {role: String, since: DateTime}]->(:Company)
(:Person)-[:MANAGES]->(:Person)
(:Person)-[:OWNS]->(:Topic)
(:Person)-[:OWNS]->(:Project)
(:Person)-[:PARTICIPATED_IN]->(:Meeting)
(:Meeting)-[:DISCUSSED]->(:Topic)
(:Meeting)-[:ABOUT]->(:Project)
(:Meeting)-[:RESULTED_IN]->(:Decision)
(:Decision)-[:AFFECTS]->(:Project)
(:Task)-[:ASSIGNED_TO]->(:Person)
(:Task)-[:RELATED_TO]->(:Project)
(:Document)-[:MENTIONS]->(:Person)
(:Document)-[:REFERENCES]->(:Project)

// Temporal Relationships
(:Topic)-[:LAST_DISCUSSED {date: DateTime}]->(:Meeting)
(:Person)-[:LAST_CONTACTED {date: DateTime, context: String}]->(:Person)
```

### 2.2 ChromaDB Collections

```python
# Collection: documents
{
    "id": "doc_uuid",
    "embedding": [...],  # Vector from content
    "metadata": {
        "type": "meeting|document|decision",
        "date": "2025-07-18",
        "participants": ["person_id_1", "person_id_2"],
        "topics": ["topic_id_1"],
        "source_file": "path/to/original.md"
    },
    "document": "Full text content..."
}

# Collection: summaries
{
    "id": "summary_uuid",
    "embedding": [...],  # Vector from summary
    "metadata": {
        "entity_type": "person|project|topic",
        "entity_id": "neo4j_node_id",
        "last_updated": "2025-07-18"
    },
    "document": "Summary text..."
}
```

## 3. MCP Tool Architecture

### 3.1 Tool Organisation

```
arc-system/tools/
├── arc_core.py          # Shared utilities and database connections
├── arc_query.py         # Query operations (read-only)
├── arc_update.py        # Update operations (write)
├── arc_analyze.py       # Analysis and insights
└── arc_memory.py        # Memory consolidation operations
```

### 3.2 Core MCP Tools

#### 3.2.1 arc-query
```python
# Query entities and relationships
arc-query person "Scott from Anyscale"
arc-query relationships "Arman" --depth 2
arc-query meetings --since "2025-07-01" --participant "Glen"
arc-query context-for-meeting "Anyscale renewal"
```

#### 3.2.2 arc-update
```python
# Update entities and relationships
arc-update add-entity person "Jane Doe" --company "TechCorp" --role "CTO"
arc-update link "Arman" owns "Anyscale support"
arc-update meeting-processed "20250718-Anyscale-Renewal.md"
```

#### 3.2.3 arc-analyze
```python
# Proactive analysis
arc-analyze upcoming-meetings
arc-analyze overdue-followups
arc-analyze relationship-patterns "vendor"
arc-analyze temporal-gaps  # "Haven't spoken to X in Y months"
```

#### 3.2.4 arc-memory
```python
# Memory consolidation (on-demand)
arc-memory scan --since "2025-07-10"
arc-memory consolidate --output "current-memory.md"
arc-memory export --format markdown --path "./backup/"
```

### 3.3 Integration with Existing Tools

- **File Operations**: Continue using DesktopCommander MCP
- **Neo4j Operations**: Check for existing Neo4j MCP tools, use if suitable
- **Command Mapping**: Preserve existing claude-* commands, route to new backend

## 4. Data Migration Strategy

### 4.1 Migration Phases

#### Phase 1: Initial Import (Automated)
```python
# 1. Parse all Claude-generated summary files from import directory
#    - Meeting insights, project summaries, decision records
#    - Already structured with clear metadata sections
# 2. Extract entities using spaCy from summaries
# 3. Build initial graph structure from relationships
# 4. Generate embeddings for semantic search
# 5. Validate data integrity

# First, manually copy files to import:
# cp -r /Users/dbdave/work/AI-Context/Curated-Context/* ./import/

migration_script.py --source "./import" --validate
```

#### Phase 2: Entity Resolution (Interactive)
```python
# 1. Identify potential duplicates (fuzzy matching)
# 2. Apply disambiguation rules:
#    - Nick = Nicholas (same context)
#    - Geoff (Enablement) ≠ Jeff (Design Gen)
#    - Ze Chen (never just "Z")
# 3. Interactive CLI for ambiguous cases:
#    Found: "Scott" and "Scott from Anyscale"
#    [1] Same person (merge)
#    [2] Different people  
#    [3] Skip for now
# 4. Store confirmed aliases in Person nodes

entity_resolution.py --interactive
```

#### Phase 3: Relationship Inference
```python
# 1. Analyse co-occurrence in meetings
# 2. Extract explicit relationships from content
# 3. Infer temporal patterns
# 4. Build relationship timeline

relationship_builder.py --confidence-threshold 0.8
```

### 4.2 Data Validation

- **Completeness Check**: Ensure all files imported
- **Relationship Validation**: Verify key relationships preserved
- **Search Testing**: Confirm semantic search returns expected results
- **Command Testing**: Validate all existing commands work

## 5. Claude Desktop Integration

### 5.1 Project Structure

```
arc-project/
├── .cursorrules
├── .gitignore              # Exclude import/, data/, backups/
├── system-prompt.md
├── import/                 # Staging area for files to import
│   ├── Meeting-Insights/
│   ├── Project-Insights/
│   ├── Decision-History/
│   └── ... (copied from AI-Context)
├── data/                   # Neo4j and ChromaDB storage
│   ├── neo4j/
│   └── chromadb/
├── backups/                # Automated backups
├── memory/
│   └── current-memory.md
├── prompts/
│   ├── proactive-check.md
│   ├── entity-extraction.md
│   └── memory-consolidation.md
├── commands/
│   └── command-reference.md
└── tools/                  # MCP tools
    ├── arc_core.py
    ├── arc_query.py
    ├── arc_update.py
    └── arc_analyze.py
```

### 5.2 System Prompt

```markdown
# ARC System - Adaptive Retention & Context

You are David's AI assistant with access to the ARC knowledge graph system. 

## Core Capabilities
1. **Graph-Aware Memory**: Access to comprehensive relationship graph
2. **Temporal Intelligence**: Track patterns and intervals
3. **Proactive Insights**: Surface relevant context without being asked
4. **Semantic Understanding**: Find related content by meaning

## ARC Commands
All commands now use the arc- prefix for consistency:
- arc-meeting: Process meeting summary
- arc-memory-update: Consolidate memory (now on-demand)
- arc-monday-tom: Generate Monday update
- arc-meeting-prep: Get context for upcoming meetings
- arc-tasks: Review strategic task list
[... other commands migrated to arc- prefix ...]

## Graph-Specific Commands
- arc-query: Query the knowledge graph
- arc-analyze: Run proactive analysis
- arc-context: Get context for upcoming events

## Proactive Behaviours
- On meeting mentions → Check past meetings, suggest participants
- On person mentions → Surface role, relationships, last interaction
- On project mentions → Show status, blockers, stakeholders
- Daily startup → Run overdue check, surface priorities

## Communication Style
[Include existing style guide]
```

### 5.3 Command Evolution

| Command | Backend Implementation |
|---------|----------------------|
| arc-meeting | Extract entities, update graph, generate embeddings |
| arc-memory-scan | Query graph for changes, not file system |
| arc-memory-update | Generate from graph traversal + temporal analysis |
| arc-meeting-prep | Graph query for person + relationship context |
| arc-monday-tom | Pull from graph with temporal awareness |
| arc-delivery-scan | Track Jira tickets with relationship context |
| arc-tasks | Integrate with Task nodes in graph |

## 6. Implementation Phases

### 6.1 Phase 1: Foundation (Days 1-2)

**Day 1: Environment Setup**
- [ ] Create project directories and .gitignore
- [ ] Copy sample data to ./import/ directory
- [ ] Install Neo4j Embedded (headless, via pip)
- [ ] Install ChromaDB, verify local persistence
- [ ] Set up Python environment with dependencies
- [ ] Create basic MCP tool structure

**Day 2: Core Implementation**
- [ ] Implement database connection managers
- [ ] Create basic CRUD operations for entities
- [ ] Build file import functionality
- [ ] Test with subset of data (10-20 files)

**Milestone**: Successfully import and query test data

### 6.2 Phase 2: Intelligence Layer (Days 2-3)

**Day 2-3: Entity & Search**
- [ ] Configure spaCy with custom rules
- [ ] Implement entity extraction pipeline
- [ ] Set up embedding generation
- [ ] Create semantic search interface

**Day 3: Relationships**
- [ ] Build relationship extraction logic
- [ ] Implement temporal tracking
- [ ] Create relationship query tools
- [ ] Test with meeting transcripts

**Milestone**: Accurate entity extraction and relationship mapping

### 6.3 Phase 3: Integration (Days 3-4)

**Day 3-4: Claude Desktop Setup**
- [ ] Create ARC project structure
- [ ] Write comprehensive system prompt
- [ ] Map existing commands to new backend
- [ ] Implement memory consolidation

**Day 4: Testing & Refinement**
- [ ] Full data migration
- [ ] Test all command patterns
- [ ] Optimise query performance
- [ ] Document setup process

**Milestone**: Fully functional ARC system with existing commands working

### 6.4 Phase 4: Enhancement (Post-MVP)

- Proactive morning briefings
- Advanced temporal patterns
- Relationship strength scoring
- Auto-categorisation improvements
- Performance optimisation

## 7. Performance Targets

| Operation | Current | Target | Method |
|-----------|---------|--------|---------|
| Memory Update | 2-5 min | <30 sec | Graph queries vs full file scan |
| Meeting Prep | 30-60 sec | <5 sec | Indexed relationships |
| Semantic Search | N/A | <2 sec | Vector similarity |
| Entity Extraction | N/A | <10 sec/doc | Cached NER model |

## 8. Testing Strategy

### 8.1 Unit Tests
- Database operations
- Entity extraction accuracy
- Relationship inference logic
- Command parsing

### 8.2 Integration Tests
- Full command workflows
- Data migration integrity
- Claude Desktop interaction
- Memory consolidation

### 8.3 User Acceptance Criteria
- [ ] All existing commands work without user-visible changes
- [ ] Memory updates complete in under 30 seconds
- [ ] Correct entity disambiguation (Geoff vs Jeff)
- [ ] Relevant context surfaces proactively
- [ ] No data loss during migration

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Data loss during migration | Maintain markdown backups, incremental migration |
| Entity extraction errors | Manual review process, confidence thresholds |
| Performance degradation | Indexed queries, connection pooling |
| Complex setup | Detailed documentation, automated setup scripts |

## 10. Maintenance & Operations

### 10.1 Backup Strategy (Three-Tier Approach)
1. **Primary Backup**: Markdown files remain authoritative
   - Original summaries preserved in AI-Context directory
   - Human-readable and version-controlled via Git
   
2. **Neo4j Native Backup**: Weekly automated dumps
   ```bash
   neo4j-admin dump --database=arc --to=/backups/arc-$(date +%Y%m%d).dump
   ```
   
3. **Export Functions**: On-demand full export
   - JSON export for complete graph structure
   - CSV export for analysis in other tools
   - Markdown generation for human review

### 10.2 Automated Maintenance
- Daily graph statistics logging
- Weekly Neo4j backup execution
- Monthly embedding model performance check
- Quarterly model updates if needed

### 10.3 Manual Maintenance
- Entity disambiguation review (as needed)
- Relationship accuracy audit (monthly)
- Command usage analytics review

## 11. Future Enhancements

1. **Multi-modal Context**: Screenshots, diagrams in graph
2. **Predictive Insights**: ML-based pattern detection  
3. **External Integrations**: Calendar, Slack (if needed)
4. **Collaborative Features**: Shared knowledge graphs

## 12. Success Metrics

- **Performance**: 10x improvement in memory operations
- **Accuracy**: 95%+ entity extraction accuracy
- **Proactivity**: 5+ useful unsolicited insights per week
- **Reliability**: Zero data corruption events
- **Usability**: No increase in command complexity

---

## Next Steps

1. Validate technical decisions with test implementations
2. Begin Phase 1 implementation
3. Create detailed setup documentation
4. Establish backup and recovery procedures

This plan provides a solid foundation while maintaining flexibility for discoveries during implementation. 