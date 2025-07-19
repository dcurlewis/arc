# ARC (Adaptive Retention & Context) System Requirements Document

## Document Information
- **Version**: 2.0
- **Date**: July 2025
- **Author**: David Curlewis
- **Purpose**: Define requirements for evolving the current context management system

## 1. Executive Summary

ARC is an evolution of the existing AI Context Management System that replaces the markdown file-based storage with a proper database while maintaining the simplicity and "magic" of the current approach. The goal is to improve performance, enable richer relationship tracking, and unlock more proactive AI assistance without adding complexity to daily use.

## 2. Current System Overview

### What Works Well
- Simple markdown files in logical folder structure
- AI generates summaries from transcripts/documents or conversations
- Weekly consolidation into "memory" files
- Memory files loaded into Claude Desktop's Project Knowledge
- Provides baseline context for each new conversation
- Human-readable, easy to backup and understand

### Current Limitations
- Inefficient context usage (large text files)
- Slow performance when updating memory (token-intensive)
- Limited ability to track relationships between entities
- No semantic or relationship-based search capabilities
- Manual process for surfacing relevant historical context

## 3. Vision for ARC

Transform the backend storage from markdown files to a database system while maintaining the same simple user experience. Enable the AI to proactively surface relevant context by understanding relationships, temporal patterns, and semantic connections across all stored knowledge.

### 3.1 Example Interaction
**User**: "I've got a meeting coming up with Anyscale to discuss contract renewal and a couple of open technical issues that their support team is working with us on in our shared Slack channel - can you prepare any talking points or things I should keep in mind?"

**AI Response**: "Sure, below is a formatted list of talking points. It might be worth inviting Arman to this call as you indicated to him in last week's 1:1 that he should take more of an active role in Anyscale support issues. Also, it's been roughly 3 months since you last spoke to Scott at Anyscale about their product roadmap - might be time to get an update on their Q4 plans!"

## 4. Core Requirements

### 4.1 Data Storage Evolution

#### Replace File System with Database
- **Requirement**: Implement a local, maintenance-free database to replace markdown files
- **Key Properties**:
  - Runs locally (no cloud dependencies)
  - Starts automatically (no Docker container management)
  - Free and open source
  - Reliable with built-in backup capabilities
  - Supports both structured data and text content

#### Maintain Human Readability
- **Requirement**: Ability to export/view data in human-readable format
- **Key Properties**:
  - Export to markdown for backup/review
  - Clear data structure that mirrors current folder organization
  - Debugging and inspection capabilities

### 4.2 Enhanced Relationship Tracking

#### Graph Capabilities
- **Requirement**: Track relationships between entities (people, projects, companies, topics)
- **Key Relationships to Track**:
  - Person ↔ Company (e.g., Scott works at Anyscale)
  - Person ↔ Topic (e.g., Arman owns Anyscale technical issues)
  - Meeting ↔ Participants ↔ Decisions
  - Timeline connections (e.g., "last discussed X on date Y")
  - Project dependencies and stakeholders

#### Temporal Awareness
- **Requirement**: Understand time-based patterns and intervals
- **Examples**:
  - "Last spoke to X about Y three months ago"
  - "This issue has come up in 3 of the last 5 meetings"
  - "Quarterly check-ins with vendor Z"

### 4.3 Intelligent Retrieval

#### Semantic Search
- **Requirement**: Find related content beyond keyword matching
- **Capabilities**:
  - Search by meaning, not just text matches
  - Find similar situations or decisions from the past
  - Surface related documents when preparing for meetings

#### Proactive Context Surfacing
- **Requirement**: AI automatically identifies relevant context
- **Examples**:
  - Suggest attendees based on topic ownership
  - Remind of previous commitments or decisions
  - Alert to time-based patterns (overdue follow-ups)
  - Connect current situation to historical precedents

### 4.4 Integration Requirements

#### MCP Tool Evolution
- **Current**: File reading/writing tools
- **New Requirements**:
  - Database query tools
  - Relationship traversal tools
  - Semantic search tools
  - Batch operations for performance

#### Backward Compatibility
- **Requirement**: Gradual migration from current system
- **Approach**:
  - Import existing markdown files into database
  - Maintain ability to export to markdown
  - Parallel operation during transition
  - No disruption to current workflows

## 5. User Experience Requirements

### 5.1 Maintain Simplicity
- Same conversation patterns with Claude Desktop
- No additional software to manage or monitor
- Commands remain natural language based
- No complex query languages to learn

### 5.2 Performance Improvements
- Memory updates complete in seconds, not minutes
- Reduced token usage for routine operations
- Faster context loading at conversation start
- Efficient handling of large knowledge base

### 5.3 Enhanced "Magic"
- AI makes connections user wouldn't think to ask about
- Relevant context appears without explicit requests
- Patterns and insights emerge from accumulated knowledge
- Feeling of having a true "second brain" that remembers everything

## 6. Technical Constraints

### 6.1 Must Have
- Runs entirely locally on macOS
- No external API dependencies for core functions
- No complex setup or maintenance
- Automatic startup with system
- Data integrity and backup capabilities

### 6.2 Should Avoid
- Docker containers requiring manual management
- Cloud-based solutions
- Subscription services
- Complex administration interfaces
- Systems prone to data corruption

## 7. Success Criteria

### 7.1 Functional Success
- All current system capabilities preserved
- 10x improvement in memory update performance
- Successful migration of all existing data
- Zero data loss or corruption events

### 7.2 Experience Success
- More "aha" moments from proactive AI suggestions
- Reduced time preparing for meetings
- Better continuity across conversations
- Increased confidence in decision-making through better context

## 8. Migration Approach

### 8.1 Phase 1: Database Foundation
- Select and implement local database
- Create MCP tools for database operations
- Import existing markdown content
- Verify data integrity

### 8.2 Phase 2: Relationship Layer
- Implement entity extraction from existing content
- Build relationship graph from historical data
- Create tools for relationship queries
- Test relationship-based retrievals

### 8.3 Phase 3: Intelligence Layer
- Add semantic search capabilities
- Implement proactive suggestion engine
- Create temporal pattern recognition
- Enable the "magical" interactions

### 8.4 Phase 4: Optimization
- Performance tuning
- Workflow refinements
- Full migration from markdown system
- Documentation and backup processes

## 9. Out of Scope (For Now)

- Multi-user capabilities
- Mobile applications
- Cloud synchronization
- External integrations (beyond current MCP tools)
- Complex UI beyond Claude Desktop
- Real-time collaboration features

## 10. Next Steps

1. Evaluate database options against requirements
2. Design MCP tool architecture
3. Create proof of concept with subset of data
4. Test performance improvements
5. Plan migration strategy for existing content

---

This document focuses on evolving the storage and retrieval backend while maintaining the simple, effective user experience that makes the current system valuable. The goal is more magic, not more complexity.