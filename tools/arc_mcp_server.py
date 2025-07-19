#!/usr/bin/env python3
"""
ARC (Augmented Recall & Context) MCP Server

A custom Model Context Protocol server that provides intelligent access to the ARC knowledge graph 
and vector database. This server enables Claude Desktop to perform sophisticated queries combining
graph relationships and semantic search capabilities.

Features:
- Neo4j graph database queries (entities, relationships, schema)
- ChromaDB vector search (semantic document retrieval)
- Combined operations (entity + related content)
- Entity disambiguation and temporal search
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json

from mcp.server import Server
from mcp.types import (
    Resource, 
    Tool, 
    TextContent, 
    ImageContent, 
    EmbeddedResource,
    CallToolRequest,
    ReadResourceRequest
)
import mcp.server.stdio

from arc_core import get_config, get_db_manager
from enhanced_embeddings import create_enhanced_embedding_system


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arc-mcp-server")

# Initialize server
server = Server("arc-knowledge-graph")

# Global database manager and enhanced embeddings (initialized in main)
db_manager = None
enhanced_query_interface = None


@server.list_resources()
async def list_resources() -> List[Resource]:
    """List available ARC system resources."""
    return [
        Resource(
            uri="arc://schema/graph",
            name="Graph Database Schema", 
            description="Neo4j graph database schema showing entities and relationships",
            mimeType="application/json"
        ),
        Resource(
            uri="arc://schema/collections",
            name="Vector Collections Schema",
            description="ChromaDB collections and their metadata",
            mimeType="application/json"
        ),
        Resource(
            uri="arc://stats/system",
            name="System Statistics",
            description="ARC database statistics and health metrics",
            mimeType="application/json"
        )
    ]


@server.read_resource()
async def read_resource(request: ReadResourceRequest) -> str:
    """Get specific ARC system resource."""
    uri = request.uri
    
    if uri == "arc://schema/graph":
        # Get Neo4j schema information
        with db_manager.neo4j.session() as session:
            result = session.run("""
                CALL db.schema.visualization() YIELD nodes, relationships
                RETURN nodes, relationships
            """)
            record = result.single()
            
            if record:
                return json.dumps({
                    "nodes": record["nodes"],
                    "relationships": record["relationships"]
                }, indent=2)
            else:
                return json.dumps({"error": "Schema not available"})
                
    elif uri == "arc://schema/collections":
        # Get ChromaDB collections info
        try:
            collections = db_manager.chromadb.list_collections()
            collection_info = []
            
            for collection in collections:
                count = collection.count()
                collection_info.append({
                    "name": collection.name,
                    "count": count,
                    "metadata": collection.metadata
                })
                
            return json.dumps(collection_info, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to get collections: {str(e)}"})
            
    elif uri == "arc://stats/system":
        # Get system statistics
        try:
            # Neo4j stats
            with db_manager.neo4j.session() as session:
                neo4j_stats = session.run("""
                    MATCH (n) RETURN count(n) as total_nodes
                    UNION ALL
                    MATCH ()-[r]-() RETURN count(r) as total_relationships
                """).data()
            
            # ChromaDB stats  
            collections = db_manager.chromadb.list_collections()
            total_documents = sum(c.count() for c in collections)
            
            stats = {
                "timestamp": datetime.now().isoformat(),
                "neo4j": {
                    "total_nodes": neo4j_stats[0]["total_nodes"] if neo4j_stats else 0,
                    "total_relationships": neo4j_stats[1]["total_relationships"] if len(neo4j_stats) > 1 else 0
                },
                "chromadb": {
                    "total_collections": len(collections),
                    "total_documents": total_documents
                }
            }
            
            return json.dumps(stats, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Failed to get stats: {str(e)}"})
    
    else:
        raise ValueError(f"Unknown resource: {uri}")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available ARC tools."""
    return [
        # Graph Database Tools
        Tool(
            name="search_entities",
            description="Search for entities (people, organizations, projects) in the knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for entity names or properties"
                    },
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Filter by entity types (PERSON, ORG, PROJECT, etc.)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        
        Tool(
            name="get_entity_relationships",
            description="Get all relationships for a specific entity",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Name of the entity to get relationships for"
                    },
                    "relationship_types": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "Optional: Filter by relationship types"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Relationship depth to traverse (default: 1)",
                        "default": 1
                    }
                },
                "required": ["entity_name"]
            }
        ),
        
        Tool(
            name="find_connection_path",
            description="Find the shortest path between two entities",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_entity": {
                        "type": "string",
                        "description": "Starting entity name"
                    },
                    "to_entity": {
                        "type": "string", 
                        "description": "Target entity name"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum path length (default: 5)",
                        "default": 5
                    }
                },
                "required": ["from_entity", "to_entity"]
            }
        ),
        
        # Vector Search Tools
        Tool(
            name="semantic_search",
            description="Search documents by semantic meaning using vector similarity",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to search for"
                    },
                    "collection": {
                        "type": "string", 
                        "description": "Collection to search in (default: documents)",
                        "default": "documents"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "where": {
                        "type": "object",
                        "description": "Optional metadata filters"
                    }
                },
                "required": ["query"]
            }
        ),
        
        Tool(
            name="get_document",
            description="Retrieve a specific document by ID",
            inputSchema={
                "type": "object", 
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document ID to retrieve"
                    },
                    "collection": {
                        "type": "string",
                        "description": "Collection name (default: documents)",
                        "default": "documents"
                    }
                },
                "required": ["document_id"]
            }
        ),
        
        # Combined Analysis Tools
        Tool(
            name="entity_context_search",
            description="Find entity information combined with related documents",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Entity to search for"
                    },
                    "include_documents": {
                        "type": "boolean",
                        "description": "Include related documents (default: true)",
                        "default": True
                    },
                    "document_limit": {
                        "type": "integer",
                        "description": "Max documents to return (default: 3)",
                        "default": 3
                    }
                },
                "required": ["entity_name"]
            }
        ),
        
        Tool(
            name="temporal_search",
            description="Search for entities and documents within a time range",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date (YYYY-MM-DD format)"
                    },
                    "end_date": {
                        "type": "string", 
                        "description": "End date (YYYY-MM-DD format)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        
        Tool(
            name="meeting_preparation",
            description="Prepare context for upcoming meetings with specific people or organizations",
            inputSchema={
                "type": "object",
                "properties": {
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of meeting attendees (people or organizations)"
                    },
                    "topic": {
                        "type": "string",
                        "description": "Optional: Meeting topic for focused search"
                    },
                    "days_back": {
                        "type": "integer", 
                        "description": "How many days back to search for context (default: 90)",
                        "default": 90
                    }
                },
                "required": ["attendees"]
            }
        ),
        
        # Enhanced Embedding Tools
        Tool(
            name="enhanced_hybrid_search",
            description="Advanced search using enhanced embeddings across documents, entities, and relationships",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "search_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["documents", "entities", "relationships", "hybrid"]},
                        "description": "Types of search to perform (default: all types)"
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional metadata filters for refined search"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        
        Tool(
            name="entity_centric_search",
            description="Search for documents and information related to a specific entity using enhanced embeddings",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Name of the entity to search for"
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "Optional: Type of entity (PERSON, ORG, etc.)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["entity_name"]
            }
        ),
        
        Tool(
            name="relationship_search",
            description="Search for relationships between entities using enhanced embeddings",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_entity": {
                        "type": "string",
                        "description": "Source entity name"
                    },
                    "target_entity": {
                        "type": "string",
                        "description": "Optional: Target entity name"
                    },
                    "relationship_type": {
                        "type": "string",
                        "description": "Optional: Type of relationship to filter by"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["source_entity"]
            }
        ),
        
        Tool(
            name="enhanced_temporal_search",
            description="Advanced temporal search with enhanced embeddings and date filtering",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date (YYYY-MM-DD format)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date (YYYY-MM-DD format)"
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["documents", "hybrid"],
                        "description": "Type of search to perform (default: documents)",
                        "default": "documents"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> List[Union[TextContent, ImageContent, EmbeddedResource]]:
    """Handle tool execution requests."""
    
    try:
        tool_name = name
        args = arguments or {}
        
        logger.info(f"Executing tool: {tool_name} with args: {args}")
        
        if tool_name == "search_entities":
            return await _search_entities(args)
        elif tool_name == "get_entity_relationships":
            return await _get_entity_relationships(args)
        elif tool_name == "find_connection_path":
            return await _find_connection_path(args)
        elif tool_name == "semantic_search":
            return await _semantic_search(args)
        elif tool_name == "get_document":
            return await _get_document(args)
        elif tool_name == "entity_context_search":
            return await _entity_context_search(args)
        elif tool_name == "temporal_search":
            return await _temporal_search(args)
        elif tool_name == "meeting_preparation":
            return await _meeting_preparation(args)
        elif tool_name == "enhanced_hybrid_search":
            return await _enhanced_hybrid_search(args)
        elif tool_name == "entity_centric_search":
            return await _enhanced_entity_centric_search(args)
        elif tool_name == "relationship_search":
            return await _enhanced_relationship_search(args)
        elif tool_name == "enhanced_temporal_search":
            return await _enhanced_temporal_search(args)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
            
    except Exception as e:
        logger.error(f"Tool execution error: {str(e)}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# Tool Implementation Functions

async def _search_entities(args: Dict[str, Any]) -> List[TextContent]:
    """Search for entities in the knowledge graph."""
    query = args["query"]
    entity_types = args.get("entity_types", [])
    limit = args.get("limit", 10)
    
    # Build Cypher query
    where_clause = "WHERE toLower(n.name) CONTAINS toLower($search_term)"
    if entity_types:
        labels_clause = " OR ".join([f"n:{label}" for label in entity_types])
        where_clause += f" AND ({labels_clause})"
    
    cypher = f"""
        MATCH (n)
        {where_clause}
        RETURN n.name as name, labels(n) as types, n.description as description
        LIMIT $limit
    """
    
    with db_manager.neo4j.session() as session:
        result = session.run(cypher, search_term=query, limit=limit)
        entities = []
        
        for record in result:
            entities.append({
                "name": record["name"],
                "types": record["types"], 
                "description": record["description"]
            })
    
    return [TextContent(
        type="text", 
        text=f"Found {len(entities)} entities:\n\n" + 
             "\n".join([f"• **{e['name']}** ({', '.join(e['types'])})" + 
                       (f": {e['description']}" if e['description'] else "")
                       for e in entities])
    )]


async def _get_entity_relationships(args: Dict[str, Any]) -> List[TextContent]:
    """Get relationships for a specific entity."""
    entity_name = args["entity_name"]
    relationship_types = args.get("relationship_types", [])
    depth = args.get("depth", 1)
    
    # Build Cypher query for relationships
    type_filter = ""
    if relationship_types:
        type_filter = "WHERE " + " OR ".join([f"type(r) = '{t}'" for t in relationship_types])
    
    cypher = f"""
        MATCH path = (start {{name: $entity_name}})-[r*1..{depth}]-(connected)
        {type_filter}
        RETURN path, relationships(path) as rels, nodes(path) as nodes
        LIMIT 50
    """
    
    with db_manager.neo4j.session() as session:
        result = session.run(cypher, entity_name=entity_name)
        relationships = []
        
        for record in result:
            rels = record["rels"]
            nodes = record["nodes"]
            
            for i, rel in enumerate(rels):
                start_node = nodes[i]
                end_node = nodes[i + 1]
                
                relationships.append({
                    "from": dict(start_node).get("name", "Unknown"),
                    "relationship": rel.type,
                    "to": dict(end_node).get("name", "Unknown"),
                    "properties": dict(rel)
                })
    
    if not relationships:
        return [TextContent(type="text", text=f"No relationships found for entity: {entity_name}")]
    
    relationship_text = f"Relationships for **{entity_name}**:\n\n"
    for rel in relationships:
        relationship_text += f"• {rel['from']} → **{rel['relationship']}** → {rel['to']}\n"
        if rel['properties']:
            relationship_text += f"  Properties: {rel['properties']}\n"
    
    return [TextContent(type="text", text=relationship_text)]


async def _find_connection_path(args: Dict[str, Any]) -> List[TextContent]:
    """Find shortest path between two entities."""
    from_entity = args["from_entity"]
    to_entity = args["to_entity"] 
    max_depth = args.get("max_depth", 5)
    
    cypher = """
        MATCH path = shortestPath((from {name: $from_entity})-[*1..{}]-(to {name: $to_entity}))
        RETURN path, length(path) as path_length
    """.format(max_depth)
    
    with db_manager.neo4j.session() as session:
        result = session.run(cypher, from_entity=from_entity, to_entity=to_entity)
        record = result.single()
        
        if not record:
            return [TextContent(
                type="text", 
                text=f"No connection found between **{from_entity}** and **{to_entity}**"
            )]
        
        path = record["path"]
        path_length = record["path_length"]
        
        # Extract path details
        nodes = list(path.nodes)
        relationships = list(path.relationships)
        
        path_text = f"Connection path from **{from_entity}** to **{to_entity}** (length: {path_length}):\n\n"
        
        for i in range(len(nodes)):
            node = nodes[i]
            # Convert node to dict to safely access properties
            node_props = dict(node)
            node_name = node_props.get("name", "Unknown")
            path_text += f"{i + 1}. **{node_name}**\n"
            
            if i < len(relationships):
                rel = relationships[i]
                path_text += f"   ↓ *{rel.type}*\n"
        
        return [TextContent(type="text", text=path_text)]


async def _semantic_search(args: Dict[str, Any]) -> List[TextContent]:
    """Perform semantic search on documents."""
    query = args["query"]
    collection_name = args.get("collection", "documents")
    limit = args.get("limit", 5)
    where_filter = args.get("where", {})
    
    try:
        collection = db_manager.chromadb.get_collection(collection_name)
        
        # Perform semantic search
        results = collection.query(
            query_texts=[query],
            n_results=limit,
            where=where_filter if where_filter else None
        )
        
        if not results["documents"] or not results["documents"][0]:
            return [TextContent(type="text", text=f"No documents found for query: '{query}'")]
        
        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
        distances = results["distances"][0] if results["distances"] else [0] * len(documents)
        
        search_text = f"Semantic search results for: **{query}**\n\n"
        
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            similarity = (1 - distance) * 100  # Convert distance to similarity percentage
            
            search_text += f"**Result {i + 1}** (Similarity: {similarity:.1f}%)\n"
            if metadata.get("file_path"):
                search_text += f"*Source: {metadata['file_path']}*\n"
            
            # Truncate long documents
            content = doc[:500] + "..." if len(doc) > 500 else doc
            search_text += f"{content}\n\n"
            search_text += "---\n\n"
        
        return [TextContent(type="text", text=search_text)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Search error: {str(e)}")]


async def _get_document(args: Dict[str, Any]) -> List[TextContent]:
    """Retrieve a specific document by ID."""
    document_id = args["document_id"]
    collection_name = args.get("collection", "documents")
    
    try:
        collection = db_manager.chromadb.get_collection(collection_name)
        
        results = collection.get(ids=[document_id])
        
        if not results["documents"]:
            return [TextContent(type="text", text=f"Document not found: {document_id}")]
        
        document = results["documents"][0]
        metadata = results["metadatas"][0] if results["metadatas"] else {}
        
        doc_text = f"**Document:** {document_id}\n"
        if metadata.get("file_path"):
            doc_text += f"**Source:** {metadata['file_path']}\n"
        if metadata.get("created_at"):
            doc_text += f"**Created:** {metadata['created_at']}\n"
        
        doc_text += f"\n**Content:**\n{document}"
        
        return [TextContent(type="text", text=doc_text)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Document retrieval error: {str(e)}")]


async def _entity_context_search(args: Dict[str, Any]) -> List[TextContent]:
    """Find entity information and related documents."""
    entity_name = args["entity_name"]
    include_documents = args.get("include_documents", True)
    document_limit = args.get("document_limit", 3)
    
    # First get entity information from Neo4j
    entity_info = await _search_entities({"query": entity_name, "limit": 1})
    
    if not include_documents:
        return entity_info
    
    # Then get related documents from ChromaDB
    try:
        collection = db_manager.chromadb.get_collection("documents")
        
        # Search for documents mentioning this entity
        results = collection.query(
            query_texts=[entity_name],
            n_results=document_limit
        )
        
        combined_text = entity_info[0].text + "\n\n**Related Documents:**\n\n"
        
        if results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
            
            for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                combined_text += f"**Document {i + 1}:**"
                if metadata.get("file_path"):
                    combined_text += f" *{metadata['file_path']}*"
                combined_text += "\n"
                
                # Show relevant excerpt
                content = doc[:300] + "..." if len(doc) > 300 else doc
                combined_text += f"{content}\n\n"
        else:
            combined_text += "No related documents found.\n"
        
        return [TextContent(type="text", text=combined_text)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Context search error: {str(e)}")]


async def _temporal_search(args: Dict[str, Any]) -> List[TextContent]:
    """Search for entities and documents within a time range."""
    query = args["query"]
    start_date = args.get("start_date")
    end_date = args.get("end_date") 
    limit = args.get("limit", 10)
    
    results_text = f"Temporal search results for: **{query}**"
    if start_date:
        results_text += f"\n**From:** {start_date}"
    if end_date:
        results_text += f"\n**To:** {end_date}"
    results_text += "\n\n"
    
    # Build ChromaDB temporal filter
    where_filter = {}
    if start_date or end_date:
        # Convert date strings to timestamps for ChromaDB
        date_conditions = []
        if start_date:
            # Convert YYYY-MM-DD to timestamp
            start_timestamp = datetime.strptime(start_date, "%Y-%m-%d").timestamp()
            date_conditions.append({"created_at": {"$gte": start_timestamp}})
        if end_date:
            # Convert YYYY-MM-DD to timestamp (end of day)
            end_timestamp = datetime.strptime(end_date + " 23:59:59", "%Y-%m-%d %H:%M:%S").timestamp()
            date_conditions.append({"created_at": {"$lte": end_timestamp}})
        
        if len(date_conditions) == 1:
            where_filter = date_conditions[0]
        elif len(date_conditions) == 2:
            where_filter = {"$and": date_conditions}
    
    # Search documents with temporal constraints
    try:
        collection = db_manager.chromadb.get_collection("documents")
        
        # Perform semantic search with temporal filter
        results = collection.query(
            query_texts=[query],
            n_results=limit,
            where=where_filter if where_filter else None
        )
        
        if results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
            distances = results["distances"][0] if results["distances"] else [0] * len(documents)
            
            results_text += "**📄 Documents:**\n\n"
            
            for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                similarity = (1 - distance) * 100
                
                results_text += f"**Document {i + 1}** (Similarity: {similarity:.1f}%)\n"
                if metadata.get("file_path"):
                    results_text += f"*Source: {metadata['file_path']}*\n"
                if metadata.get("created_at"):
                    results_text += f"*Date: {metadata['created_at']}*\n"
                
                # Show relevant excerpt
                content = doc[:300] + "..." if len(doc) > 300 else doc
                results_text += f"{content}\n\n"
                results_text += "---\n\n"
        else:
            results_text += "**📄 Documents:** None found\n\n"
            
    except Exception as e:
        results_text += f"**📄 Documents:** Error searching - {str(e)}\n\n"
    
    # Search entities with temporal constraints
    try:
        # Build Cypher query for temporal entity search
        where_conditions = []
        params = {"search_term": query, "limit": limit}
        
        if start_date:
            where_conditions.append("(n.created_at >= $start_date OR n.modified_at >= $start_date)")
            params["start_date"] = start_date
            
        if end_date:
            where_conditions.append("(n.created_at <= $end_date OR n.modified_at <= $end_date)")
            params["end_date"] = end_date
        
        # Build full Cypher query
        base_where = "WHERE toLower(n.name) CONTAINS toLower($search_term)"
        if where_conditions:
            temporal_where = " AND (" + " AND ".join(where_conditions) + ")"
            base_where += temporal_where
        
        cypher = f"""
            MATCH (n)
            {base_where}
            RETURN n.name as name, labels(n) as types, n.description as description,
                   n.created_at as created_at, n.modified_at as modified_at
            ORDER BY COALESCE(n.modified_at, n.created_at) DESC
            LIMIT $limit
        """
        
        with db_manager.neo4j.session() as session:
            result = session.run(cypher, **params)
            entities = []
            
            for record in result:
                entities.append({
                    "name": record["name"],
                    "types": record["types"],
                    "description": record["description"],
                    "created_at": record["created_at"],
                    "modified_at": record["modified_at"]
                })
        
        results_text += "**🔗 Entities:**\n\n"
        
        if entities:
            for entity in entities:
                results_text += f"• **{entity['name']}** ({', '.join(entity['types'])})"
                if entity['description']:
                    results_text += f": {entity['description']}"
                
                # Add temporal information
                if entity['created_at']:
                    results_text += f"\n  *Created: {entity['created_at']}*"
                if entity['modified_at']:
                    results_text += f"\n  *Modified: {entity['modified_at']}*"
                results_text += "\n\n"
        else:
            results_text += "None found within the specified time range\n\n"
            
    except Exception as e:
        results_text += f"Error searching entities: {str(e)}\n\n"
    
    return [TextContent(type="text", text=results_text)]


async def _meeting_preparation(args: Dict[str, Any]) -> List[TextContent]:
    """Prepare context for meetings with specific attendees."""
    attendees = args["attendees"]
    topic = args.get("topic", "")
    days_back = args.get("days_back", 90)
    
    context_text = f"**Meeting Preparation Context**\n\n"
    context_text += f"**Attendees:** {', '.join(attendees)}\n"
    if topic:
        context_text += f"**Topic:** {topic}\n"
    context_text += f"**Context Period:** Last {days_back} days\n\n"
    
    for attendee in attendees:
        context_text += f"## Context for {attendee}\n\n"
        
        # Get entity relationships
        try:
            entity_context = await _entity_context_search({
                "entity_name": attendee,
                "include_documents": True,
                "document_limit": 2
            })
            
            context_text += entity_context[0].text + "\n\n"
            
        except Exception as e:
            context_text += f"Error getting context for {attendee}: {str(e)}\n\n"
    
    # If there's a specific topic, search for related content
    if topic:
        context_text += f"## Topic-Related Content: {topic}\n\n"
        try:
            topic_results = await _semantic_search({
                "query": topic,
                "limit": 3
            })
            context_text += topic_results[0].text + "\n"
        except Exception as e:
            context_text += f"Error searching topic content: {str(e)}\n"
    
    return [TextContent(type="text", text=context_text)]


# Enhanced Embedding Tool Handlers

async def _enhanced_hybrid_search(args: Dict[str, Any]) -> List[TextContent]:
    """Perform enhanced hybrid search across multiple embedding types."""
    query = args["query"]
    search_types = args.get("search_types", ["documents", "entities", "relationships"])
    filters = args.get("filters", {})
    limit = args.get("limit", 10)
    
    try:
        if enhanced_query_interface is None:
            return [TextContent(type="text", text="Enhanced query interface not initialized")]
        
        results = enhanced_query_interface.hybrid_search(
            query=query,
            search_types=search_types,
            filters=filters,
            limit=limit
        )
        
        if not results:
            return [TextContent(type="text", text=f"No results found for hybrid search: '{query}'")]
        
        search_text = f"**Enhanced Hybrid Search Results for:** {query}\n\n"
        search_text += f"*Search Types:* {', '.join(search_types)}\n"
        search_text += f"*Results Found:* {len(results)}\n\n"
        
        for i, result in enumerate(results):
            search_text += f"**Result {i + 1}** ({result['search_type']}, Score: {result['score']:.3f})\n"
            
            # Add metadata context
            metadata = result.get('metadata', {})
            if metadata.get('file_name'):
                search_text += f"*Source:* {metadata['file_name']}\n"
            
            if metadata.get('entity_text'):
                search_text += f"*Entity:* {metadata['entity_text']} ({metadata.get('entity_label', 'Unknown')})\n"
            
            if metadata.get('relationship_type'):
                search_text += f"*Relationship:* {metadata['source_entity']} {metadata['relationship_type']} {metadata['target_entity']}\n"
            
            # Content preview
            content = result['content']
            if len(content) > 300:
                content = content[:300] + "..."
            search_text += f"{content}\n\n"
            search_text += "---\n\n"
        
        return [TextContent(type="text", text=search_text)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Enhanced hybrid search error: {str(e)}")]


async def _enhanced_entity_centric_search(args: Dict[str, Any]) -> List[TextContent]:
    """Search for documents related to a specific entity using enhanced embeddings."""
    entity_name = args["entity_name"]
    entity_type = args.get("entity_type")
    limit = args.get("limit", 10)
    
    try:
        if enhanced_query_interface is None:
            return [TextContent(type="text", text="Enhanced query interface not initialized")]
        
        results = enhanced_query_interface.entity_centric_search(
            entity_name=entity_name,
            entity_type=entity_type,
            limit=limit
        )
        
        if not results:
            return [TextContent(type="text", text=f"No results found for entity: '{entity_name}'")]
        
        search_text = f"**Enhanced Entity-Centric Search for:** {entity_name}\n\n"
        if entity_type:
            search_text += f"*Entity Type:* {entity_type}\n"
        search_text += f"*Results Found:* {len(results)}\n\n"
        
        for i, result in enumerate(results):
            search_text += f"**Result {i + 1}** (Score: {result['score']:.3f})\n"
            
            metadata = result.get('metadata', {})
            if metadata.get('file_name'):
                search_text += f"*Source:* {metadata['file_name']}\n"
            
            if metadata.get('entity_canonical'):
                search_text += f"*Canonical Name:* {metadata['entity_canonical']}\n"
            
            content = result['content']
            if len(content) > 400:
                content = content[:400] + "..."
            search_text += f"{content}\n\n"
            search_text += "---\n\n"
        
        return [TextContent(type="text", text=search_text)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Enhanced entity search error: {str(e)}")]


async def _enhanced_relationship_search(args: Dict[str, Any]) -> List[TextContent]:
    """Search for relationships between entities using enhanced embeddings."""
    source_entity = args["source_entity"]
    target_entity = args.get("target_entity")
    relationship_type = args.get("relationship_type")
    limit = args.get("limit", 10)
    
    try:
        if enhanced_query_interface is None:
            return [TextContent(type="text", text="Enhanced query interface not initialized")]
        
        results = enhanced_query_interface.relationship_search(
            source_entity=source_entity,
            target_entity=target_entity,
            relationship_type=relationship_type,
            limit=limit
        )
        
        if not results:
            return [TextContent(type="text", text=f"No relationships found for: '{source_entity}'")]
        
        search_text = f"**Enhanced Relationship Search for:** {source_entity}\n\n"
        if target_entity:
            search_text += f"*Target Entity:* {target_entity}\n"
        if relationship_type:
            search_text += f"*Relationship Type:* {relationship_type}\n"
        search_text += f"*Results Found:* {len(results)}\n\n"
        
        for i, result in enumerate(results):
            search_text += f"**Result {i + 1}** (Score: {result['score']:.3f})\n"
            
            metadata = result.get('metadata', {})
            if metadata.get('relationship_type'):
                search_text += f"*Relationship:* {metadata['source_entity']} **{metadata['relationship_type']}** {metadata['target_entity']}\n"
            
            if metadata.get('file_name'):
                search_text += f"*Source:* {metadata['file_name']}\n"
            
            content = result['content']
            if len(content) > 300:
                content = content[:300] + "..."
            search_text += f"{content}\n\n"
            search_text += "---\n\n"
        
        return [TextContent(type="text", text=search_text)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Enhanced relationship search error: {str(e)}")]


async def _enhanced_temporal_search(args: Dict[str, Any]) -> List[TextContent]:
    """Perform enhanced temporal search with date filtering."""
    query = args["query"]
    start_date = args.get("start_date")
    end_date = args.get("end_date")
    search_type = args.get("search_type", "documents")
    limit = args.get("limit", 10)
    
    try:
        if enhanced_query_interface is None:
            return [TextContent(type="text", text="Enhanced query interface not initialized")]
        
        results = enhanced_query_interface.temporal_search(
            query=query,
            start_date=start_date,
            end_date=end_date,
            search_type=search_type,
            limit=limit
        )
        
        if not results:
            date_range = ""
            if start_date and end_date:
                date_range = f" between {start_date} and {end_date}"
            elif start_date:
                date_range = f" after {start_date}"
            elif end_date:
                date_range = f" before {end_date}"
            
            return [TextContent(type="text", text=f"No results found for temporal search: '{query}'{date_range}")]
        
        search_text = f"**Enhanced Temporal Search for:** {query}\n\n"
        if start_date:
            search_text += f"*Start Date:* {start_date}\n"
        if end_date:
            search_text += f"*End Date:* {end_date}\n"
        search_text += f"*Search Type:* {search_type}\n"
        search_text += f"*Results Found:* {len(results)}\n\n"
        
        for i, result in enumerate(results):
            search_text += f"**Result {i + 1}** (Score: {result['score']:.3f})\n"
            
            metadata = result.get('metadata', {})
            if metadata.get('file_name'):
                search_text += f"*Source:* {metadata['file_name']}\n"
            
            if metadata.get('created_at_iso'):
                search_text += f"*Date:* {metadata['created_at_iso'][:10]}\n"
            elif metadata.get('date'):
                search_text += f"*Date:* {metadata['date']}\n"
            
            content = result['content']
            if len(content) > 400:
                content = content[:400] + "..."
            search_text += f"{content}\n\n"
            search_text += "---\n\n"
        
        return [TextContent(type="text", text=search_text)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Enhanced temporal search error: {str(e)}")]


async def main():
    """Main entry point for the ARC MCP server."""
    global db_manager, enhanced_query_interface
    
    # Initialize database manager
    try:
        config = get_config()
        db_manager = get_db_manager()
        
        # Initialize enhanced embedding system
        embedding_generator, enhanced_query_interface = create_enhanced_embedding_system(db_manager)
        
        logger.info("ARC MCP Server initialized successfully with enhanced embeddings")
        
        # Run the server
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
            
    except Exception as e:
        logger.error(f"Failed to start ARC MCP server: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 