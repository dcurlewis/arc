#!/usr/bin/env python3
"""
ARC Query Tool
MCP tool for querying the ARC knowledge graph and vector database.
"""

import json
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime

from arc_core import get_db_manager, get_config


def query_person(name: str) -> Dict[str, Any]:
    """Query information about a person."""
    db = get_db_manager()
    
    with db.neo4j.session() as session:
        # Query person and their relationships
        result = session.run("""
            MATCH (p:Person)
            WHERE p.name CONTAINS $name OR $name IN p.aliases
            OPTIONAL MATCH (p)-[r1:WORKS_AT]->(c:Company)
            OPTIONAL MATCH (p)-[r2:MANAGES]->(subordinate:Person)
            OPTIONAL MATCH (p)-[r3:OWNS]->(topic:Topic)
            OPTIONAL MATCH (p)-[r4:OWNS]->(project:Project)
            RETURN p, 
                   collect(DISTINCT {company: c.name, role: r1.role, since: r1.since}) as companies,
                   collect(DISTINCT subordinate.name) as manages,
                   collect(DISTINCT topic.name) as owns_topics,
                   collect(DISTINCT project.name) as owns_projects
        """, name=name)
        
        records = list(result)
        if not records:
            return {"error": f"No person found matching '{name}'"}
        
        person_data = []
        for record in records:
            person = record["p"]
            person_data.append({
                "id": person["id"],
                "name": person["name"],
                "aliases": person.get("aliases", []),
                "email": person.get("email"),
                "companies": [c for c in record["companies"] if c["company"]],
                "manages": record["manages"],
                "owns_topics": record["owns_topics"],
                "owns_projects": record["owns_projects"]
            })
        
        return {"persons": person_data}


def query_relationships(entity_name: str, depth: int = 1) -> Dict[str, Any]:
    """Query relationships for an entity with specified depth."""
    db = get_db_manager()
    
    with db.neo4j.session() as session:
        # Query relationships with variable depth
        result = session.run("""
            MATCH path = (start)-[*1..$depth]-(end)
            WHERE start.name CONTAINS $name
            RETURN path
            LIMIT 50
        """, name=entity_name, depth=depth)
        
        relationships = []
        for record in result:
            path = record["path"]
            path_data = {
                "start": {"type": list(path.start_node.labels)[0], "name": path.start_node["name"]},
                "end": {"type": list(path.end_node.labels)[0], "name": path.end_node["name"]},
                "relationships": [{"type": rel.type, "properties": dict(rel)} for rel in path.relationships]
            }
            relationships.append(path_data)
        
        return {"relationships": relationships}


def query_meetings(since: Optional[str] = None, participant: Optional[str] = None, topic: Optional[str] = None) -> Dict[str, Any]:
    """Query meetings with optional filters."""
    db = get_db_manager()
    
    conditions = []
    params = {}
    
    if since:
        conditions.append("m.date >= $since")
        params["since"] = since
    
    if participant:
        conditions.append("(p:Person)-[:PARTICIPATED_IN]->(m)")
        conditions.append("p.name CONTAINS $participant")
        params["participant"] = participant
    
    if topic:
        conditions.append("(m)-[:DISCUSSED]->(t:Topic)")
        conditions.append("t.name CONTAINS $topic")
        params["topic"] = topic
    
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    
    query = f"""
        MATCH (m:Meeting)
        {where_clause}
        OPTIONAL MATCH (m)<-[:PARTICIPATED_IN]-(participants:Person)
        OPTIONAL MATCH (m)-[:DISCUSSED]->(topics:Topic)
        OPTIONAL MATCH (m)-[:ABOUT]->(projects:Project)
        RETURN m,
               collect(DISTINCT participants.name) as participants,
               collect(DISTINCT topics.name) as topics,
               collect(DISTINCT projects.name) as projects
        ORDER BY m.date DESC
        LIMIT 20
    """
    
    with db.neo4j.session() as session:
        result = session.run(query, **params)
        
        meetings = []
        for record in result:
            meeting = record["m"]
            meetings.append({
                "id": meeting["id"],
                "title": meeting["title"],
                "date": meeting["date"],
                "type": meeting.get("type"),
                "source_file": meeting.get("source_file"),
                "participants": record["participants"],
                "topics": record["topics"],
                "projects": record["projects"]
            })
        
        return {"meetings": meetings}


def context_for_meeting(meeting_info: str) -> Dict[str, Any]:
    """Get relevant context for a meeting."""
    db = get_db_manager()
    
    # First, try to extract entities from the meeting description
    # Note: get_entity_extractor removed - use EnhancedEntityExtractor directly
    from enhanced_entity_extractor import EnhancedEntityExtractor
    config = get_config()
    extractor = EnhancedEntityExtractor(config)
    entities = extractor.extract_entities(meeting_info)
    
    context = {
        "extracted_entities": entities,
        "relevant_history": [],
        "suggested_participants": [],
        "related_projects": []
    }
    
    # Query for each extracted person
    for person in entities.get('persons', []):
        person_context = query_person(person['text'])
        if 'persons' in person_context:
            context["suggested_participants"].extend(person_context['persons'])
    
    # Query for each extracted organization
    for org in entities.get('organizations', []):
        with db.neo4j.session() as session:
            result = session.run("""
                MATCH (c:Company)
                WHERE c.name CONTAINS $org_name
                OPTIONAL MATCH (c)<-[:WORKS_AT]-(people:Person)
                RETURN c, collect(people.name) as people
            """, org_name=org['text'])
            
            for record in result:
                company = record["c"]
                context["related_projects"].append({
                    "company": company["name"],
                    "people": record["people"]
                })
    
    # Use semantic search to find similar meetings
    chroma_client = db.chromadb
    try:
        collection = chroma_client.get_collection("documents")
        embedding = db.embeddings.encode([meeting_info])
        results = collection.query(
            query_embeddings=embedding.tolist(),
            n_results=5
        )
        
        if results['documents']:
            context["relevant_history"] = [
                {
                    "document": doc,
                    "metadata": meta,
                    "distance": dist
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
    except Exception as e:
        context["semantic_search_error"] = str(e)
    
    return context


def semantic_search(query_text: str, limit: int = 10) -> Dict[str, Any]:
    """Perform semantic search across all documents."""
    db = get_db_manager()
    
    try:
        chroma_client = db.chromadb
        collection = chroma_client.get_collection("documents")
        
        # Generate embedding for query
        embedding = db.embeddings.encode([query_text])
        
        # Search for similar documents
        results = collection.query(
            query_embeddings=embedding.tolist(),
            n_results=limit
        )
        
        if not results['documents']:
            return {"results": [], "message": "No documents found"}
        
        search_results = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            search_results.append({
                "content": doc[:500] + "..." if len(doc) > 500 else doc,
                "metadata": meta,
                "similarity_score": 1 - dist,  # Convert distance to similarity
                "distance": dist
            })
        
        return {"results": search_results}
        
    except Exception as e:
        return {"error": f"Semantic search failed: {str(e)}"}


def main():
    """Main function for CLI usage."""
    if len(sys.argv) < 2:
        print("Usage: arc-query <command> [args...]")
        print("Commands:")
        print("  person <name>                    - Query person information")
        print("  relationships <entity> [depth]   - Query relationships")
        print("  meetings [--since YYYY-MM-DD] [--participant NAME] [--topic TOPIC]")
        print("  context-for-meeting <description> - Get context for meeting")
        print("  search <query>                   - Semantic search")
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "person":
            if len(sys.argv) < 3:
                print("Error: person command requires a name")
                sys.exit(1)
            result = query_person(sys.argv[2])
            
        elif command == "relationships":
            if len(sys.argv) < 3:
                print("Error: relationships command requires an entity name")
                sys.exit(1)
            depth = int(sys.argv[3]) if len(sys.argv) > 3 else 1
            result = query_relationships(sys.argv[2], depth)
            
        elif command == "meetings":
            # Parse optional arguments
            since = None
            participant = None
            topic = None
            
            i = 2
            while i < len(sys.argv):
                if sys.argv[i] == "--since" and i + 1 < len(sys.argv):
                    since = sys.argv[i + 1]
                    i += 2
                elif sys.argv[i] == "--participant" and i + 1 < len(sys.argv):
                    participant = sys.argv[i + 1]
                    i += 2
                elif sys.argv[i] == "--topic" and i + 1 < len(sys.argv):
                    topic = sys.argv[i + 1]
                    i += 2
                else:
                    i += 1
            
            result = query_meetings(since, participant, topic)
            
        elif command == "context-for-meeting":
            if len(sys.argv) < 3:
                print("Error: context-for-meeting command requires a description")
                sys.exit(1)
            result = context_for_meeting(" ".join(sys.argv[2:]))
            
        elif command == "search":
            if len(sys.argv) < 3:
                print("Error: search command requires a query")
                sys.exit(1)
            result = semantic_search(" ".join(sys.argv[2:]))
            
        else:
            print(f"Error: Unknown command '{command}'")
            sys.exit(1)
        
        print(json.dumps(result, indent=2, default=str))
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 