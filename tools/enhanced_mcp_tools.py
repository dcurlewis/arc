"""
Enhanced MCP Tools for Data Ingestion
Write-enabled tools for adding content directly through Claude Desktop.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from arc_core import get_db_manager, get_config
from enhanced_entity_extractor import EnhancedEntityExtractor
from enhanced_embeddings import create_enhanced_embedding_system

logger = logging.getLogger(__name__)


class ContentIngestor:
    """Handles direct content ingestion from Claude Desktop."""
    
    def __init__(self):
        self.config = get_config()
        self.db_manager = get_db_manager()
        
        # Load enhanced entity configuration
        enhanced_config = self._load_enhanced_config()
        self.entity_extractor = EnhancedEntityExtractor(enhanced_config)
        
        # Initialize enhanced embedding system
        self.embedding_generator, self.query_interface = create_enhanced_embedding_system(self.db_manager)
        
        logger.info("Initialized ContentIngestor for direct content ingestion")
    
    def _load_enhanced_config(self) -> Dict[str, Any]:
        """Load enhanced entity extraction configuration."""
        config_path = Path(self.config.config_path).parent / "config" / "enhanced_entity_config.yaml"
        
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Use basic configuration if enhanced config not available
            return {
                'entity_patterns': {},
                'disambiguation_rules': {},
                'relationship_keywords': {}
            }
    
    def ingest_content(self, content: str, title: str, content_type: str = "text", 
                      source: str = "claude_desktop", tags: List[str] = None) -> Dict[str, Any]:
        """
        Ingest raw content directly into the ARC system.
        
        Args:
            content: The text content to ingest
            title: Title/description for the content
            content_type: Type of content (meeting_transcript, document, note, etc.)
            source: Source of the content
            tags: Optional tags for categorization
            
        Returns:
            Ingestion results with counts and IDs
        """
        try:
            # Generate unique document ID
            doc_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Create metadata
            metadata = {
                'content_hash': self._generate_content_hash(content),
                'file_name': f"{content_type}_{doc_id[:8]}.md",
                'title': title,
                'content_type': content_type,
                'source': source,
                'tags': tags or [],
                'created_at': timestamp,
                'modified_at': timestamp,
                'file_size': len(content),
                'ingested_via': 'claude_desktop'
            }
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(
                content, 
                context={'title': title, 'content_type': content_type, 'source': source}
            )
            
            # Extract relationships
            relationships = self.entity_extractor.infer_relationships(
                entities, content, metadata
            )
            
            # Create document node in Neo4j
            doc_node_id = self._create_document_node(metadata, doc_id)
            
            # Create entity nodes
            entity_ids = []
            for entity in entities:
                entity_id = self._create_entity_node(entity, metadata)
                entity_ids.append(entity_id)
            
            # Create relationships
            relationship_count = 0
            for relationship in relationships:
                if self._create_relationship(relationship):
                    relationship_count += 1
            
            # Link entities to document
            self._link_entities_to_document(entity_ids, doc_node_id)
            
            # Index with enhanced embeddings
            self.embedding_generator.index_enhanced_document(
                content, entities, relationships, metadata
            )
            
            result = {
                'success': True,
                'document_id': doc_id,
                'neo4j_doc_id': doc_node_id,
                'entities_created': len(entities),
                'relationships_created': relationship_count,
                'entity_ids': entity_ids,
                'content_type': content_type,
                'title': title,
                'timestamp': timestamp
            }
            
            logger.info(f"Successfully ingested content: {title} ({len(entities)} entities, {relationship_count} relationships)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to ingest content '{title}': {e}")
            return {
                'success': False,
                'error': str(e),
                'title': title
            }
    
    def add_meeting_transcript(self, transcript: str, meeting_title: str, 
                              attendees: List[str] = None, date: str = None) -> Dict[str, Any]:
        """
        Specialized method for meeting transcripts.
        
        Args:
            transcript: Meeting transcript content
            meeting_title: Title of the meeting
            attendees: List of meeting attendees
            date: Meeting date (ISO format)
            
        Returns:
            Ingestion results
        """
        # Enhance content with structured information
        enhanced_content = f"# {meeting_title}\n\n"
        
        if date:
            enhanced_content += f"**Date:** {date}\n"
        
        if attendees:
            enhanced_content += f"**Attendees:** {', '.join(attendees)}\n"
        
        enhanced_content += f"\n## Transcript\n\n{transcript}"
        
        # Add attendee context for better entity extraction
        tags = ['meeting', 'transcript']
        if attendees:
            tags.extend([f"attendee:{name}" for name in attendees])
        
        return self.ingest_content(
            content=enhanced_content,
            title=meeting_title,
            content_type="meeting_transcript",
            source="claude_desktop",
            tags=tags
        )
    
    def add_document_summary(self, original_content: str, summary: str, title: str,
                           source_type: str = "document") -> Dict[str, Any]:
        """
        Add a document with its AI-generated summary.
        
        Args:
            original_content: Original document content
            summary: AI-generated summary
            title: Document title
            source_type: Type of source document
            
        Returns:
            Ingestion results
        """
        # Combine original and summary for comprehensive indexing
        enhanced_content = f"# {title}\n\n## Summary\n\n{summary}\n\n## Full Content\n\n{original_content}"
        
        return self.ingest_content(
            content=enhanced_content,
            title=title,
            content_type=f"{source_type}_with_summary",
            source="claude_desktop",
            tags=['summary', source_type]
        )
    
    def add_quick_note(self, note_content: str, title: str = None) -> Dict[str, Any]:
        """
        Add a quick note or context snippet.
        
        Args:
            note_content: Note content
            title: Optional title (auto-generated if not provided)
            
        Returns:
            Ingestion results
        """
        if not title:
            # Generate title from first line or content preview
            first_line = note_content.split('\n')[0].strip()
            title = first_line[:50] + "..." if len(first_line) > 50 else first_line
            if not title:
                title = f"Quick Note {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        return self.ingest_content(
            content=note_content,
            title=title,
            content_type="quick_note",
            source="claude_desktop",
            tags=['note']
        )
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _create_document_node(self, metadata: Dict[str, Any], doc_id: str) -> str:
        """Create document node in Neo4j."""
        with self.db_manager.neo4j.session() as session:
            node_id = f"doc_{doc_id[:12]}"
            
            properties = {
                'id': node_id,
                'title': metadata['title'],
                'content_type': metadata['content_type'],
                'source': metadata['source'],
                'content_hash': metadata['content_hash'],
                'created_at': metadata['created_at'],
                'modified_at': metadata['modified_at'],
                'file_size': metadata['file_size'],
                'ingested_via': metadata['ingested_via'],
                'tags': metadata['tags']
            }
            
            query = """
            CREATE (d:Document)
            SET d += $properties, d.updated_at = datetime()
            RETURN d.id as id
            """
            
            result = session.run(query, properties=properties)
            record = result.single()
            return record['id'] if record else node_id
    
    def _create_entity_node(self, entity: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Create entity node in Neo4j."""
        with self.db_manager.neo4j.session() as session:
            entity_type = self._get_entity_type(entity['label'])
            
            # Use canonical name for ID generation to ensure consistent entity resolution
            canonical_name = entity.get('canonical_name', entity['text'])
            node_id = self._generate_node_id(canonical_name, entity_type)
            
            properties = {
                'id': node_id,
                'name': entity['text'],
                'canonical_name': canonical_name,
                'entity_type': entity['label'],
                'created_at': metadata['created_at'],
                'modified_at': metadata['modified_at'],
                'source': metadata['source'],
                'confidence': entity.get('confidence', 1.0)
            }
            
            # Create or update node based on entity type
            if entity_type == 'Person':
                query = """
                MERGE (p:Person {id: $id})
                SET p += $properties, p.updated_at = datetime()
                RETURN p.id as id
                """
            elif entity_type == 'Organization':
                query = """
                MERGE (o:Organization {id: $id})
                SET o += $properties, o.updated_at = datetime()
                RETURN o.id as id
                """
            else:
                query = f"""
                MERGE (e:{entity_type} {{id: $id}})
                SET e += $properties, e.updated_at = datetime()
                RETURN e.id as id
                """
            
            result = session.run(query, id=node_id, properties=properties)
            record = result.single()
            return record['id'] if record else node_id
    
    def _create_relationship(self, relationship: Dict[str, Any]) -> bool:
        """Create relationship between entities."""
        with self.db_manager.neo4j.session() as session:
            # Use canonical names if available for consistent entity resolution
            source_name = relationship.get('source_canonical', relationship['source'])
            target_name = relationship.get('target_canonical', relationship['target'])
            
            source_id = self._generate_node_id(source_name, 'Person')
            target_id = self._generate_node_id(target_name, 'Person')
            rel_type = relationship['type']
            properties = relationship.get('properties', {})
            
            # Prepare properties for MERGE
            confidence = properties.get('confidence', 1.0)
            source_file = properties.get('source_file', '')
            
            query = f"""
            MATCH (a {{id: $source_id}})
            MATCH (b {{id: $target_id}})
            MERGE (a)-[r:{rel_type}]->(b)
            ON CREATE SET 
                r.created_at = datetime(),
                r.confidence = $confidence,
                r.source_file = $source_file
            ON MATCH SET 
                r.updated_at = datetime(),
                r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END
            RETURN r
            """
            
            try:
                result = session.run(query, 
                    source_id=source_id, 
                    target_id=target_id, 
                    confidence=confidence,
                    source_file=source_file
                )
                return result.single() is not None
            except Exception as e:
                logger.error(f"Failed to create relationship: {e}")
                return False
    
    def _link_entities_to_document(self, entity_ids: List[str], doc_id: str):
        """Link entities to document."""
        with self.db_manager.neo4j.session() as session:
            for entity_id in entity_ids:
                query = """
                MATCH (e {id: $entity_id})
                MATCH (d:Document {id: $doc_id})
                MERGE (e)-[r:MENTIONED_IN]->(d)
                SET r.created_at = datetime()
                """
                session.run(query, entity_id=entity_id, doc_id=doc_id)
    
    def _get_entity_type(self, spacy_label: str) -> str:
        """Map spaCy labels to node types."""
        mapping = {
            'PERSON': 'Person',
            'ORG': 'Organization',
            'GPE': 'Location',
            'EVENT': 'Event',
            'PRODUCT': 'Product',
            'WORK_OF_ART': 'Document',
            'LAW': 'Document',
            'LANGUAGE': 'Concept',
            'DATE': 'Date',
            'TIME': 'Time'
        }
        return mapping.get(spacy_label, 'Entity')
    
    def _generate_node_id(self, text: str, entity_type: str) -> str:
        """Generate consistent node ID."""
        import re
        import hashlib
        
        normalized = re.sub(r'[^\w\s-]', '', text.lower())
        normalized = re.sub(r'\s+', '_', normalized.strip())
        content_hash = hashlib.sha256(normalized.encode()).hexdigest()
        return f"{entity_type.lower()}_{content_hash[:12]}"


# Global content ingestor instance
content_ingestor = None


def get_content_ingestor():
    """Get or create content ingestor instance."""
    global content_ingestor
    if content_ingestor is None:
        content_ingestor = ContentIngestor()
    return content_ingestor 