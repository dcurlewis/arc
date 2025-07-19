"""
ARC Import Pipeline
Handles importing markdown files into the Neo4j knowledge graph and ChromaDB vector store.
"""

import re
import os
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass

from arc_core import get_config, get_db_manager, FileProcessor, EntityExtractor, ContentHasher
from enhanced_entity_extractor import EnhancedEntityExtractor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImportStats:
    """Statistics for import operations."""
    files_processed: int = 0
    files_skipped: int = 0
    entities_created: int = 0
    relationships_created: int = 0
    documents_indexed: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class EntityRelationshipExtractor:
    """Extracts entities and relationships from markdown content."""
    
    def __init__(self, config):
        self.config = config
        self.nlp = None
        self._load_nlp_model()
    
    def _load_nlp_model(self):
        """Load spaCy NLP model."""
        try:
            import spacy
            model_name = self.config.get('spacy.model', 'en_core_web_lg')
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise
    
    def extract_entities(self, text: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Extract entities from text."""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Skip very short entities or common stopwords
            if len(ent.text.strip()) < 2:
                continue
                
            entity = {
                'text': ent.text.strip(),
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': getattr(ent, 'confidence', 1.0)
            }
            
            # Apply disambiguation rules
            entity = self._apply_disambiguation_rules(entity)
            
            # Add context if provided
            if context:
                entity['context'] = context
                
            entities.append(entity)
        
        return self._deduplicate_entities(entities)
    
    def _apply_disambiguation_rules(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Apply disambiguation rules from config."""
        rules = self.config.get('entity.disambiguation_rules', {})
        
        text = entity['text']
        if text in rules:
            entity['canonical_name'] = rules[text]
            entity['text'] = rules[text]
        
        return entity
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities, keeping the longest/most confident."""
        seen = {}
        
        for entity in entities:
            key = entity['text'].lower()
            if key not in seen or entity.get('confidence', 1.0) > seen[key].get('confidence', 1.0):
                seen[key] = entity
        
        return list(seen.values())
    
    def infer_relationships(self, entities: List[Dict[str, Any]], text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Infer relationships between entities based on context."""
        relationships = []
        
        # Meeting participant relationships
        if 'meeting' in metadata.get('title', '').lower() or 'meeting' in text.lower():
            people = [e for e in entities if e['label'] == 'PERSON']
            for i, person1 in enumerate(people):
                for person2 in people[i+1:]:
                    relationships.append({
                        'source': person1['text'],
                        'target': person2['text'],
                        'type': 'ATTENDED_MEETING_WITH',
                        'properties': {
                            'meeting_date': metadata.get('date'),
                            'meeting_title': metadata.get('title')
                        }
                    })
        
        # Person-Organization relationships
        people = [e for e in entities if e['label'] == 'PERSON']
        orgs = [e for e in entities if e['label'] == 'ORG']
        
        for person in people:
            for org in orgs:
                # Look for employment/affiliation indicators
                person_start = person['start']
                org_start = org['start']
                
                # Check if person and org are close in the text
                distance = abs(person_start - org_start)
                if distance < 200:  # Within ~200 characters
                    # Look for work-related keywords between them
                    text_between = text[min(person_start, org_start):max(person['end'], org['end'])]
                    work_keywords = ['works at', 'employed by', 'from', 'at', 'with', '(', ')']
                    
                    if any(keyword in text_between.lower() for keyword in work_keywords):
                        relationships.append({
                            'source': person['text'],
                            'target': org['text'],
                            'type': 'AFFILIATED_WITH',
                            'properties': {
                                'source_file': metadata.get('file_name'),
                                'confidence': 0.8
                            }
                        })
        
        return relationships


class GraphManager:
    """Manages Neo4j graph operations for import."""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.driver = db_manager.neo4j
    
    def create_constraints(self):
        """Create necessary constraints and indexes."""
        with self.driver.session() as session:
            # Create constraints for unique entities
            constraints = [
                "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT org_id IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE", 
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT meeting_id IF NOT EXISTS FOR (m:Meeting) REQUIRE m.id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.debug(f"Created constraint: {constraint}")
                except Exception as e:
                    logger.debug(f"Constraint already exists or failed: {e}")
    
    def create_entity_node(self, entity: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Create entity node in graph."""
        with self.driver.session() as session:
            entity_type = self._get_entity_type(entity['label'])
            node_id = self._generate_node_id(entity['text'], entity_type)
            
            # Prepare properties  
            properties = {
                'id': node_id,
                'name': entity['text'],
                'canonical_name': entity.get('canonical_name', entity['text']),
                'entity_type': entity['label'],
                'created_at': metadata.get('created_at', datetime.now().isoformat()),
                'modified_at': metadata.get('modified_at', datetime.now().isoformat()),
                'source_file': metadata.get('file_name'),
                'confidence': entity.get('confidence', 1.0)
            }
            
            # Create or update node
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
                # Generic entity
                query = f"""
                MERGE (e:{entity_type} {{id: $id}})
                SET e += $properties, e.updated_at = datetime()
                RETURN e.id as id
                """
            
            result = session.run(query, id=node_id, properties=properties)
            record = result.single()
            return record['id'] if record else node_id
    
    def create_document_node(self, metadata: Dict[str, Any]) -> str:
        """Create document node in graph."""
        with self.driver.session() as session:
            doc_id = self._generate_node_id(metadata['file_name'], 'Document')
            
            properties = {
                'id': doc_id,
                'title': metadata.get('title', ''),
                'file_name': metadata['file_name'],
                'file_path': metadata['file_path'],
                'content_hash': metadata['content_hash'],
                'date': metadata.get('date'),
                'created_at': metadata.get('created_at', datetime.now().isoformat()),
                'modified_at': metadata.get('modified_at', datetime.now().isoformat()),
                'file_size': metadata.get('file_size', 0)
            }
            
            query = """
            MERGE (d:Document {id: $id})
            SET d += $properties, d.updated_at = datetime()
            RETURN d.id as id
            """
            
            result = session.run(query, id=doc_id, properties=properties)
            record = result.single()
            return record['id'] if record else doc_id
    
    def create_relationship(self, relationship: Dict[str, Any]) -> bool:
        """Create relationship between entities."""
        with self.driver.session() as session:
            source_id = self._generate_node_id(relationship['source'], 'Person')
            target_id = self._generate_node_id(relationship['target'], 'Person')
            rel_type = relationship['type']
            properties = relationship.get('properties', {})
            
            # Add metadata
            properties.update({
                'created_at': datetime.now().isoformat(),
                'confidence': properties.get('confidence', 1.0)
            })
            
            query = f"""
            MATCH (a {{id: $source_id}})
            MATCH (b {{id: $target_id}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += $properties, r.updated_at = datetime()
            RETURN r
            """
            
            try:
                result = session.run(query, source_id=source_id, target_id=target_id, properties=properties)
                return result.single() is not None
            except Exception as e:
                logger.error(f"Failed to create relationship {rel_type} between {relationship['source']} and {relationship['target']}: {e}")
                return False
    
    def link_entities_to_document(self, entity_ids: List[str], doc_id: str):
        """Link entities to document they appear in."""
        with self.driver.session() as session:
            for entity_id in entity_ids:
                query = """
                MATCH (e {id: $entity_id})
                MATCH (d:Document {id: $doc_id})
                MERGE (e)-[r:MENTIONED_IN]->(d)
                SET r.created_at = datetime()
                """
                session.run(query, entity_id=entity_id, doc_id=doc_id)
    
    def _get_entity_type(self, spacy_label: str) -> str:
        """Map spaCy labels to our node types."""
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
            'TIME': 'Time',
            'PERCENT': 'Metric',
            'MONEY': 'Metric',
            'QUANTITY': 'Metric',
            'CARDINAL': 'Number',
            'ORDINAL': 'Number'
        }
        return mapping.get(spacy_label, 'Entity')
    
    def _generate_node_id(self, text: str, entity_type: str) -> str:
        """Generate consistent node ID."""
        # Normalize text for ID generation
        normalized = re.sub(r'[^\w\s-]', '', text.lower())
        normalized = re.sub(r'\s+', '_', normalized.strip())
        return f"{entity_type.lower()}_{ContentHasher.hash_content(normalized)[:12]}"


class VectorManager:
    """Manages ChromaDB vector operations for import."""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.client = db_manager.chromadb
        self.embedding_model = db_manager.embeddings
    
    def index_document(self, metadata: Dict[str, Any]) -> bool:
        """Index document content in ChromaDB."""
        try:
            # Get or create collection
            collection_name = self.db_manager.config.get('chromadb.document_collection', 'documents')
            collection = self.client.get_or_create_collection(collection_name)
            
            # Prepare document for indexing
            doc_id = metadata['content_hash']
            content = metadata['content']
            
            # Create metadata for ChromaDB (must be JSON serializable)
            # Convert datetime strings to timestamps for temporal filtering
            created_at_ts = None
            modified_at_ts = None
            
            if metadata.get('created_at'):
                try:
                    created_at_dt = datetime.fromisoformat(metadata['created_at'])
                    created_at_ts = created_at_dt.timestamp()
                except (ValueError, TypeError):
                    pass
            
            if metadata.get('modified_at'):
                try:
                    modified_at_dt = datetime.fromisoformat(metadata['modified_at'])
                    modified_at_ts = modified_at_dt.timestamp()
                except (ValueError, TypeError):
                    pass
            
            chroma_metadata = {
                'file_name': metadata['file_name'],
                'title': metadata.get('title', ''),
                'date': metadata.get('date'),
                'file_size': metadata.get('file_size', 0),
                'created_at': created_at_ts,
                'modified_at': modified_at_ts,
                'created_at_iso': metadata.get('created_at'),
                'modified_at_iso': metadata.get('modified_at')
            }
            
            # Remove None values
            chroma_metadata = {k: v for k, v in chroma_metadata.items() if v is not None}
            
            # Add to collection
            collection.add(
                documents=[content],
                metadatas=[chroma_metadata],
                ids=[doc_id]
            )
            
            logger.debug(f"Indexed document: {metadata['file_name']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index document {metadata.get('file_name', 'unknown')}: {e}")
            return False


class ARCImporter:
    """Main import orchestrator."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.db_manager = get_db_manager()
        self.file_processor = FileProcessor(self.config)
        
        # Load enhanced entity configuration
        enhanced_config = self._load_enhanced_config()
        self.entity_extractor = EnhancedEntityExtractor(enhanced_config)
        
        self.graph_manager = GraphManager(self.db_manager)
        self.vector_manager = VectorManager(self.db_manager)
        
        # Initialize enhanced embedding system
        from enhanced_embeddings import create_enhanced_embedding_system
        self.embedding_generator, self.query_interface = create_enhanced_embedding_system(self.db_manager)
        
        # Initialize constraints
        self.graph_manager.create_constraints()
        
        logger.info("Initialized ARCImporter with enhanced entity extraction and embeddings")
    
    def _load_enhanced_config(self) -> Dict[str, Any]:
        """Load enhanced entity extraction configuration."""
        config_path = Path("config/enhanced_entity_config.yaml")
        template_path = Path("config/enhanced_entity_config.template.yaml")
        
        # Try to load the actual config file first
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    enhanced_config = yaml.safe_load(f)
                logger.info(f"Loaded enhanced entity config from {config_path}")
                return enhanced_config
            except Exception as e:
                logger.warning(f"Failed to load enhanced config: {e}")
        
        # Fallback to template if actual config doesn't exist
        elif template_path.exists():
            try:
                with open(template_path, 'r') as f:
                    enhanced_config = yaml.safe_load(f)
                logger.info(f"Loaded enhanced entity config from template: {template_path}")
                logger.info("Consider copying the template to enhanced_entity_config.yaml and customizing it")
                return enhanced_config
            except Exception as e:
                logger.warning(f"Failed to load template config: {e}")
        
        # Return basic config if neither file exists
        logger.warning("No enhanced entity config found, using basic configuration")
        logger.info("To use enhanced entity extraction, copy config/enhanced_entity_config.template.yaml to config/enhanced_entity_config.yaml")
        return self.config
    
    def clear_databases(self):
        """Clear all data from Neo4j and ChromaDB."""
        logger.info("🗑️  Clearing all databases...")
        
        # Clear Neo4j
        try:
            with self.db_manager.neo4j.session() as session:
                # Delete all nodes and relationships
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("✅ Neo4j database cleared")
        except Exception as e:
            logger.error(f"❌ Error clearing Neo4j: {e}")
            raise
        
        # Clear ChromaDB
        try:
            client = self.db_manager.chromadb
            # Get collection name from config
            collection_name = self.config.get('chromadb.collection', 'arc_documents')
            
            try:
                # Try to delete the collection if it exists
                client.delete_collection(collection_name)
                logger.info("✅ ChromaDB collection deleted")
            except Exception:
                # Collection might not exist, which is fine
                logger.info("✅ ChromaDB collection was already empty or non-existent")
                
        except Exception as e:
            logger.error(f"❌ Error clearing ChromaDB: {e}")
            raise
        
        logger.info("🎯 All databases cleared successfully")
    
    def import_file(self, file_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Import a single markdown file."""
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Parse metadata
            metadata = self.file_processor.parse_markdown_metadata(file_path)
            
            # Check if file already processed (by content hash)
            if self._is_file_already_processed(metadata['content_hash']):
                logger.info(f"File already processed (same content hash): {file_path}")
                return True, {'skipped': True, 'reason': 'already_processed'}
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(
                metadata['content'], 
                context={'file_name': metadata['file_name'], 'date': metadata.get('date')}
            )
            
            # Extract relationships
            relationships = self.entity_extractor.infer_relationships(
                entities, metadata['content'], metadata
            )
            
            # Create document node
            doc_id = self.graph_manager.create_document_node(metadata)
            
            # Create entity nodes and collect IDs
            entity_ids = []
            for entity in entities:
                entity_id = self.graph_manager.create_entity_node(entity, metadata)
                entity_ids.append(entity_id)
            
            # Create relationships
            for relationship in relationships:
                self.graph_manager.create_relationship(relationship)
            
            # Link entities to document
            self.graph_manager.link_entities_to_document(entity_ids, doc_id)
            
            # Index in vector store (basic)
            self.vector_manager.index_document(metadata)
            
            # Index with enhanced embeddings
            self.embedding_generator.index_enhanced_document(
                metadata['content'], entities, relationships, metadata
            )
            
            result = {
                'entities_count': len(entities),
                'relationships_count': len(relationships),
                'doc_id': doc_id,
                'entity_ids': entity_ids
            }
            
            logger.info(f"Successfully processed {file_path}: {len(entities)} entities, {len(relationships)} relationships")
            return True, result
            
        except Exception as e:
            logger.error(f"Failed to import file {file_path}: {e}")
            return False, {'error': str(e)}
    
    def import_directory(self, limit: Optional[int] = None) -> ImportStats:
        """Import all markdown files from the configured directory."""
        stats = ImportStats()
        
        # Get all markdown files
        files = self.file_processor.list_markdown_files()
        
        if limit:
            files = files[:limit]
        
        logger.info(f"Starting import of {len(files)} files...")
        
        for file_path in files:
            success, result = self.import_file(file_path)
            
            if success:
                if result.get('skipped'):
                    stats.files_skipped += 1
                else:
                    stats.files_processed += 1
                    stats.entities_created += result.get('entities_count', 0)
                    stats.relationships_created += result.get('relationships_count', 0)
                    stats.documents_indexed += 1
            else:
                stats.errors.append(f"{file_path}: {result.get('error', 'Unknown error')}")
        
        logger.info(f"Import complete: {stats.files_processed} processed, {stats.files_skipped} skipped, {len(stats.errors)} errors")
        return stats
    
    def _is_file_already_processed(self, content_hash: str) -> bool:
        """Check if file with this content hash was already processed."""
        with self.db_manager.neo4j.session() as session:
            result = session.run(
                "MATCH (d:Document {content_hash: $hash}) RETURN d.id",
                hash=content_hash
            )
            return result.single() is not None


def main():
    """Main import function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Import markdown files into ARC knowledge graph')
    parser.add_argument('--limit', type=int, help='Limit number of files to process')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--clear', action='store_true', help='Clear all existing data before importing')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize importer
    importer = ARCImporter()
    
    # Clear databases if requested
    if args.clear:
        importer.clear_databases()
    
    # Run import
    stats = importer.import_directory(limit=args.limit)
    
    # Print results
    print("\n" + "="*60)
    print("ARC IMPORT RESULTS")
    print("="*60)
    print(f"Files processed: {stats.files_processed}")
    print(f"Files skipped: {stats.files_skipped}")
    print(f"Entities created: {stats.entities_created}")
    print(f"Relationships created: {stats.relationships_created}")
    print(f"Documents indexed: {stats.documents_indexed}")
    
    if stats.errors:
        print(f"\nErrors ({len(stats.errors)}):")
        for error in stats.errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more errors")


if __name__ == '__main__':
    main() 