#!/usr/bin/env python3
"""
ARC Migration Script
Imports existing markdown files into the Neo4j and ChromaDB databases.
"""

import sys
import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add tools directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

from arc_core import (
    get_db_manager, get_config, get_entity_extractor, 
    FileProcessor, ContentHasher
)


class ARCMigration:
    """Handles migration of markdown files to ARC system."""
    
    def __init__(self, source_dir: str, validate: bool = False):
        self.source_dir = Path(source_dir)
        self.validate = validate
        self.config = get_config()
        self.db = get_db_manager()
        self.extractor = get_entity_extractor()
        self.processor = FileProcessor(self.config)
        
        self.stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'entities_created': 0,
            'relationships_created': 0,
            'errors': []
        }
    
    def run(self, limit: int = None):
        """Run the migration process."""
        print(f"Starting ARC migration from: {self.source_dir}")
        print(f"Validation mode: {self.validate}")
        
        # Get list of markdown files
        files = self._get_markdown_files()
        print(f"Found {len(files)} markdown files to process")
        
        if limit:
            files = files[:limit]
            print(f"Processing first {len(files)} files (limited)")
        
        # Initialize databases
        self._initialize_databases()
        
        # Process files
        for i, file_path in enumerate(files, 1):
            print(f"\nProcessing ({i}/{len(files)}): {file_path.name}")
            try:
                self._process_file(file_path)
                self.stats['files_processed'] += 1
            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {str(e)}"
                print(f"  ✗ {error_msg}")
                self.stats['errors'].append(error_msg)
                self.stats['files_skipped'] += 1
        
        # Print final statistics
        self._print_stats()
        
        return self.stats['errors'] == []
    
    def _get_markdown_files(self) -> List[Path]:
        """Get list of markdown files to process."""
        files = []
        for path in self.source_dir.rglob('*.md'):
            if path.is_file():
                files.append(path)
        return sorted(files)
    
    def _initialize_databases(self):
        """Initialize database schemas and collections."""
        print("\nInitializing databases...")
        
        # Initialize Neo4j constraints and indexes
        with self.db.neo4j.session() as session:
            # Create constraints for unique IDs
            constraints = [
                "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE", 
                "CREATE CONSTRAINT project_id IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
                "CREATE CONSTRAINT meeting_id IF NOT EXISTS FOR (m:Meeting) REQUIRE m.id IS UNIQUE",
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT task_id IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception:
                    pass  # Constraint may already exist
        
        # Initialize ChromaDB collections
        try:
            self.db.chromadb.get_collection("documents")
        except:
            self.db.chromadb.create_collection("documents")
        
        try:
            self.db.chromadb.get_collection("summaries") 
        except:
            self.db.chromadb.create_collection("summaries")
        
        print("  ✓ Database schemas initialized")
    
    def _process_file(self, file_path: Path):
        """Process a single markdown file."""
        # Parse file metadata
        metadata = self.processor.parse_markdown_metadata(file_path)
        
        if self.validate:
            print(f"  ✓ File parsed (validation mode)")
            return
        
        # Extract entities from content
        entities = self.extractor.extract_entities(metadata['content'])
        print(f"  Found entities: {sum(len(v) for v in entities.values())}")
        
        # Create document node
        doc_id = self._create_document_node(metadata)
        
        # Create entity nodes and relationships
        self._create_entities_and_relationships(doc_id, entities, metadata)
        
        # Store in ChromaDB for semantic search
        self._store_in_chromadb(doc_id, metadata)
        
        print(f"  ✓ Processed successfully")
    
    def _create_document_node(self, metadata: Dict[str, Any]) -> str:
        """Create a document node in Neo4j."""
        doc_id = str(uuid.uuid4())
        
        with self.db.neo4j.session() as session:
            session.run("""
                CREATE (d:Document {
                    id: $id,
                    title: $title,
                    type: $type,
                    date_created: datetime($date),
                    source_file: $source_file,
                    content_hash: $content_hash,
                    file_size: $file_size,
                    created_at: datetime($created_at)
                })
            """, 
                id=doc_id,
                title=metadata['title'],
                type=self._determine_document_type(metadata),
                date=metadata.get('date', metadata['created_at']),
                source_file=metadata['file_path'],
                content_hash=metadata['content_hash'],
                file_size=metadata['file_size'],
                created_at=metadata['created_at']
            )
        
        return doc_id
    
    def _determine_document_type(self, metadata: Dict[str, Any]) -> str:
        """Determine document type based on directory and filename."""
        directory = metadata['directory'].lower()
        filename = metadata['file_name'].lower()
        
        if 'meeting' in directory or 'meeting' in filename:
            return 'meeting'
        elif 'decision' in directory or 'decision' in filename:
            return 'decision'
        elif 'project' in directory or 'project' in filename:
            return 'project'
        elif 'strategic' in directory or 'strategic' in filename:
            return 'strategy'
        elif 'team' in directory or 'team' in filename:
            return 'team'
        else:
            return 'document'
    
    def _create_entities_and_relationships(self, doc_id: str, entities: Dict[str, List], metadata: Dict[str, Any]):
        """Create entity nodes and relationships."""
        with self.db.neo4j.session() as session:
            # Create person entities
            for person in entities.get('persons', []):
                person_id = self._create_or_get_person(session, person)
                if person_id:
                    # Link person to document
                    session.run("""
                        MATCH (d:Document {id: $doc_id}), (p:Person {id: $person_id})
                        CREATE (d)-[:MENTIONS]->(p)
                    """, doc_id=doc_id, person_id=person_id)
                    self.stats['relationships_created'] += 1
            
            # Create organization entities
            for org in entities.get('organizations', []):
                org_id = self._create_or_get_organization(session, org)
                if org_id:
                    # Link organization to document
                    session.run("""
                        MATCH (d:Document {id: $doc_id}), (c:Company {id: $org_id})
                        CREATE (d)-[:REFERENCES]->(c)
                    """, doc_id=doc_id, org_id=org_id)
                    self.stats['relationships_created'] += 1
            
            # Create meeting node if this is a meeting document
            if metadata['directory'].lower() == 'meeting-insights':
                meeting_id = self._create_meeting_node(session, metadata, doc_id)
                if meeting_id:
                    # Link participants to meeting
                    for person in entities.get('persons', []):
                        person_id = self._create_or_get_person(session, person)
                        if person_id:
                            session.run("""
                                MATCH (m:Meeting {id: $meeting_id}), (p:Person {id: $person_id})
                                CREATE (p)-[:PARTICIPATED_IN]->(m)
                            """, meeting_id=meeting_id, person_id=person_id)
                            self.stats['relationships_created'] += 1
    
    def _create_or_get_person(self, session, person_entity: Dict[str, Any]) -> str:
        """Create or get existing person entity."""
        name = person_entity['text'].strip()
        normalized_name = person_entity.get('normalized', name.lower())
        
        # Check if person already exists
        result = session.run("""
            MATCH (p:Person)
            WHERE p.name = $name OR $normalized IN p.aliases
            RETURN p.id as id
        """, name=name, normalized=normalized_name)
        
        existing = result.single()
        if existing:
            return existing['id']
        
        # Create new person
        person_id = str(uuid.uuid4())
        session.run("""
            CREATE (p:Person {
                id: $id,
                name: $name,
                aliases: [$normalized],
                created_at: datetime($created_at),
                updated_at: datetime($created_at)
            })
        """, 
            id=person_id,
            name=name,
            normalized=normalized_name,
            created_at=datetime.now().isoformat()
        )
        
        self.stats['entities_created'] += 1
        return person_id
    
    def _create_or_get_organization(self, session, org_entity: Dict[str, Any]) -> str:
        """Create or get existing organization entity."""
        name = org_entity['text'].strip()
        
        # Check if organization already exists
        result = session.run("""
            MATCH (c:Company)
            WHERE c.name = $name
            RETURN c.id as id
        """, name=name)
        
        existing = result.single()
        if existing:
            return existing['id']
        
        # Create new organization
        org_id = str(uuid.uuid4())
        session.run("""
            CREATE (c:Company {
                id: $id,
                name: $name,
                type: "unknown",
                created_at: datetime($created_at)
            })
        """, 
            id=org_id,
            name=name,
            created_at=datetime.now().isoformat()
        )
        
        self.stats['entities_created'] += 1
        return org_id
    
    def _create_meeting_node(self, session, metadata: Dict[str, Any], doc_id: str) -> str:
        """Create a meeting node."""
        meeting_id = str(uuid.uuid4())
        
        session.run("""
            CREATE (m:Meeting {
                id: $id,
                title: $title,
                date: datetime($date),
                type: "meeting",
                source_file: $source_file
            })
        """,
            id=meeting_id,
            title=metadata['title'],
            date=metadata.get('date', metadata['created_at']),
            source_file=metadata['file_path']
        )
        
        # Link meeting to document
        session.run("""
            MATCH (m:Meeting {id: $meeting_id}), (d:Document {id: $doc_id})
            CREATE (m)-[:DOCUMENTED_IN]->(d)
        """, meeting_id=meeting_id, doc_id=doc_id)
        
        self.stats['entities_created'] += 1
        self.stats['relationships_created'] += 1
        return meeting_id
    
    def _store_in_chromadb(self, doc_id: str, metadata: Dict[str, Any]):
        """Store document in ChromaDB for semantic search."""
        collection = self.db.chromadb.get_collection("documents")
        
        # Generate embedding
        content = metadata['content']
        embedding = self.db.embeddings.encode([content])
        
        # Store document
        collection.add(
            ids=[doc_id],
            embeddings=embedding.tolist(),
            documents=[content],
            metadatas=[{
                'title': metadata['title'],
                'type': metadata['directory'],
                'date': metadata.get('date', metadata['created_at']),
                'source_file': metadata['file_path'],
                'file_size': metadata['file_size']
            }]
        )
    
    def _print_stats(self):
        """Print final migration statistics."""
        print("\n" + "=" * 50)
        print("Migration Complete!")
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files skipped: {self.stats['files_skipped']}")
        print(f"Entities created: {self.stats['entities_created']}")
        print(f"Relationships created: {self.stats['relationships_created']}")
        
        if self.stats['errors']:
            print(f"\nErrors ({len(self.stats['errors'])}):")
            for error in self.stats['errors']:
                print(f"  - {error}")
        else:
            print("\n🎉 No errors encountered!")


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate markdown files to ARC system')
    parser.add_argument('--source', required=True, help='Source directory containing markdown files')
    parser.add_argument('--validate', action='store_true', help='Validation mode (no database changes)')
    parser.add_argument('--limit', type=int, help='Limit number of files to process (for testing)')
    
    args = parser.parse_args()
    
    if not Path(args.source).exists():
        print(f"Error: Source directory '{args.source}' does not exist")
        sys.exit(1)
    
    try:
        migration = ARCMigration(args.source, args.validate)
        success = migration.run(args.limit)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 