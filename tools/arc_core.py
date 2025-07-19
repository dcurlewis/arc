"""
ARC Core Module
Shared utilities and database connections for the ARC system.
"""

import os
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

import neo4j
import chromadb
import spacy
from sentence_transformers import SentenceTransformer
import yaml
from dotenv import load_dotenv


class ARCConfig:
    """Configuration management for ARC system."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load environment variables from .env file
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
        
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables, YAML file, or use defaults."""
        # Start with defaults
        config = {
            "neo4j": {
                "uri": "neo4j://localhost:7687",
                "embedded": True,
                "database": "arc"
            },
            "chromadb": {
                "path": "./data/chromadb",
                "collection_documents": "documents",
                "collection_summaries": "summaries"
            },
            "spacy": {
                "model": "en_core_web_lg"
            },
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "import": {
                "source_dir": "./import",
                "backup_dir": "./backups"
            }
        }
        
        # Load from YAML file if it exists
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                # Merge with defaults
                if loaded_config:
                    config.update(loaded_config)
        
        # Override with environment variables (highest priority)
        # Neo4j configuration
        if os.getenv('NEO4J_URI'):
            config['neo4j']['uri'] = os.getenv('NEO4J_URI')
        if os.getenv('NEO4J_USER'):
            config['neo4j']['user'] = os.getenv('NEO4J_USER')
        if os.getenv('NEO4J_PASSWORD'):
            config['neo4j']['password'] = os.getenv('NEO4J_PASSWORD')
        
        # ChromaDB configuration
        if os.getenv('CHROMADB_PATH'):
            config['chromadb']['path'] = os.getenv('CHROMADB_PATH')
        
        # spaCy configuration
        if os.getenv('SPACY_MODEL'):
            config['spacy']['model'] = os.getenv('SPACY_MODEL')
        
        # Import configuration
        if os.getenv('IMPORT_SOURCE_DIR'):
            config['import']['source_dir'] = os.getenv('IMPORT_SOURCE_DIR')
        
        return config
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation (e.g., 'neo4j.uri')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


class DatabaseManager:
    """Manages connections to Neo4j and ChromaDB."""
    
    def __init__(self, config: ARCConfig):
        self.config = config
        self._neo4j_driver = None
        self._chromadb_client = None
        self._nlp = None
        self._embeddings_model = None
    
    @property
    def neo4j(self) -> neo4j.Driver:
        """Get Neo4j driver instance."""
        if self._neo4j_driver is None:
            from neo4j import GraphDatabase
            uri = self.config.get('neo4j.uri', 'bolt://localhost:7687')
            
            # Always use authentication - Neo4j requires it
            auth = (
                self.config.get('neo4j.user', 'neo4j'),
                self.config.get('neo4j.password', 'neo4j')
            )
            self._neo4j_driver = GraphDatabase.driver(uri, auth=auth)
        return self._neo4j_driver
    
    @property
    def chromadb(self) -> chromadb.ClientAPI:
        """Get ChromaDB client instance."""
        if self._chromadb_client is None:
            persist_directory = self.config.get('chromadb.path', './data/chromadb')
            os.makedirs(persist_directory, exist_ok=True)
            self._chromadb_client = chromadb.PersistentClient(path=persist_directory)
        return self._chromadb_client
    
    @property
    def nlp(self) -> spacy.Language:
        """Get spaCy NLP model."""
        if self._nlp is None:
            model_name = self.config.get('spacy.model', 'en_core_web_lg')
            self._nlp = spacy.load(model_name)
        return self._nlp
    
    @property
    def embeddings(self) -> SentenceTransformer:
        """Get sentence transformer model."""
        if self._embeddings_model is None:
            model_name = self.config.get('embeddings.model', 'sentence-transformers/all-MiniLM-L6-v2')
            self._embeddings_model = SentenceTransformer(model_name)
        return self._embeddings_model
    
    def close(self):
        """Close all database connections."""
        if self._neo4j_driver:
            self._neo4j_driver.close()


class EntityExtractor:
    """Extracts entities from text using spaCy."""
    
    def __init__(self, nlp_model: spacy.Language):
        self.nlp = nlp_model
        # Custom entity resolution rules based on your requirements
        self.name_aliases = {
            "nick": "nicholas",
            "ze chen": "ze chen",  # Never abbreviate to "Z"
            "glenn": "glen",  # Glen not Glenn
            "jeff zhu": "jeff zhu",  # Different from Geoff Breemer
            "geoff breemer": "geoff breemer",  # Evaluation team
            "ping": "peng",  # Common transcription error
            "artur mogozov": "artur mogozov",  # Not Arthur
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities from text."""
        doc = self.nlp(text)
        entities = {
            'persons': [],
            'organizations': [],
            'topics': [],
            'dates': []
        }
        
        for ent in doc.ents:
            entity_data = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': getattr(ent, 'confidence', 1.0)
            }
            
            if ent.label_ in ['PERSON']:
                # Apply name normalization
                normalized_name = self._normalize_name(ent.text.lower())
                entity_data['normalized'] = normalized_name
                entities['persons'].append(entity_data)
            elif ent.label_ in ['ORG', 'COMPANY']:
                entities['organizations'].append(entity_data)
            elif ent.label_ in ['DATE', 'TIME']:
                entities['dates'].append(entity_data)
            elif ent.label_ in ['TOPIC', 'PRODUCT', 'EVENT']:
                entities['topics'].append(entity_data)
        
        return entities
    
    def _normalize_name(self, name: str) -> str:
        """Normalize person names based on known aliases."""
        name_lower = name.lower().strip()
        return self.name_aliases.get(name_lower, name_lower)


class ContentHasher:
    """Generates content hashes for deduplication."""
    
    @staticmethod
    def hash_content(content: str) -> str:
        """Generate SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    @staticmethod
    def hash_file(file_path: str) -> str:
        """Generate hash of file content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return ContentHasher.hash_content(content)


class FileProcessor:
    """Processes markdown files from the import directory."""
    
    def __init__(self, config: ARCConfig):
        self.config = config
        self.source_dir = Path(config.get('import.source_dir', './import'))
    
    def list_markdown_files(self) -> List[Path]:
        """List all markdown files in the import directory."""
        markdown_files = []
        for path in self.source_dir.rglob('*.md'):
            if path.is_file():
                markdown_files.append(path)
        return sorted(markdown_files)
    
    def parse_markdown_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Parse metadata from markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'directory': file_path.parent.name,
            'content': content,
            'content_hash': ContentHasher.hash_content(content),
            'file_size': len(content),
            'created_at': datetime.now().isoformat()
        }
        
        # Extract date from filename (YYYYMMDD-*.md pattern)
        filename = file_path.stem
        if len(filename) >= 8 and filename[:8].isdigit():
            try:
                date_str = filename[:8]
                metadata['date'] = datetime.strptime(date_str, '%Y%m%d').isoformat()
            except ValueError:
                pass
        
        # Extract title from first header or filename
        lines = content.split('\n')
        title = None
        for line in lines:
            if line.startswith('# '):
                title = line[2:].strip()
                break
        
        metadata['title'] = title or filename.replace('-', ' ').title()
        
        return metadata


# Global instances (initialized when needed)
_config = None
_db_manager = None


def get_config() -> ARCConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = ARCConfig()
    return _config


def get_db_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(get_config())
    return _db_manager


def get_entity_extractor() -> EntityExtractor:
    """Get entity extractor instance."""
    db = get_db_manager()
    return EntityExtractor(db.nlp)


def cleanup():
    """Cleanup global resources."""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None 