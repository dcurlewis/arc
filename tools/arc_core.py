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

# Load .env file from the project root at the top of the module
# This ensures environment variables are set before any other code runs.
dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.is_file():
    load_dotenv(dotenv_path=dotenv_path)


class ARCConfig:
    """Configuration management for ARC system."""
    
    def __init__(self, config_path: Optional[str] = None):
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

        self.neo4j_uri = self.config.get('neo4j.uri', 'neo4j://localhost:7687')
        self.neo4j_user = self.config.get('neo4j.auth.username', 'neo4j')
        # self.neo4j_password = os.getenv('NEO4J_PASSWORD') # Defer loading to property
        self.neo4j_db = self.config.get('neo4j.database', 'arc')
        self.neo4j_embedded = self.config.get('neo4j.embedded', False) # Set to false
        self.neo4j_dir = Path(self.config.get('neo4j.home', 'data/neo4j'))
        
        self.chroma_path = self.config.get('chromadb.path', 'data/chroma')
        self.embedding_model_name = self.config.get('embeddings.model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.chromadb_collection_documents = self.config.get('chromadb.collection_documents', 'documents')
        self._neo4j_driver = None
        self._chromadb_client = None
        self._nlp = None
        self._embeddings_model = None
    
    @property
    def neo4j(self) -> neo4j.Driver:
        """Return a real Neo4j driver instance.

        This method does *not* provide an in-memory fallback – if the
        connection or authentication fails the exception will propagate so
        that tests or calling code fail loudly (as they should when the real
        database isn’t available).
        """
        if self._neo4j_driver is None:
            from neo4j import GraphDatabase

            # Prefer environment variable over config so developers can simply
            # `export NEO4J_PASSWORD=…`.
            password = (
                os.getenv("NEO4J_PASSWORD")
                or self.config.get("neo4j.password")
                or self.config.get("neo4j.auth.password")
            )

            if not password:
                raise ValueError(
                    "Neo4j password not provided. Set NEO4J_PASSWORD env var or configure 'neo4j.password' in config.yaml or test fixture."
                )

            auth = (self.neo4j_user, password)
            self._neo4j_driver = GraphDatabase.driver(self.neo4j_uri, auth=auth)

        return self._neo4j_driver
    
    @property
    def chromadb(self) -> chromadb.ClientAPI:
        """Get ChromaDB client instance."""
        if self._chromadb_client is None:
            # Resolve path relative to project root (two levels up from this file)
            raw_path = self.config.get('chromadb.path', 'data/chroma')
            path_obj = Path(raw_path)
            if not path_obj.is_absolute():
                project_root = Path(__file__).parent.parent  # arc/
                path_obj = project_root / path_obj
            path_obj.mkdir(parents=True, exist_ok=True)
            self._chromadb_client = chromadb.PersistentClient(path=str(path_obj))
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
        """Return the production SentenceTransformer model.

        No stub fallback – if the model cannot be downloaded or loaded the
        exception will surface so that tests fail, signalling a real problem.
        """
        if self._embeddings_model is None:
            model_name = self.config.get('embeddings.model', 'sentence-transformers/all-MiniLM-L6-v2')
            self._embeddings_model = SentenceTransformer(model_name)
        return self._embeddings_model
    
    def close(self):
        """Close all database connections."""
        if self._neo4j_driver:
            self._neo4j_driver.close()

    def get_neo4j_session(self, **kwargs):
        """Return a Neo4j session. This is a thin wrapper around ``self.neo4j.session`` so
        that tests can simply do ``with db_manager.get_neo4j_session():`` without
        accessing the underlying driver directly.
        """
        return self.neo4j.session(**kwargs)

    @property
    def chromadb_manager(self):
        """Backwards-compatibility alias expected by older tests. Returns the
        underlying ChromaDB client instance.
        """
        return self.chromadb

    def clear_all_data(self, force: bool = False):
        """Utility to wipe all data from Neo4j and ChromaDB during tests.

        Args:
            force: If ``True`` swallow any exception that occurs (useful in CI
                   when the database might be mocked). If ``False`` the first
                   exception encountered will be re-raised.
        """
        # Wipe Neo4j
        try:
            with self.neo4j.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
        except Exception as exc:
            if not force:
                raise exc
        # Wipe all ChromaDB collections
        try:
            for col in self.chromadb.list_collections():
                self.chromadb.delete_collection(name=col.name)
        except Exception as exc:
            if not force:
                raise exc


# Note: EntityExtractor class removed - consolidated to use EnhancedEntityExtractor only


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
    
    def __init__(self, config: Dict[str, Any]):
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
        """Parse metadata from markdown file with enhanced document context analysis."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get actual file timestamps
        import os
        file_stat = os.stat(file_path)
        file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
        file_ctime = datetime.fromtimestamp(file_stat.st_ctime)
        
        # Use the earlier of creation time and modification time as created_at
        created_at = min(file_ctime, file_mtime)
        
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'directory': file_path.parent.name,
            'content': content,
            'content_hash': ContentHasher.hash_content(content),
            'file_size': len(content),
            'created_at': created_at.isoformat(),
            'modified_at': file_mtime.isoformat()
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
        
        # Enhanced document context analysis
        document_context = self._analyze_document_context(content, metadata)
        metadata.update(document_context)
        
        return metadata
    
    def _analyze_document_context(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract rich context from document content structure."""
        import re
        
        lines = content.split('\n')
        
        context = {
            'document_type': None,
            'meeting_participants': [],
            'topics': [],
            'project_mentions': [],
            'urgency_indicators': [],
            'key_themes': [],
            'document_sections': []
        }
        
        # Analyze first 15 lines for context clues
        header_section = '\n'.join(lines[:15])
        filename = metadata.get('file_name', '').lower()
        title = metadata.get('title', '').lower()
        
        # Detect document type from headers/structure/filename
        if (re.search(r'(meeting|call|sync|standup)', header_section, re.IGNORECASE) or
            'meeting' in filename or 'sync' in filename):
            context['document_type'] = 'meeting'
        elif (re.search(r'(1:1|one.on.one|catch.up)', header_section, re.IGNORECASE) or
              '1on1' in filename or 'one-on-one' in filename):
            context['document_type'] = 'one_on_one'
        elif (re.search(r'(design|spec|architecture|rfc)', header_section, re.IGNORECASE) or
              'design' in filename or 'spec' in filename):
            context['document_type'] = 'design_doc'
        elif (re.search(r'(standup|stand.up|daily)', header_section, re.IGNORECASE) or
              'standup' in filename):
            context['document_type'] = 'standup'
        elif (re.search(r'(retrospective|retro|post.mortem)', header_section, re.IGNORECASE) or
              'retro' in filename):
            context['document_type'] = 'retrospective'
        else:
            context['document_type'] = 'general'
        
        # Extract participants from header (common patterns)
        participant_patterns = [
            r'(?:with|attendees?|participants?):\s*([^\n]+)',
            r'(?:present|attending):\s*([^\n]+)',
            r'(?:people|folks|team):\s*([^\n]+)',
        ]
        
        for pattern in participant_patterns:
            matches = re.findall(pattern, header_section, re.IGNORECASE)
            for match in matches:
                # Parse comma or 'and' separated names
                names = re.split(r',|\sand\s', match)
                cleaned_names = [name.strip() for name in names if len(name.strip()) > 2]
                context['meeting_participants'].extend(cleaned_names)
        
        # Remove duplicates and clean up participants
        context['meeting_participants'] = list(set([
            name for name in context['meeting_participants'] 
            if len(name) > 2 and not re.match(r'^[^a-zA-Z]*$', name)
        ]))
        
        # Extract topics from headers and bullet points
        for line in lines[:20]:  # First 20 lines
            line = line.strip()
            if line.startswith('##') and len(line) > 3:
                topic = line.lstrip('#').strip()
                if len(topic) > 3:
                    context['topics'].append(topic)
            elif line.startswith('- ') and len(line) > 5:
                topic = line[2:].strip()
                if len(topic) > 3 and not re.match(r'^[^a-zA-Z]*$', topic):
                    context['topics'].append(topic)
        
        # Extract project mentions from content
        project_patterns = [
            r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # "Project Name" pattern
            r'\b([A-Z]{2,})\b',  # Acronyms like "API", "GPU"
        ]
        
        for pattern in project_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if (len(match) > 2 and 
                    match not in context['project_mentions'] and
                    not re.match(r'^(THE|AND|FOR|WITH|FROM|THIS|THAT)$', match.upper())):
                    context['project_mentions'].append(match)
        
        # Extract urgency/priority indicators
        urgency_patterns = [
            r'\b(urgent|asap|critical|high.priority|blocker|blocked)\b',
            r'\b(deadline|due|needs.to.be.done)\b',
            r'\b(emergency|crisis|immediate)\b'
        ]
        
        for pattern in urgency_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                context['urgency_indicators'].append('high_priority')
                break
        
        # Extract key themes from frequent words (simple approach)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        word_freq = {}
        for word in words:
            if word not in ['this', 'that', 'with', 'from', 'they', 'were', 'been', 'have', 'will', 'would', 'could', 'should']:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top 5 most frequent meaningful words
        top_themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        context['key_themes'] = [theme[0] for theme in top_themes if theme[1] > 2]
        
        # Extract document sections (headers)
        current_section = None
        for line in lines:
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                section_title = line.lstrip('#').strip()
                if section_title:
                    context['document_sections'].append({
                        'level': level,
                        'title': section_title
                    })
        
        # Limit arrays to reasonable sizes
        context['meeting_participants'] = context['meeting_participants'][:10]
        context['topics'] = context['topics'][:10]
        context['project_mentions'] = context['project_mentions'][:15]
        context['key_themes'] = context['key_themes'][:8]
        context['document_sections'] = context['document_sections'][:20]
        
        return context


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


# Note: get_entity_extractor() function removed - use EnhancedEntityExtractor directly


def cleanup():
    """Cleanup global resources."""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None 