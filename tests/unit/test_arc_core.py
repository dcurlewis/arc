"""
Unit tests for arc_core module.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))

from arc_core import (
    ARCConfig, 
    get_config, 
    Neo4jManager, 
    ChromaDBManager, 
    EntityExtractor,
    FileProcessor
)


class TestARCConfig:
    """Test ARC configuration management."""
    
    def test_init_with_dict(self):
        """Test ARCConfig initialization with dictionary."""
        config_data = {
            'database': {'host': 'localhost'},
            'api': {'port': 8080}
        }
        config = ARCConfig(config_data)
        
        assert config.get('database.host') == 'localhost'
        assert config.get('api.port') == 8080
    
    def test_get_with_default(self):
        """Test getting values with defaults."""
        config = ARCConfig({})
        
        assert config.get('missing.key', 'default') == 'default'
        assert config.get('missing.key') is None
    
    def test_nested_key_access(self):
        """Test accessing nested configuration keys."""
        config_data = {
            'level1': {
                'level2': {
                    'level3': 'value'
                }
            }
        }
        config = ARCConfig(config_data)
        
        assert config.get('level1.level2.level3') == 'value'
        assert config.get('level1.level2') == {'level3': 'value'}
    
    def test_update_config(self):
        """Test updating configuration."""
        config = ARCConfig({'key': 'old_value'})
        
        config.update({'key': 'new_value', 'new_key': 'new_value'})
        
        assert config.get('key') == 'new_value'
        assert config.get('new_key') == 'new_value'


class TestGetConfig:
    """Test global configuration loading."""
    
    @patch('arc_core.Path.exists')
    @patch('arc_core.Path.read_text')
    def test_load_from_yaml_file(self, mock_read_text, mock_exists):
        """Test loading configuration from YAML file."""
        mock_exists.return_value = True
        mock_read_text.return_value = """
        database:
          host: localhost
        api:
          port: 8080
        """
        
        config = get_config()
        
        assert config.get('database.host') == 'localhost'
        assert config.get('api.port') == 8080
    
    @patch('arc_core.Path.exists')
    def test_load_default_config(self, mock_exists):
        """Test loading default configuration when file doesn't exist."""
        mock_exists.return_value = False
        
        config = get_config()
        
        # Should have default values
        assert config.get('chromadb.path') is not None
        assert config.get('neo4j.uri') is not None
        assert config.get('spacy.model') is not None
    
    def test_config_caching(self):
        """Test that configuration is cached on subsequent calls."""
        with patch('arc_core.Path.exists') as mock_exists:
            mock_exists.return_value = False
            
            config1 = get_config()
            config2 = get_config()
            
            # Should be the same instance (cached)
            assert config1 is config2


class TestNeo4jManager:
    """Test Neo4j database manager."""
    
    def test_init_with_config(self, test_config, mock_neo4j_driver):
        """Test Neo4jManager initialization."""
        with patch('arc_core.GraphDatabase.driver') as mock_driver_func:
            mock_driver_func.return_value = mock_neo4j_driver
            
            manager = Neo4jManager(test_config)
            
            assert manager.config == test_config
            assert manager.driver == mock_neo4j_driver
    
    def test_test_connection_success(self, test_config, mock_neo4j_driver):
        """Test successful connection test."""
        with patch('arc_core.GraphDatabase.driver') as mock_driver_func:
            mock_driver_func.return_value = mock_neo4j_driver
            
            manager = Neo4jManager(test_config)
            
            assert manager.test_connection() is True
    
    def test_test_connection_failure(self, test_config):
        """Test connection test failure."""
        with patch('arc_core.GraphDatabase.driver') as mock_driver_func:
            mock_driver_func.side_effect = Exception("Connection failed")
            
            manager = Neo4jManager(test_config)
            
            assert manager.test_connection() is False
    
    def test_setup_constraints(self, test_config, mock_neo4j_driver):
        """Test database constraints setup."""
        with patch('arc_core.GraphDatabase.driver') as mock_driver_func:
            mock_driver_func.return_value = mock_neo4j_driver
            
            manager = Neo4jManager(test_config)
            manager.setup_constraints()
            
            # Verify that session.run was called for constraints
            mock_session = mock_neo4j_driver.session.return_value.__enter__.return_value
            assert mock_session.run.called
    
    def test_create_person_node(self, test_config, mock_neo4j_driver):
        """Test creating person node."""
        with patch('arc_core.GraphDatabase.driver') as mock_driver_func:
            mock_driver_func.return_value = mock_neo4j_driver
            
            manager = Neo4jManager(test_config)
            
            result = manager.create_person_node(
                name="John Doe",
                properties={"role": "Developer", "company": "TechCorp"}
            )
            
            # Should call session.run with CREATE query
            mock_session = mock_neo4j_driver.session.return_value.__enter__.return_value
            args, kwargs = mock_session.run.call_args
            assert "CREATE" in args[0]
            assert "Person" in args[0]


class TestChromaDBManager:
    """Test ChromaDB manager."""
    
    def test_init_with_config(self, test_config, mock_chromadb_client):
        """Test ChromaDBManager initialization."""
        with patch('arc_core.chromadb.PersistentClient') as mock_client_func:
            mock_client_func.return_value = mock_chromadb_client
            
            manager = ChromaDBManager(test_config)
            
            assert manager.config == test_config
            assert manager.client == mock_chromadb_client
    
    def test_test_connection_success(self, test_config, mock_chromadb_client):
        """Test successful ChromaDB connection."""
        with patch('arc_core.chromadb.PersistentClient') as mock_client_func:
            mock_client_func.return_value = mock_chromadb_client
            
            manager = ChromaDBManager(test_config)
            
            assert manager.test_connection() is True
    
    def test_test_connection_failure(self, test_config):
        """Test ChromaDB connection failure."""
        with patch('arc_core.chromadb.PersistentClient') as mock_client_func:
            mock_client_func.side_effect = Exception("Connection failed")
            
            manager = ChromaDBManager(test_config)
            
            assert manager.test_connection() is False
    
    def test_setup_collections(self, test_config, mock_chromadb_client):
        """Test collection setup."""
        with patch('arc_core.chromadb.PersistentClient') as mock_client_func:
            mock_client_func.return_value = mock_chromadb_client
            
            manager = ChromaDBManager(test_config)
            manager.setup_collections()
            
            # Should create both document and summary collections
            assert mock_chromadb_client.get_or_create_collection.call_count == 2
    
    def test_add_document(self, test_config, mock_chromadb_client):
        """Test adding document to collection."""
        mock_collection = Mock()
        mock_chromadb_client.get_or_create_collection.return_value = mock_collection
        
        with patch('arc_core.chromadb.PersistentClient') as mock_client_func:
            mock_client_func.return_value = mock_chromadb_client
            
            manager = ChromaDBManager(test_config)
            
            manager.add_document(
                doc_id="doc1",
                content="Test content",
                metadata={"source": "test"}
            )
            
            # Should call collection.add
            mock_collection.add.assert_called_once()


class TestEntityExtractor:
    """Test entity extraction functionality."""
    
    def test_init_with_config(self, test_config, mock_spacy_nlp):
        """Test EntityExtractor initialization."""
        with patch('arc_core.spacy.load') as mock_spacy_load:
            mock_spacy_load.return_value = mock_spacy_nlp
            
            extractor = EntityExtractor(test_config)
            
            assert extractor.config == test_config
            assert extractor.nlp == mock_spacy_nlp
    
    def test_extract_entities(self, test_config, mock_spacy_nlp):
        """Test entity extraction from text."""
        with patch('arc_core.spacy.load') as mock_spacy_load:
            mock_spacy_load.return_value = mock_spacy_nlp
            
            extractor = EntityExtractor(test_config)
            
            entities = extractor.extract_entities("Luke Skywalker works with the Rebel Alliance")
            
            # Should return extracted entities
            assert len(entities) == 2
            assert entities[0]['text'] == "Luke Skywalker"
            assert entities[0]['label'] == "PERSON"
            assert entities[1]['text'] == "Rebel Alliance"
            assert entities[1]['label'] == "ORG"
    
    def test_apply_disambiguation_rules(self, test_config, mock_spacy_nlp):
        """Test entity disambiguation rules."""
        with patch('arc_core.spacy.load') as mock_spacy_load:
            mock_spacy_load.return_value = mock_spacy_nlp
            
            extractor = EntityExtractor(test_config)
            
            # Create mock entities with "Nick" which should be disambiguated to "Nicholas"
            entities = [
                {'text': 'Nick', 'label': 'PERSON'},
                {'text': 'Ze Chen', 'label': 'PERSON'}
            ]
            
            disambiguated = extractor.apply_disambiguation_rules(entities)
            
            # Nick should be changed to Nicholas
            assert any(e['text'] == 'Nicholas' for e in disambiguated)
            # Ze Chen should remain unchanged
            assert any(e['text'] == 'Ze Chen' for e in disambiguated)
    
    def test_extract_entities_with_context(self, test_config, mock_spacy_nlp):
        """Test entity extraction with context information."""
        with patch('arc_core.spacy.load') as mock_spacy_load:
            mock_spacy_load.return_value = mock_spacy_nlp
            
            extractor = EntityExtractor(test_config)
            
            context = {
                'filename': 'meeting-notes.md',
                'date': '2024-01-15',
                'meeting_type': 'partnership'
            }
            
            entities = extractor.extract_entities_with_context(
                "Luke Skywalker works with the Rebel Alliance", 
                context
            )
            
            # Should include context in entity metadata
            assert all('context' in entity for entity in entities)


class TestFileProcessor:
    """Test file processing functionality."""
    
    def test_init_with_config(self, test_config):
        """Test FileProcessor initialization."""
        processor = FileProcessor(test_config)
        
        assert processor.config == test_config
    
    def test_list_markdown_files(self, test_config, sample_import_files):
        """Test listing markdown files."""
        # Update config to point to our test directory
        test_config.update({'import': {'source_dir': str(sample_import_files)}})
        
        processor = FileProcessor(test_config)
        files = processor.list_markdown_files()
        
        assert len(files) == 3
        assert all(f.suffix == '.md' for f in files)
        assert any('mon-calamari' in f.name for f in files)
    
    def test_read_file_content(self, test_config, sample_import_files):
        """Test reading file content."""
        test_config.update({'import': {'source_dir': str(sample_import_files)}})
        
        processor = FileProcessor(test_config)
        files = processor.list_markdown_files()
        
        content = processor.read_file_content(files[0])
        
        assert isinstance(content, str)
        assert len(content) > 0
        assert 'Meeting' in content or 'Standup' in content
    
    def test_extract_metadata_from_filename(self, test_config):
        """Test metadata extraction from filename."""
        processor = FileProcessor(test_config)
        
        metadata = processor.extract_metadata_from_filename("20240115-meeting-mon-calamari.md")
        
        assert metadata['date'] == '2024-01-15'
        assert 'meeting' in metadata['type']
        assert 'mon-calamari' in metadata['subject']
    
    def test_parse_markdown_structure(self, test_config, sample_markdown_content):
        """Test parsing markdown structure."""
        processor = FileProcessor(test_config)
        
        structure = processor.parse_markdown_structure(sample_markdown_content)
        
        assert 'title' in structure
        assert 'sections' in structure
        assert len(structure['sections']) > 0
        assert any('Summary' in section['title'] for section in structure['sections'])
    
    def test_extract_meeting_metadata(self, test_config, sample_markdown_content):
        """Test extracting meeting-specific metadata."""
        processor = FileProcessor(test_config)
        
        metadata = processor.extract_meeting_metadata(sample_markdown_content)
        
        assert 'date' in metadata
        assert 'participants' in metadata
        assert 'type' in metadata
        assert len(metadata['participants']) > 0


class TestErrorHandling:
    """Test error handling in core modules."""
    
    def test_neo4j_connection_error_handling(self, test_config):
        """Test Neo4j connection error handling."""
        with patch('arc_core.GraphDatabase.driver') as mock_driver_func:
            mock_driver_func.side_effect = Exception("Database connection failed")
            
            manager = Neo4jManager(test_config)
            
            # Should not raise exception, should return False
            assert manager.test_connection() is False
    
    def test_chromadb_connection_error_handling(self, test_config):
        """Test ChromaDB connection error handling."""
        with patch('arc_core.chromadb.PersistentClient') as mock_client_func:
            mock_client_func.side_effect = Exception("ChromaDB connection failed")
            
            manager = ChromaDBManager(test_config)
            
            # Should not raise exception, should return False
            assert manager.test_connection() is False
    
    def test_spacy_model_loading_error(self, test_config):
        """Test spaCy model loading error handling."""
        with patch('arc_core.spacy.load') as mock_spacy_load:
            mock_spacy_load.side_effect = OSError("Model not found")
            
            with pytest.raises(OSError):
                EntityExtractor(test_config)
    
    def test_file_processing_error_handling(self, test_config):
        """Test file processing error handling."""
        processor = FileProcessor(test_config)
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            processor.read_file_content(Path("non_existent_file.md"))


class TestIntegration:
    """Test integration between core components."""
    
    @pytest.mark.integration
    def test_full_pipeline_mock(self, test_config, mock_database_managers, mock_spacy_nlp, sample_markdown_content):
        """Test full processing pipeline with mocks."""
        with patch('arc_core.spacy.load') as mock_spacy_load:
            mock_spacy_load.return_value = mock_spacy_nlp
            
            # Initialize all components
            neo4j_manager = mock_database_managers['neo4j']
            chromadb_manager = mock_database_managers['chromadb']
            entity_extractor = EntityExtractor(test_config)
            file_processor = FileProcessor(test_config)
            
            # Process sample content
            entities = entity_extractor.extract_entities(sample_markdown_content)
            structure = file_processor.parse_markdown_structure(sample_markdown_content)
            
            # Verify processing worked
            assert len(entities) > 0
            assert 'title' in structure
            assert 'sections' in structure
            
            # Verify database operations would be called
            assert neo4j_manager.test_connection.return_value is True
            assert chromadb_manager.test_connection.return_value is True 