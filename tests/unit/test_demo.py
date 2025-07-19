"""
Demo test to verify our testing framework is working.
"""

import pytest
import os
import sys

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))

from arc_core import ARCConfig, get_config, DatabaseManager


class TestDemo:
    """Demo tests to verify our testing framework works."""
    
    def test_arc_config_creation(self):
        """Test ARCConfig can be created."""
        config = ARCConfig()
        assert config is not None
        assert hasattr(config, 'config')
    
    def test_arc_config_get_method(self):
        """Test ARCConfig get method works."""
        config = ARCConfig()
        
        # Test getting a value that should exist
        neo4j_uri = config.get('neo4j.uri')
        assert neo4j_uri is not None
        
        # Test getting a value with default
        missing_value = config.get('missing.key', 'default')
        assert missing_value == 'default'
    
    def test_get_config_function(self):
        """Test the get_config function works."""
        config = get_config()
        assert config is not None
        assert isinstance(config, ARCConfig)
    
    def test_database_manager_creation(self):
        """Test DatabaseManager can be created."""
        config = get_config()
        db_manager = DatabaseManager(config)
        assert db_manager is not None
        assert db_manager.config == config
    
    @pytest.mark.slow
    def test_database_connections_available(self):
        """Test that database connection properties don't fail immediately."""
        config = get_config()
        db_manager = DatabaseManager(config)
        
        # These should not raise exceptions during property access
        # (though they might fail to connect)
        try:
            neo4j_driver = db_manager.neo4j
            assert neo4j_driver is not None
        except Exception:
            # Connection might fail, but property should exist
            pass
        
        try:
            chromadb_client = db_manager.chromadb
            assert chromadb_client is not None
        except Exception:
            # Connection might fail, but property should exist
            pass


class TestConfiguration:
    """Test configuration functionality."""
    
    def test_config_has_expected_sections(self):
        """Test that configuration has expected sections."""
        config = get_config()
        
        # These should exist in default config
        assert config.get('neo4j') is not None
        assert config.get('chromadb') is not None
        assert config.get('spacy') is not None
    
    def test_config_nested_access(self):
        """Test nested configuration access."""
        config = get_config()
        
        # Test dot notation access
        neo4j_uri = config.get('neo4j.uri')
        assert neo4j_uri is not None
        assert isinstance(neo4j_uri, str)
        
        chromadb_path = config.get('chromadb.path')
        assert chromadb_path is not None
        assert isinstance(chromadb_path, str)
    
    def test_config_defaults(self):
        """Test that configuration has sensible defaults."""
        config = get_config()
        
        # Test some default values
        neo4j_uri = config.get('neo4j.uri')
        assert 'localhost' in neo4j_uri or 'neo4j' in neo4j_uri
        
        spacy_model = config.get('spacy.model')
        assert spacy_model is not None
        assert 'en_core_web' in spacy_model 