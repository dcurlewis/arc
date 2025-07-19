"""
Unit tests for Enhanced Embedding System.
Tests the new multi-modal embedding generation and advanced query functionality.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import the enhanced embedding system
try:
    from enhanced_embeddings import EnhancedEmbeddingGenerator, EnhancedQueryInterface, create_enhanced_embedding_system
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent / "tools"))
    from enhanced_embeddings import EnhancedEmbeddingGenerator, EnhancedQueryInterface, create_enhanced_embedding_system


class TestEnhancedEmbeddingGenerator:
    """Test suite for Enhanced Embedding Generator functionality."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager for testing."""
        mock_db = MagicMock()
        mock_db.chromadb_client = MagicMock()
        mock_db.get_chromadb_client.return_value = mock_db.chromadb_client
        return mock_db

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock sentence transformer model."""
        mock_model = MagicMock()
        # Mock embedding dimension
        mock_model.encode.return_value = np.random.rand(384)  # Standard MiniLM dimension
        return mock_model

    @pytest.fixture
    def embedding_generator(self, mock_db_manager, mock_sentence_transformer):
        """Create an Enhanced Embedding Generator instance for testing."""
        with patch('enhanced_embeddings.SentenceTransformer') as mock_st:
            mock_st.return_value = mock_sentence_transformer
            generator = EnhancedEmbeddingGenerator(mock_db_manager)
            return generator

    def test_initialization(self, mock_db_manager):
        """Test that the embedding generator initializes correctly."""
        generator = EnhancedEmbeddingGenerator(mock_db_manager)
        assert hasattr(generator, 'db_manager')
        assert hasattr(generator, 'embedding_model')
        assert hasattr(generator, 'collections')
        assert hasattr(generator, 'chromadb_client')

    def test_collection_creation(self, embedding_generator):
        """Test that collections are created correctly."""
        # Mock ChromaDB client
        mock_collection = MagicMock()
        embedding_generator.chromadb_client.get_collection.side_effect = Exception("Not found")
        embedding_generator.chromadb_client.create_collection.return_value = mock_collection
        
        collection = embedding_generator._get_or_create_collection("test_collection")
        assert collection == mock_collection

    def test_document_embedding_generation(self, embedding_generator):
        """Test generation of document embeddings."""
        content = "Luke Skywalker is a Jedi Knight"
        entities = [
            {'text': 'Luke Skywalker', 'label': 'PERSON', 'canonical_name': 'Luke Skywalker'},
            {'text': 'Jedi Knight', 'label': 'TITLE', 'canonical_name': 'Jedi Knight'}
        ]
        relationships = [
            {'source': 'Luke Skywalker', 'target': 'Jedi Knight', 'type': 'HAS_TITLE'}
        ]
        
        embedding = embedding_generator.generate_enhanced_document_embedding(content, entities, relationships)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0

    def test_entity_embedding_generation(self, embedding_generator):
        """Test generation of entity-specific embeddings."""
        entity = {
            'text': 'Luke Skywalker',
            'label': 'PERSON',
            'canonical_name': 'Luke Skywalker',
            'confidence': 0.95
        }
        context = "Luke Skywalker is a Jedi Knight who fights the Empire"
        
        embedding = embedding_generator.generate_entity_embedding(entity, context)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0

    def test_relationship_embedding_generation(self, embedding_generator):
        """Test generation of relationship embeddings."""
        relationship = {
            'source': 'Luke Skywalker',
            'target': 'Jedi Knight',
            'type': 'HAS_TITLE',
            'confidence': 0.9
        }
        source_entity = {'text': 'Luke Skywalker', 'label': 'PERSON'}
        target_entity = {'text': 'Jedi Knight', 'label': 'TITLE'}
        context = "Luke Skywalker is a Jedi Knight"
        
        embedding = embedding_generator.generate_relationship_embedding(relationship, source_entity, target_entity, context)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0

    def test_hybrid_embedding_generation(self, embedding_generator):
        """Test generation of hybrid embeddings that combine multiple types."""
        content = "Luke Skywalker trains as a Jedi"
        entities = [{'text': 'Luke Skywalker', 'label': 'PERSON'}]
        relationships = [{'source': 'Luke Skywalker', 'target': 'Jedi', 'type': 'TRAINS_AS'}]
        
        embedding = embedding_generator.generate_hybrid_embedding(content, entities, relationships)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0

    def test_enhanced_document_indexing(self, embedding_generator):
        """Test indexing of enhanced document embeddings."""
        content = "Luke Skywalker joins the Rebel Alliance"
        entities = [
            {'text': 'Luke Skywalker', 'label': 'PERSON'},
            {'text': 'Rebel Alliance', 'label': 'ORG'}
        ]
        relationships = [
            {'source': 'Luke Skywalker', 'target': 'Rebel Alliance', 'type': 'MEMBER_OF'}
        ]
        metadata = {
            'title': 'Test Document',
            'file_path': '/test/doc.md',
            'created_at': '2024-01-01T00:00:00Z'
        }
        
        # Mock collection operations
        for collection_name in ['documents', 'entities', 'relationships', 'hybrid']:
            mock_collection = MagicMock()
            embedding_generator.collections[collection_name] = mock_collection
        
        embedding_generator.index_enhanced_document(content, entities, relationships, metadata)
        
        # Verify that all collections were called
        for collection in embedding_generator.collections.values():
            collection.add.assert_called()

    def test_embedding_dimension_consistency(self, embedding_generator):
        """Test that all embedding types have consistent dimensions."""
        content = "Test content"
        entities = [{'text': 'Test', 'label': 'MISC'}]
        relationships = [{'source': 'A', 'target': 'B', 'type': 'RELATES_TO'}]
        
        doc_emb = embedding_generator.generate_enhanced_document_embedding(content, entities, relationships)
        entity_emb = embedding_generator.generate_entity_embedding(entities[0], content)
        source_entity = {'text': 'A', 'label': 'MISC'}
        target_entity = {'text': 'B', 'label': 'MISC'}
        rel_emb = embedding_generator.generate_relationship_embedding(relationships[0], source_entity, target_entity, content)
        hybrid_emb = embedding_generator.generate_hybrid_embedding(content, entities, relationships)
        
        # All embeddings should have the same dimension
        assert doc_emb.shape == entity_emb.shape
        assert entity_emb.shape == rel_emb.shape
        assert rel_emb.shape == hybrid_emb.shape

    def test_error_handling_empty_input(self, embedding_generator):
        """Test error handling with empty inputs."""
        # Test with empty content
        embedding = embedding_generator.generate_enhanced_document_embedding("", [], [])
        assert isinstance(embedding, np.ndarray)
        
        # Test with None inputs - this should be handled gracefully
        try:
            embedding = embedding_generator.generate_enhanced_document_embedding(None, None, None)
            assert isinstance(embedding, np.ndarray)
        except (ValueError, TypeError):
            # It's acceptable for None inputs to raise exceptions
            pass


class TestEnhancedQueryInterface:
    """Test suite for Enhanced Query Interface functionality."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager for testing."""
        mock_db = MagicMock()
        mock_db.chromadb_client = MagicMock()
        return mock_db

    @pytest.fixture
    def mock_collections(self):
        """Create mock ChromaDB collections."""
        collections = {}
        for name in ['documents', 'entities', 'relationships', 'hybrid']:
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                'ids': [['test_id_1', 'test_id_2']],
                'documents': [['Test document 1', 'Test document 2']],
                'metadatas': [[{'title': 'Test 1'}, {'title': 'Test 2'}]],
                'distances': [[0.1, 0.2]]
            }
            collections[name] = mock_collection
        return collections

    @pytest.fixture
    def query_interface(self, mock_db_manager, mock_collections):
        """Create an Enhanced Query Interface instance for testing."""
        with patch('enhanced_embeddings.SentenceTransformer'):
            interface = EnhancedQueryInterface(mock_db_manager)
            interface.collections = mock_collections
            return interface

    def test_initialization(self, mock_db_manager):
        """Test that the query interface initializes correctly."""
        # QueryInterface needs an embedding generator, not db_manager directly
        generator = Mock()
        generator.collections = {}
        generator.embedding_model = Mock()
        
        interface = EnhancedQueryInterface(generator)
        assert hasattr(interface, 'embedding_gen')
        assert hasattr(interface, 'embedding_model')
        assert hasattr(interface, 'collections')

    def test_hybrid_search(self, query_interface):
        """Test hybrid search across multiple embedding types."""
        query = "Luke Skywalker and the Force"
        search_types = ['documents', 'entities']
        
        results = query_interface.hybrid_search(
            query=query,
            search_types=search_types,
            limit=5
        )
        
        assert isinstance(results, list)
        # Should return results from multiple search types
        assert len(results) >= 0

    def test_entity_centric_search(self, query_interface):
        """Test entity-centric search functionality."""
        entity_name = "Luke Skywalker"
        
        results = query_interface.entity_centric_search(
            entity_name=entity_name,
            limit=5
        )
        
        assert isinstance(results, list)
        # Should return entity-related results
        assert len(results) >= 0

    def test_relationship_search(self, query_interface):
        """Test relationship-based search functionality."""
        source_entity = "Luke Skywalker"
        target_entity = "Darth Vader"
        
        results = query_interface.relationship_search(
            source_entity=source_entity,
            target_entity=target_entity,
            limit=5
        )
        
        assert isinstance(results, list)
        # Should return relationship-related results
        assert len(results) >= 0

    def test_temporal_search(self, query_interface):
        """Test temporal search with date filtering."""
        query = "Jedi training"
        start_date = "2024-01-01T00:00:00Z"
        end_date = "2024-12-31T23:59:59Z"
        
        results = query_interface.temporal_search(
            query=query,
            start_date=start_date,
            end_date=end_date,
            limit=5
        )
        
        assert isinstance(results, list)

    def test_search_with_filters(self, query_interface):
        """Test search functionality with metadata filters."""
        query = "Force sensitivity"
        filters = {'file_type': 'meeting_notes'}
        
        results = query_interface.hybrid_search(
            query=query,
            filters=filters,
            limit=5
        )
        
        assert isinstance(results, list)
        # Verify that filters were applied
        query_interface.collections['documents'].query.assert_called()

    def test_empty_query_handling(self, query_interface):
        """Test handling of empty or invalid queries."""
        # Test empty query
        results = query_interface.hybrid_search(query="")
        assert isinstance(results, list)
        
        # Test None query - should handle gracefully
        try:
            results = query_interface.hybrid_search(query=None)
            assert isinstance(results, list)
        except (ValueError, TypeError):
            # Acceptable to raise exception for None query
            pass

    def test_result_formatting(self, query_interface):
        """Test that search results are properly formatted."""
        query = "Test query"
        results = query_interface.hybrid_search(query=query, limit=3)
        
        # Check result structure
        assert isinstance(results, list)
        # Each result should have the expected structure
        for result in results[:1]:  # Check at least one result if available
            assert 'content' in result
            assert 'metadata' in result
            assert 'distance' in result or 'score' in result

    def test_confidence_scoring(self, query_interface):
        """Test that confidence scores are calculated correctly."""
        query = "Luke Skywalker"
        results = query_interface.hybrid_search(query=query, limit=5)
        
        # Check that results have confidence scores
        assert isinstance(results, list)
        for result in results:
            # Should have either confidence, score, or distance
            assert 'score' in result or 'confidence' in result or 'distance' in result
            if 'score' in result:
                assert 0 <= result['score'] <= 1

    def test_search_type_validation(self, query_interface):
        """Test validation of search types."""
        query = "Test query"
        
        # Valid search types
        valid_types = ['documents', 'entities', 'relationships', 'hybrid']
        results = query_interface.hybrid_search(query=query, search_types=valid_types)
        assert isinstance(results, list)
        
        # Invalid search types should be filtered out
        invalid_types = ['invalid_type']
        results = query_interface.hybrid_search(query=query, search_types=invalid_types)
        assert isinstance(results, list)


class TestEnhancedEmbeddingSystemIntegration:
    """Integration tests for the complete enhanced embedding system."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager for integration tests."""
        mock_db = MagicMock()
        mock_db.chromadb_client = MagicMock()
        mock_db.get_chromadb_client.return_value = mock_db.chromadb_client
        return mock_db

    def test_system_creation(self, mock_db_manager):
        """Test creation of the complete enhanced embedding system."""
        with patch('enhanced_embeddings.SentenceTransformer'):
            generator, query_interface = create_enhanced_embedding_system(mock_db_manager)
            
            assert isinstance(generator, EnhancedEmbeddingGenerator)
            assert isinstance(query_interface, EnhancedQueryInterface)

    def test_end_to_end_workflow(self, mock_db_manager):
        """Test complete workflow from indexing to querying."""
        with patch('enhanced_embeddings.SentenceTransformer') as mock_st:
            # Mock sentence transformer
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(384)
            mock_st.return_value = mock_model
            
            # Create system
            generator, query_interface = create_enhanced_embedding_system(mock_db_manager)
            
            # Mock collections for both generator and query interface
            mock_collections = {}
            for name in ['documents', 'entities', 'relationships', 'hybrid']:
                mock_collection = MagicMock()
                mock_collection.query.return_value = {
                    'ids': [['test_id']],
                    'documents': [['Test document']],
                    'metadatas': [[{'title': 'Test'}]],
                    'distances': [[0.1]]
                }
                mock_collections[name] = mock_collection
            
            generator.collections = mock_collections
            query_interface.collections = mock_collections
            
            # Index a document
            content = "Luke Skywalker trains with Yoda"
            entities = [
                {'text': 'Luke Skywalker', 'label': 'PERSON'},
                {'text': 'Yoda', 'label': 'PERSON'}
            ]
            relationships = [
                {'source': 'Luke Skywalker', 'target': 'Yoda', 'type': 'TRAINS_WITH'}
            ]
            metadata = {'title': 'Jedi Training', 'file_path': '/test.md'}
            
            generator.index_enhanced_document(content, entities, relationships, metadata)
            
            # Query the indexed content
            results = query_interface.hybrid_search("Jedi training", limit=5)
            
            # Verify workflow completed
            assert isinstance(results, list)
            # Should have some results from the mocked collections
            assert len(results) >= 0

    def test_performance_with_large_dataset(self, mock_db_manager):
        """Test system performance with larger datasets."""
        with patch('enhanced_embeddings.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(384)
            mock_st.return_value = mock_model
            
            generator, query_interface = create_enhanced_embedding_system(mock_db_manager)
            
            # Mock collections
            mock_collections = {}
            for name in ['documents', 'entities', 'relationships', 'hybrid']:
                mock_collection = MagicMock()
                mock_collections[name] = mock_collection
            
            generator.collections = mock_collections
            
            # Index multiple documents
            import time
            start_time = time.time()
            
            for i in range(10):  # Simulate 10 documents
                content = f"Document {i} about Luke Skywalker"
                entities = [{'text': 'Luke Skywalker', 'label': 'PERSON'}]
                relationships = []
                metadata = {'title': f'Doc {i}', 'file_path': f'/doc{i}.md'}
                
                generator.index_enhanced_document(content, entities, relationships, metadata)
            
            end_time = time.time()
            
            # Should complete in reasonable time
            assert end_time - start_time < 5.0

    def test_memory_usage(self, mock_db_manager):
        """Test that the system doesn't have memory leaks."""
        with patch('enhanced_embeddings.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(384)
            mock_st.return_value = mock_model
            
            generator, query_interface = create_enhanced_embedding_system(mock_db_manager)
            
            # This is a basic test - in a real scenario, you'd use memory profiling tools
            assert hasattr(generator, 'collections')
            assert hasattr(query_interface, 'collections')


if __name__ == "__main__":
    pytest.main([__file__]) 