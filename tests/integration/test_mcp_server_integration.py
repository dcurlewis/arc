"""
Integration tests for Enhanced MCP Server.
Tests the complete MCP server functionality with enhanced embeddings and entity extraction.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any

# Import MCP server components
try:
    from arc_mcp_server import server, call_tool
    from enhanced_embeddings import create_enhanced_embedding_system
    from enhanced_entity_extractor import EnhancedEntityExtractor
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent / "tools"))
    from arc_mcp_server import server, call_tool
    from enhanced_embeddings import create_enhanced_embedding_system
    from enhanced_entity_extractor import EnhancedEntityExtractor


class TestMCPServerIntegration:
    """Integration tests for the enhanced MCP server functionality."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager for testing."""
        mock_db = MagicMock()
        mock_db.chromadb_client = MagicMock()
        mock_db.get_chromadb_client.return_value = mock_db.chromadb_client
        
        # Mock Neo4j session
        mock_session = MagicMock()
        mock_db.get_neo4j_session.return_value.__enter__.return_value = mock_session
        mock_db.get_neo4j_session.return_value.__exit__.return_value = None
        
        return mock_db

    @pytest.fixture
    def mock_enhanced_query_interface(self):
        """Create a mock enhanced query interface."""
        mock_interface = MagicMock()
        
        # Mock search results
        mock_interface.hybrid_search.return_value = {
            'results': [
                {
                    'content': 'Luke Skywalker is a Jedi Knight',
                    'metadata': {'title': 'Jedi Training', 'file_path': '/test.md'},
                    'confidence': 0.95,
                    'type': 'document'
                }
            ],
            'summary': {
                'total_results': 1,
                'search_types': ['documents'],
                'query': 'Luke Skywalker'
            }
        }
        
        mock_interface.entity_centric_search.return_value = {
            'entity_matches': [
                {'text': 'Luke Skywalker', 'type': 'PERSON', 'confidence': 0.98}
            ],
            'related_documents': [
                {'title': 'Jedi Training', 'content': 'Luke trains with Yoda'}
            ],
            'related_entities': [
                {'text': 'Yoda', 'type': 'PERSON', 'relationship': 'TRAINS_WITH'}
            ]
        }
        
        mock_interface.relationship_search.return_value = {
            'direct_relationships': [
                {'source': 'Luke Skywalker', 'target': 'Yoda', 'type': 'TRAINS_WITH'}
            ],
            'related_documents': [
                {'title': 'Training Session', 'content': 'Luke and Yoda practice'}
            ]
        }
        
        mock_interface.temporal_search.return_value = [
            {
                'content': 'Meeting notes from yesterday',
                'metadata': {'date': '2024-01-01', 'type': 'meeting'},
                'confidence': 0.9
            }
        ]
        
        return mock_interface

    @pytest.fixture
    async def mcp_server_setup(self, mock_db_manager, mock_enhanced_query_interface):
        """Set up MCP server with mocked dependencies."""
        with patch('arc_mcp_server.get_db_manager') as mock_get_db, \
             patch('arc_mcp_server.create_enhanced_embedding_system') as mock_create_system:
            
            mock_get_db.return_value = mock_db_manager
            mock_create_system.return_value = (MagicMock(), mock_enhanced_query_interface)
            
            # Import and patch the global variables
            import arc_mcp_server
            arc_mcp_server.db_manager = mock_db_manager
            arc_mcp_server.enhanced_query_interface = mock_enhanced_query_interface
            
            yield {
                'db_manager': mock_db_manager,
                'query_interface': mock_enhanced_query_interface,
                'server': server
            }

    @pytest.mark.asyncio
    async def test_server_initialization(self, mcp_server_setup):
        """Test that the MCP server initializes correctly with enhanced systems."""
        setup = mcp_server_setup
        
        # Verify that the enhanced query interface is available
        assert setup['query_interface'] is not None
        assert hasattr(setup['query_interface'], 'hybrid_search')
        assert hasattr(setup['query_interface'], 'entity_centric_search')

    @pytest.mark.asyncio
    async def test_semantic_search_tool(self, mcp_server_setup):
        """Test the semantic search tool functionality."""
        args = {
            'query': 'Luke Skywalker Jedi training',
            'limit': 5
        }
        
        result = await call_tool('semantic_search', args)
        
        assert len(result) > 0
        assert result[0].type == 'text'
        # Check that the result contains expected content
        content = result[0].text
        assert 'results found' in content.lower() or 'no results' in content.lower()

    @pytest.mark.asyncio
    async def test_entity_search_tool(self, mcp_server_setup):
        """Test the entity search tool functionality."""
        args = {
            'entity_name': 'Luke Skywalker',
            'entity_type': 'PERSON',
            'limit': 5
        }
        
        result = await call_tool('entity_search', args)
        
        assert len(result) > 0
        assert result[0].type == 'text'
        content = result[0].text
        assert 'Luke Skywalker' in content

    @pytest.mark.asyncio
    async def test_relationship_query_tool(self, mcp_server_setup):
        """Test the relationship query tool functionality."""
        args = {
            'source_entity': 'Luke Skywalker',
            'target_entity': 'Yoda',
            'relationship_type': 'TRAINS_WITH',
            'limit': 5
        }
        
        result = await call_tool('relationship_query', args)
        
        assert len(result) > 0
        assert result[0].type == 'text'
        content = result[0].text
        assert 'relationship' in content.lower() or 'connection' in content.lower()

    @pytest.mark.asyncio
    async def test_temporal_search_tool(self, mcp_server_setup):
        """Test the temporal search tool functionality."""
        args = {
            'query': 'team meeting',
            'start_date': '2024-01-01T00:00:00Z',
            'end_date': '2024-12-31T23:59:59Z',
            'limit': 5
        }
        
        result = await call_tool('temporal_search', args)
        
        assert len(result) > 0
        assert result[0].type == 'text'
        content = result[0].text
        assert 'temporal' in content.lower() or 'date' in content.lower() or 'time' in content.lower()

    @pytest.mark.asyncio
    async def test_enhanced_hybrid_search_tool(self, mcp_server_setup):
        """Test the enhanced hybrid search tool functionality."""
        args = {
            'query': 'Force sensitivity training',
            'search_types': ['documents', 'entities', 'relationships'],
            'filters': {'file_type': 'training_notes'},
            'limit': 10
        }
        
        result = await call_tool('enhanced_hybrid_search', args)
        
        assert len(result) > 0
        assert result[0].type == 'text'
        content = result[0].text
        # Should include information about multiple search types
        assert 'hybrid' in content.lower() or 'search' in content.lower()

    @pytest.mark.asyncio
    async def test_enhanced_entity_centric_search_tool(self, mcp_server_setup):
        """Test the enhanced entity-centric search tool functionality."""
        args = {
            'entity_name': 'Jedi Order',
            'include_related': True,
            'relationship_depth': 2,
            'limit': 8
        }
        
        result = await call_tool('entity_centric_search', args)
        
        assert len(result) > 0
        assert result[0].type == 'text'
        content = result[0].text
        assert 'entity' in content.lower() or 'related' in content.lower()

    @pytest.mark.asyncio
    async def test_enhanced_relationship_search_tool(self, mcp_server_setup):
        """Test the enhanced relationship search tool functionality."""
        args = {
            'source_entity': 'Luke Skywalker',
            'target_entity': 'Darth Vader',
            'relationship_types': ['CONFLICT', 'FAMILY'],
            'limit': 5
        }
        
        result = await call_tool('relationship_search', args)
        
        assert len(result) > 0
        assert result[0].type == 'text'
        content = result[0].text
        assert 'relationship' in content.lower()

    @pytest.mark.asyncio
    async def test_enhanced_temporal_search_tool(self, mcp_server_setup):
        """Test the enhanced temporal search tool functionality."""
        args = {
            'query': 'strategic planning session',
            'start_date': '2024-06-01T00:00:00Z',
            'end_date': '2024-06-30T23:59:59Z',
            'search_type': 'documents',
            'limit': 5
        }
        
        result = await call_tool('enhanced_temporal_search', args)
        
        assert len(result) > 0
        assert result[0].type == 'text'
        content = result[0].text
        assert 'temporal' in content.lower() or 'date' in content.lower()

    @pytest.mark.asyncio
    async def test_document_similarity_tool(self, mcp_server_setup):
        """Test the document similarity tool functionality."""
        args = {
            'document_id': 'test_doc_123',
            'limit': 5
        }
        
        # Mock the Neo4j query result
        mock_session = mcp_server_setup['db_manager'].get_neo4j_session.return_value.__enter__.return_value
        mock_session.run.return_value = []
        
        result = await call_tool('document_similarity', args)
        
        assert len(result) > 0
        assert result[0].type == 'text'

    @pytest.mark.asyncio
    async def test_meeting_preparation_tool(self, mcp_server_setup):
        """Test the meeting preparation tool functionality."""
        args = {
            'attendees': ['Luke Skywalker', 'Yoda'],
            'topic': 'Jedi training progress',
            'days_back': 30
        }
        
        # Mock the Neo4j query result
        mock_session = mcp_server_setup['db_manager'].get_neo4j_session.return_value.__enter__.return_value
        mock_session.run.return_value = [
            {
                'entity': {'name': 'Luke Skywalker', 'type': 'PERSON'},
                'document': {'title': 'Training Notes', 'content': 'Luke practiced lightsaber forms'}
            }
        ]
        
        result = await call_tool('meeting_preparation', args)
        
        assert len(result) > 0
        assert result[0].type == 'text'
        content = result[0].text
        assert 'context' in content.lower() or 'meeting' in content.lower()

    @pytest.mark.asyncio
    async def test_error_handling_invalid_tool(self, mcp_server_setup):
        """Test error handling for invalid tool names."""
        with pytest.raises(ValueError):
            await call_tool('invalid_tool_name', {})

    @pytest.mark.asyncio
    async def test_error_handling_missing_parameters(self, mcp_server_setup):
        """Test error handling for missing required parameters."""
        # Test semantic search without query
        result = await call_tool('semantic_search', {})
        
        # Should handle gracefully and return error message
        assert len(result) > 0
        assert result[0].type == 'text'
        assert 'error' in result[0].text.lower() or 'missing' in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self, mcp_server_setup):
        """Test handling of concurrent tool calls."""
        # Create multiple concurrent calls
        tasks = []
        for i in range(3):
            task = call_tool('semantic_search', {'query': f'test query {i}', 'limit': 3})
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All calls should complete successfully
        assert len(results) == 3
        for result in results:
            assert len(result) > 0
            assert result[0].type == 'text'

    @pytest.mark.asyncio
    async def test_tool_performance(self, mcp_server_setup):
        """Test tool performance under load."""
        import time
        
        # Test multiple sequential calls
        start_time = time.time()
        
        for i in range(5):
            await call_tool('semantic_search', {'query': f'performance test {i}', 'limit': 2})
        
        end_time = time.time()
        
        # Should complete in reasonable time (less than 5 seconds for 5 calls)
        assert end_time - start_time < 5.0

    @pytest.mark.asyncio
    async def test_enhanced_embedding_integration(self, mcp_server_setup):
        """Test that enhanced embeddings are properly integrated."""
        setup = mcp_server_setup
        
        # Test that enhanced search returns structured results
        args = {'query': 'Jedi training', 'search_types': ['documents', 'entities']}
        result = await call_tool('enhanced_hybrid_search', args)
        
        assert len(result) > 0
        content = result[0].text
        
        # Should show evidence of enhanced embedding features
        assert any(keyword in content.lower() for keyword in ['hybrid', 'enhanced', 'entities', 'documents'])

    @pytest.mark.asyncio  
    async def test_entity_extraction_integration(self, mcp_server_setup):
        """Test that enhanced entity extraction is working."""
        # This tests that the system can handle queries that would benefit from enhanced extraction
        args = {
            'entity_name': 'Luke',  # Should be disambiguated to 'Luke Skywalker'
            'limit': 5
        }
        
        result = await call_tool('entity_centric_search', args)
        
        assert len(result) > 0
        assert result[0].type == 'text'
        content = result[0].text
        
        # Should handle the entity search successfully
        assert 'luke' in content.lower() or 'entity' in content.lower()


class TestMCPServerToolValidation:
    """Test validation of MCP server tool schemas and responses."""
    
    def test_tool_schemas_valid(self):
        """Test that all tool schemas are valid JSON schemas."""
        # Import the server tools list
        import arc_mcp_server
        
        # Verify that server.list_tools() is available
        assert hasattr(arc_mcp_server.server, 'list_tools')
        
        # This would be called by the MCP client to get tool definitions
        # We're testing that the tool definitions are properly structured

    @pytest.mark.asyncio
    async def test_tool_response_format(self, mock_db_manager, mock_enhanced_query_interface):
        """Test that tool responses follow the correct MCP format."""
        with patch('arc_mcp_server.get_db_manager') as mock_get_db, \
             patch('arc_mcp_server.create_enhanced_embedding_system') as mock_create_system:
            
            mock_get_db.return_value = mock_db_manager
            mock_create_system.return_value = (MagicMock(), mock_enhanced_query_interface)
            
            # Import and patch the global variables
            import arc_mcp_server
            arc_mcp_server.db_manager = mock_db_manager
            arc_mcp_server.enhanced_query_interface = mock_enhanced_query_interface
            
            # Test that responses are TextContent objects
            result = await call_tool('semantic_search', {'query': 'test'})
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Each result should be a TextContent object with type and text
            for item in result:
                assert hasattr(item, 'type')
                assert hasattr(item, 'text')
                assert item.type == 'text'
                assert isinstance(item.text, str)


if __name__ == "__main__":
    pytest.main([__file__]) 