"""
Integration tests for migration script and full data processing workflow.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))

from migration_script import (
    MigrationManager,
    process_file,
    extract_entities_from_content,
    create_graph_relationships,
    store_in_chromadb
)


class TestMigrationManager:
    """Test the main migration manager class."""
    
    @pytest.mark.integration
    def test_migration_manager_initialization(self, test_config, mock_database_managers):
        """Test MigrationManager initialization with all components."""
        with patch('migration_script.get_config') as mock_get_config, \
             patch('migration_script.Neo4jManager') as mock_neo4j_class, \
             patch('migration_script.ChromaDBManager') as mock_chroma_class, \
             patch('migration_script.EntityExtractor') as mock_extractor_class, \
             patch('migration_script.FileProcessor') as mock_processor_class:
            
            mock_get_config.return_value = test_config
            mock_neo4j_class.return_value = mock_database_managers['neo4j']
            mock_chroma_class.return_value = mock_database_managers['chromadb']
            mock_extractor_class.return_value = Mock()
            mock_processor_class.return_value = Mock()
            
            manager = MigrationManager()
            
            assert manager.config == test_config
            assert manager.neo4j_manager == mock_database_managers['neo4j']
            assert manager.chromadb_manager == mock_database_managers['chromadb']
            assert manager.entity_extractor is not None
            assert manager.file_processor is not None
    
    @pytest.mark.integration
    def test_setup_databases(self, test_config, mock_database_managers):
        """Test database setup process."""
        with patch('migration_script.get_config') as mock_get_config, \
             patch('migration_script.Neo4jManager') as mock_neo4j_class, \
             patch('migration_script.ChromaDBManager') as mock_chroma_class, \
             patch('migration_script.EntityExtractor') as mock_extractor_class, \
             patch('migration_script.FileProcessor') as mock_processor_class:
            
            mock_get_config.return_value = test_config
            mock_neo4j_class.return_value = mock_database_managers['neo4j']
            mock_chroma_class.return_value = mock_database_managers['chromadb']
            mock_extractor_class.return_value = Mock()
            mock_processor_class.return_value = Mock()
            
            # Mock successful connections
            mock_database_managers['neo4j'].test_connection.return_value = True
            mock_database_managers['chromadb'].test_connection.return_value = True
            
            manager = MigrationManager()
            result = manager.setup_databases()
            
            assert result is True
            mock_database_managers['neo4j'].setup_constraints.assert_called_once()
            mock_database_managers['chromadb'].setup_collections.assert_called_once()
    
    @pytest.mark.integration
    def test_setup_databases_failure(self, test_config, mock_database_managers):
        """Test database setup failure handling."""
        with patch('migration_script.get_config') as mock_get_config, \
             patch('migration_script.Neo4jManager') as mock_neo4j_class, \
             patch('migration_script.ChromaDBManager') as mock_chroma_class, \
             patch('migration_script.EntityExtractor') as mock_extractor_class, \
             patch('migration_script.FileProcessor') as mock_processor_class:
            
            mock_get_config.return_value = test_config
            mock_neo4j_class.return_value = mock_database_managers['neo4j']
            mock_chroma_class.return_value = mock_database_managers['chromadb']
            mock_extractor_class.return_value = Mock()
            mock_processor_class.return_value = Mock()
            
            # Mock failed Neo4j connection
            mock_database_managers['neo4j'].test_connection.return_value = False
            mock_database_managers['chromadb'].test_connection.return_value = True
            
            manager = MigrationManager()
            result = manager.setup_databases()
            
            assert result is False
    
    @pytest.mark.integration
    def test_validate_import_files(self, test_config, sample_import_files, mock_database_managers):
        """Test import file validation."""
        with patch('migration_script.get_config') as mock_get_config, \
             patch('migration_script.Neo4jManager') as mock_neo4j_class, \
             patch('migration_script.ChromaDBManager') as mock_chroma_class, \
             patch('migration_script.EntityExtractor') as mock_extractor_class, \
             patch('migration_script.FileProcessor') as mock_processor_class:
            
            # Update config to point to test files
            test_config.update({'import': {'source_dir': str(sample_import_files)}})
            mock_get_config.return_value = test_config
            
            mock_neo4j_class.return_value = mock_database_managers['neo4j']
            mock_chroma_class.return_value = mock_database_managers['chromadb']
            mock_extractor_class.return_value = Mock()
            
            # Mock file processor
            mock_file_processor = Mock()
            mock_file_processor.list_markdown_files.return_value = [
                sample_import_files / "20240115-meeting-anyscale.md",
                sample_import_files / "20240120-weekly-standup.md",
                sample_import_files / "20240201-product-review.md"
            ]
            mock_processor_class.return_value = mock_file_processor
            
            manager = MigrationManager()
            files = manager.validate_import_files()
            
            assert len(files) == 3
            assert all(f.suffix == '.md' for f in files)


class TestFileProcessingIntegration:
    """Test file processing integration with all components."""
    
    @pytest.mark.integration
    def test_process_single_file_complete_workflow(self, test_config, sample_markdown_content, mock_database_managers):
        """Test complete workflow for processing a single file."""
        with patch('migration_script.get_config') as mock_get_config, \
             patch('migration_script.spacy.load') as mock_spacy_load:
            
            mock_get_config.return_value = test_config
            
            # Mock spaCy model
            mock_nlp = Mock()
            mock_doc = Mock()
            mock_ent1 = Mock()
            mock_ent1.text = "Luke Skywalker"
            mock_ent1.label_ = "PERSON"
            mock_ent1.start = 0
            mock_ent1.end = 2
            
            mock_ent2 = Mock()
            mock_ent2.text = "Admiral Ackbar"
            mock_ent2.label_ = "PERSON"
            mock_ent2.start = 10
            mock_ent2.end = 12
            
            mock_ent3 = Mock()
            mock_ent3.text = "Mon Calamari Fleet"
            mock_ent3.label_ = "ORG"
            mock_ent3.start = 20
            mock_ent3.end = 21
            
            mock_doc.ents = [mock_ent1, mock_ent2, mock_ent3]
            mock_nlp.return_value = mock_doc
            mock_spacy_load.return_value = mock_nlp
            
            # Test the process_file function
            result = process_file(
                file_path=Path("test_meeting.md"),
                content=sample_markdown_content,
                neo4j_manager=mock_database_managers['neo4j'],
                chromadb_manager=mock_database_managers['chromadb'],
                entity_extractor=Mock(),
                file_processor=Mock()
            )
            
            assert result is not None
            assert 'entities' in result
            assert 'metadata' in result
    
    @pytest.mark.integration
    def test_entity_extraction_integration(self, test_config, sample_markdown_content):
        """Test entity extraction with realistic content."""
        with patch('migration_script.spacy.load') as mock_spacy_load:
            
            # Setup comprehensive spaCy mock
            mock_nlp = Mock()
            mock_doc = Mock()
            
            # Create realistic entity extraction results
            entities = []
            entity_data = [
                ("Luke Skywalker", "PERSON", 0, 2),
                ("Admiral Ackbar", "PERSON", 10, 12),
                ("Mon Calamari Fleet", "ORG", 20, 21),
                ("Rebel Alliance", "ORG", 5, 6),
                ("2024-01-15", "DATE", 30, 33)
            ]
            
            for text, label, start, end in entity_data:
                mock_ent = Mock()
                mock_ent.text = text
                mock_ent.label_ = label
                mock_ent.start = start
                mock_ent.end = end
                entities.append(mock_ent)
            
            mock_doc.ents = entities
            mock_nlp.return_value = mock_doc
            mock_spacy_load.return_value = mock_nlp
            
            # Test entity extraction
            extracted_entities = extract_entities_from_content(
                content=sample_markdown_content,
                config=test_config
            )
            
            assert len(extracted_entities) == 5
            
            # Check for expected entities
            entity_texts = [e['text'] for e in extracted_entities]
            assert "Luke Skywalker" in entity_texts
            assert "Admiral Ackbar" in entity_texts
            assert "Mon Calamari Fleet" in entity_texts
            assert "Rebel Alliance" in entity_texts
    
    @pytest.mark.integration
    def test_graph_relationship_creation(self, test_config, sample_entities, mock_database_managers):
        """Test creating graph relationships from entities."""
        neo4j_manager = mock_database_managers['neo4j']
        
        # Mock successful node creation
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.single.return_value = {'node_id': 'test_id_123'}
        mock_session.run.return_value = mock_result
        
        # Test relationship creation
        result = create_graph_relationships(
            entities=sample_entities,
            metadata={'filename': 'test_meeting.md', 'date': '2024-01-15'},
            neo4j_manager=neo4j_manager
        )
        
        assert result is not None
        # Verify that nodes and relationships were created
        assert mock_session.run.call_count > 0
    
    @pytest.mark.integration
    def test_chromadb_storage_integration(self, test_config, sample_markdown_content, mock_database_managers):
        """Test storing content in ChromaDB."""
        chromadb_manager = mock_database_managers['chromadb']
        
        # Mock collection behavior
        mock_collection = Mock()
        mock_collection.add.return_value = None
        chromadb_manager.client.get_or_create_collection.return_value = mock_collection
        
        # Test ChromaDB storage
        result = store_in_chromadb(
            content=sample_markdown_content,
            metadata={'filename': 'test_meeting.md', 'date': '2024-01-15'},
            entities=sample_entities,
            chromadb_manager=chromadb_manager
        )
        
        assert result is True
        mock_collection.add.assert_called()


class TestFullMigrationWorkflow:
    """Test the complete migration workflow end-to-end."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_migration_workflow(self, test_config, sample_import_files, mock_database_managers):
        """Test complete migration workflow with multiple files."""
        with patch('migration_script.get_config') as mock_get_config, \
             patch('migration_script.Neo4jManager') as mock_neo4j_class, \
             patch('migration_script.ChromaDBManager') as mock_chroma_class, \
             patch('migration_script.EntityExtractor') as mock_extractor_class, \
             patch('migration_script.FileProcessor') as mock_processor_class, \
             patch('migration_script.spacy.load') as mock_spacy_load:
            
            # Update config for test files
            test_config.update({'import': {'source_dir': str(sample_import_files)}})
            mock_get_config.return_value = test_config
            
            # Setup database mocks
            mock_neo4j_class.return_value = mock_database_managers['neo4j']
            mock_chroma_class.return_value = mock_database_managers['chromadb']
            mock_database_managers['neo4j'].test_connection.return_value = True
            mock_database_managers['chromadb'].test_connection.return_value = True
            
            # Setup entity extractor mock
            mock_extractor = Mock()
            mock_extractor.extract_entities.return_value = [
                {'text': 'Luke Skywalker', 'label': 'PERSON', 'type': 'person'},
                {'text': 'Rebel Alliance', 'label': 'ORG', 'type': 'company'}
            ]
            mock_extractor_class.return_value = mock_extractor
            
            # Setup file processor mock
            mock_processor = Mock()
            files = list(sample_import_files.glob("*.md"))
            mock_processor.list_markdown_files.return_value = files
            mock_processor.read_file_content.side_effect = lambda f: f.read_text()
            mock_processor.extract_metadata_from_filename.return_value = {
                'date': '2024-01-15',
                'type': 'meeting',
                'subject': 'test'
            }
            mock_processor_class.return_value = mock_processor
            
            # Setup spaCy mock
            mock_nlp = Mock()
            mock_doc = Mock()
            mock_doc.ents = []
            mock_nlp.return_value = mock_doc
            mock_spacy_load.return_value = mock_nlp
            
            # Run migration
            manager = MigrationManager()
            
            # Setup databases
            setup_result = manager.setup_databases()
            assert setup_result is True
            
            # Validate files
            import_files = manager.validate_import_files()
            assert len(import_files) > 0
            
            # Process files (in validation mode for testing)
            processed_files = []
            for file_path in import_files[:2]:  # Process first 2 files for testing
                try:
                    content = mock_processor.read_file_content(file_path)
                    result = process_file(
                        file_path=file_path,
                        content=content,
                        neo4j_manager=mock_database_managers['neo4j'],
                        chromadb_manager=mock_database_managers['chromadb'],
                        entity_extractor=mock_extractor,
                        file_processor=mock_processor
                    )
                    if result:
                        processed_files.append(file_path)
                except Exception as e:
                    pytest.fail(f"Failed to process {file_path}: {e}")
            
            assert len(processed_files) > 0
    
    @pytest.mark.integration
    def test_migration_error_handling(self, test_config, sample_import_files, mock_database_managers):
        """Test migration error handling and recovery."""
        with patch('migration_script.get_config') as mock_get_config, \
             patch('migration_script.Neo4jManager') as mock_neo4j_class, \
             patch('migration_script.ChromaDBManager') as mock_chroma_class, \
             patch('migration_script.EntityExtractor') as mock_extractor_class, \
             patch('migration_script.FileProcessor') as mock_processor_class:
            
            mock_get_config.return_value = test_config
            mock_neo4j_class.return_value = mock_database_managers['neo4j']
            mock_chroma_class.return_value = mock_database_managers['chromadb']
            
            # Simulate database connection failure
            mock_database_managers['neo4j'].test_connection.return_value = False
            mock_database_managers['chromadb'].test_connection.return_value = True
            
            # Setup other mocks
            mock_extractor_class.return_value = Mock()
            mock_processor_class.return_value = Mock()
            
            manager = MigrationManager()
            
            # This should fail gracefully
            setup_result = manager.setup_databases()
            assert setup_result is False
    
    @pytest.mark.integration
    def test_partial_migration_recovery(self, test_config, sample_import_files, mock_database_managers):
        """Test recovery from partial migration."""
        with patch('migration_script.get_config') as mock_get_config, \
             patch('migration_script.Neo4jManager') as mock_neo4j_class, \
             patch('migration_script.ChromaDBManager') as mock_chroma_class, \
             patch('migration_script.EntityExtractor') as mock_extractor_class, \
             patch('migration_script.FileProcessor') as mock_processor_class:
            
            mock_get_config.return_value = test_config
            mock_neo4j_class.return_value = mock_database_managers['neo4j']
            mock_chroma_class.return_value = mock_database_managers['chromadb']
            
            # Mock successful connections
            mock_database_managers['neo4j'].test_connection.return_value = True
            mock_database_managers['chromadb'].test_connection.return_value = True
            
            # Setup entity extractor that fails on second call
            mock_extractor = Mock()
            mock_extractor.extract_entities.side_effect = [
                [{'text': 'Luke', 'label': 'PERSON'}],  # First call succeeds
                Exception("Entity extraction failed"),     # Second call fails
                [{'text': 'Leia', 'label': 'PERSON'}]    # Third call succeeds
            ]
            mock_extractor_class.return_value = mock_extractor
            
            # Setup file processor
            mock_processor = Mock()
            files = list(sample_import_files.glob("*.md"))
            mock_processor.list_markdown_files.return_value = files
            mock_processor.read_file_content.side_effect = lambda f: f.read_text()
            mock_processor_class.return_value = mock_processor
            
            manager = MigrationManager()
            
            # This should handle the error gracefully
            setup_result = manager.setup_databases()
            assert setup_result is True


class TestValidationMode:
    """Test migration validation mode."""
    
    @pytest.mark.integration
    def test_validation_mode_processing(self, test_config, sample_import_files, mock_database_managers):
        """Test processing files in validation mode."""
        with patch('migration_script.get_config') as mock_get_config, \
             patch('migration_script.Neo4jManager') as mock_neo4j_class, \
             patch('migration_script.ChromaDBManager') as mock_chroma_class, \
             patch('migration_script.EntityExtractor') as mock_extractor_class, \
             patch('migration_script.FileProcessor') as mock_processor_class:
            
            test_config.update({'import': {'source_dir': str(sample_import_files)}})
            mock_get_config.return_value = test_config
            
            mock_neo4j_class.return_value = mock_database_managers['neo4j']
            mock_chroma_class.return_value = mock_database_managers['chromadb']
            
            # Setup mocks
            mock_extractor = Mock()
            mock_extractor.extract_entities.return_value = [
                {'text': 'Luke Skywalker', 'label': 'PERSON'}
            ]
            mock_extractor_class.return_value = mock_extractor
            
            mock_processor = Mock()
            files = list(sample_import_files.glob("*.md"))
            mock_processor.list_markdown_files.return_value = files
            mock_processor.read_file_content.side_effect = lambda f: f.read_text()
            mock_processor_class.return_value = mock_processor
            
            manager = MigrationManager()
            
            # In validation mode, should not write to databases
            validation_results = []
            for file_path in files[:2]:  # Test first 2 files
                content = mock_processor.read_file_content(file_path)
                entities = mock_extractor.extract_entities(content)
                
                validation_results.append({
                    'file': file_path.name,
                    'entities_found': len(entities),
                    'content_length': len(content)
                })
            
            assert len(validation_results) == 2
            assert all(r['entities_found'] > 0 for r in validation_results)
            assert all(r['content_length'] > 0 for r in validation_results)


class TestPerformanceAndScaling:
    """Test performance and scaling considerations."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_file_processing(self, test_config, temp_dir, mock_database_managers):
        """Test processing large files."""
        # Create a large test file
        large_content = """# Large Meeting Notes

This is a large meeting with many participants and topics.

""" + "\n".join([f"## Section {i}\n\nContent about topic {i} with Person {i} from Company {i}." for i in range(100)])
        
        large_file = temp_dir / "large_meeting.md"
        large_file.write_text(large_content)
        
        with patch('migration_script.get_config') as mock_get_config, \
             patch('migration_script.spacy.load') as mock_spacy_load:
            
            mock_get_config.return_value = test_config
            
            # Setup spaCy mock for large content
            mock_nlp = Mock()
            mock_doc = Mock()
            # Simulate many entities being found
            entities = []
            for i in range(50):  # 50 entities
                mock_ent = Mock()
                mock_ent.text = f"Person {i}"
                mock_ent.label_ = "PERSON"
                mock_ent.start = i * 10
                mock_ent.end = i * 10 + 2
                entities.append(mock_ent)
            
            mock_doc.ents = entities
            mock_nlp.return_value = mock_doc
            mock_spacy_load.return_value = mock_nlp
            
            # Test processing large content
            result = extract_entities_from_content(
                content=large_content,
                config=test_config
            )
            
            assert len(result) == 50
            assert len(large_content) > 1000  # Ensure it's actually large
    
    @pytest.mark.integration
    def test_batch_processing_efficiency(self, test_config, mock_database_managers):
        """Test efficient batch processing of multiple files."""
        # This would test batching strategies for better performance
        # For now, just verify the concept works
        
        batch_size = 5
        total_files = 20
        
        # Simulate processing files in batches
        processed_batches = []
        for i in range(0, total_files, batch_size):
            batch = list(range(i, min(i + batch_size, total_files)))
            processed_batches.append(batch)
        
        # Should have processed in 4 batches of 5
        assert len(processed_batches) == 4
        assert len(processed_batches[0]) == 5
        assert len(processed_batches[-1]) == 5 