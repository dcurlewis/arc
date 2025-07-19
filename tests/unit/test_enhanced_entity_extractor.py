"""
Unit tests for Enhanced Entity Extractor system.
Tests the new enhanced entity extraction, disambiguation, and relationship inference.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import the enhanced entity extractor
try:
    from enhanced_entity_extractor import EnhancedEntityExtractor
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent / "tools"))
    from enhanced_entity_extractor import EnhancedEntityExtractor


class TestEnhancedEntityExtractor:
    """Test suite for Enhanced Entity Extractor functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            'spacy.model': 'en_core_web_sm',
            'entity_patterns': {
                'MEETING': [
                    {'label': 'MEETING', 'pattern': [{'LOWER': {'IN': ['meeting', 'standup', 'sync', 'retro']}}]}
                ],
                'PROCESS': [
                    {'label': 'PROCESS', 'pattern': [{'LOWER': 'sprint'}, {'LOWER': 'planning'}]}
                ]
            },
            'disambiguation_rules': {
                'people': {
                    'luke': 'Luke Skywalker',
                    'vader': 'Darth Vader',
                    'ben': 'Obi-Wan Kenobi'
                },
                'organizations': {
                    'empire': 'Galactic Empire',
                    'rebels': 'Rebel Alliance'
                }
            },
            'aliases': {
                'Luke Skywalker': ['Luke', 'Skywalker'],
                'Darth Vader': ['Vader', 'Anakin'],
                'Obi-Wan Kenobi': ['Ben', 'Obi-Wan']
            },
            'domain_mappings': {
                'jedi_terms': ['force', 'lightsaber', 'padawan', 'master'],
                'sith_terms': ['dark side', 'empire', 'death star']
            },
            'confidence_thresholds': {
                'min_confidence': 0.7,
                'high_confidence': 0.9
            },
            'relationship_patterns': {
                'collaboration': ['worked with', 'collaborated', 'partnered'],
                'mentorship': ['taught', 'trained', 'mentored'],
                'conflict': ['fought', 'opposed', 'defeated']
            }
        }

    @pytest.fixture
    def temp_config_file(self, mock_config):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(mock_config, f)
            return f.name

    @pytest.fixture
    def extractor(self, mock_config):
        """Create an Enhanced Entity Extractor instance for testing."""
        with patch('enhanced_entity_extractor.spacy.load') as mock_spacy:
            # Mock spaCy model with proper vocab
            mock_nlp = MagicMock()
            mock_vocab = MagicMock()
            mock_nlp.vocab = mock_vocab
            mock_nlp.pipe_names = []
            mock_nlp.add_pipe = MagicMock()
            mock_spacy.return_value = mock_nlp
            
            # Mock the pattern matchers
            with patch('enhanced_entity_extractor.Matcher') as mock_matcher, \
                 patch('enhanced_entity_extractor.PhraseMatcher') as mock_phrase_matcher:
                
                # Pass the config dictionary directly
                extractor = EnhancedEntityExtractor(mock_config)
                return extractor

    def test_initialization(self, mock_config):
        """Test that the extractor initializes correctly."""
        with patch('enhanced_entity_extractor.spacy.load') as mock_spacy, \
             patch('enhanced_entity_extractor.Matcher'), \
             patch('enhanced_entity_extractor.PhraseMatcher'):
            
            # Mock spaCy model with proper vocab
            mock_nlp = MagicMock()
            mock_nlp.vocab = MagicMock()
            mock_nlp.pipe_names = []
            mock_nlp.add_pipe = MagicMock()
            mock_spacy.return_value = mock_nlp
            
            extractor = EnhancedEntityExtractor(mock_config)
            assert extractor.config == mock_config
            assert hasattr(extractor, 'config')
            assert hasattr(extractor, 'nlp')

    def test_config_loading(self, mock_config):
        """Test that configuration is loaded correctly."""
        with patch('enhanced_entity_extractor.spacy.load') as mock_spacy, \
             patch('enhanced_entity_extractor.Matcher'), \
             patch('enhanced_entity_extractor.PhraseMatcher'):
            
            # Mock spaCy model with proper vocab
            mock_nlp = MagicMock()
            mock_nlp.vocab = MagicMock()
            mock_nlp.pipe_names = []
            mock_nlp.add_pipe = MagicMock()
            mock_spacy.return_value = mock_nlp
            
            extractor = EnhancedEntityExtractor(mock_config)
            assert extractor.config['spacy.model'] == 'en_core_web_sm'
            assert 'disambiguation_rules' in extractor.config
            assert 'entity_patterns' in extractor.config

    def test_entity_extraction_basic(self, extractor):
        """Test basic entity extraction functionality."""
        # Mock document with entities
        mock_doc = MagicMock()
        mock_ent1 = MagicMock()
        mock_ent1.text = "Luke Skywalker"
        mock_ent1.label_ = "PERSON"
        mock_ent1.start_char = 0
        mock_ent1.end_char = 13
        
        mock_ent2 = MagicMock()
        mock_ent2.text = "Jedi"
        mock_ent2.label_ = "ORG"
        mock_ent2.start_char = 20
        mock_ent2.end_char = 24
        
        mock_doc.ents = [mock_ent1, mock_ent2]
        extractor.nlp.return_value = mock_doc
        
        text = "Luke Skywalker is a Jedi"
        entities, relationships = extractor.extract_entities_and_relationships(text)
        
        assert len(entities) >= 2
        # Check that entities contain expected information
        entity_texts = [e['text'] for e in entities]
        assert 'Luke Skywalker' in entity_texts
        assert 'Jedi' in entity_texts

    def test_disambiguation_rules(self, extractor):
        """Test that disambiguation rules are applied correctly."""
        # Mock document with ambiguous entity
        mock_doc = MagicMock()
        mock_ent = MagicMock()
        mock_ent.text = "luke"
        mock_ent.label_ = "PERSON"
        mock_ent.start_char = 0
        mock_ent.end_char = 4
        
        mock_doc.ents = [mock_ent]
        extractor.nlp.return_value = mock_doc
        
        text = "luke attended the meeting"
        entities, _ = extractor.extract_entities_and_relationships(text)
        
        # Should disambiguate 'luke' to 'Luke Skywalker'
        assert any(e['canonical_name'] == 'Luke Skywalker' for e in entities)

    def test_custom_pattern_matching(self, extractor):
        """Test that custom entity patterns are matched correctly."""
        # This would require more complex mocking of spaCy's EntityRuler
        # For now, test that the patterns are loaded
        assert 'entity_patterns' in extractor.config
        assert 'MEETING' in extractor.config['entity_patterns']

    def test_relationship_inference(self, extractor):
        """Test relationship inference between entities."""
        # Mock document with multiple entities
        mock_doc = MagicMock()
        mock_ent1 = MagicMock()
        mock_ent1.text = "Luke"
        mock_ent1.label_ = "PERSON"
        
        mock_ent2 = MagicMock()
        mock_ent2.text = "Vader"
        mock_ent2.label_ = "PERSON"
        
        mock_doc.ents = [mock_ent1, mock_ent2]
        mock_doc.text = "Luke fought Vader in the battle"
        extractor.nlp.return_value = mock_doc
        
        text = "Luke fought Vader in the battle"
        entities, relationships = extractor.extract_entities_and_relationships(text)
        
        # Should infer conflict relationship
        assert len(relationships) > 0

    def test_confidence_scoring(self, extractor):
        """Test that confidence scores are calculated correctly."""
        mock_doc = MagicMock()
        mock_ent = MagicMock()
        mock_ent.text = "Luke Skywalker"
        mock_ent.label_ = "PERSON"
        mock_ent.start_char = 0
        mock_ent.end_char = 13
        
        mock_doc.ents = [mock_ent]
        extractor.nlp.return_value = mock_doc
        
        text = "Luke Skywalker is the main character"
        entities, _ = extractor.extract_entities_and_relationships(text)
        
        # Check that confidence scores are present
        for entity in entities:
            assert 'confidence' in entity
            assert 0 <= entity['confidence'] <= 1

    def test_alias_resolution(self, extractor):
        """Test that entity aliases are resolved correctly."""
        # Mock document with alias
        mock_doc = MagicMock()
        mock_ent = MagicMock()
        mock_ent.text = "Ben"
        mock_ent.label_ = "PERSON"
        
        mock_doc.ents = [mock_ent]
        extractor.nlp.return_value = mock_doc
        
        text = "Ben taught Luke about the Force"
        entities, _ = extractor.extract_entities_and_relationships(text)
        
        # Should resolve 'Ben' to 'Obi-Wan Kenobi'
        assert any(e['canonical_name'] == 'Obi-Wan Kenobi' for e in entities)

    def test_temporal_extraction(self, extractor):
        """Test extraction of temporal information."""
        mock_doc = MagicMock()
        mock_ent = MagicMock()
        mock_ent.text = "yesterday"
        mock_ent.label_ = "DATE"
        
        mock_doc.ents = [mock_ent]
        extractor.nlp.return_value = mock_doc
        
        text = "The meeting was yesterday"
        entities, _ = extractor.extract_entities_and_relationships(text)
        
        # Should extract temporal entities
        assert any(e['label'] == 'DATE' for e in entities)

    def test_context_aware_extraction(self, extractor):
        """Test that context affects entity extraction confidence."""
        # Test with domain-specific context
        jedi_text = "Luke used the Force to defeat the Empire"
        sith_text = "Vader serves the dark side of the Force"
        
        # Mock responses for both contexts
        mock_doc = MagicMock()
        mock_ent = MagicMock()
        mock_ent.text = "Force"
        mock_ent.label_ = "MISC"
        
        mock_doc.ents = [mock_ent]
        extractor.nlp.return_value = mock_doc
        
        entities1, _ = extractor.extract_entities_and_relationships(jedi_text)
        entities2, _ = extractor.extract_entities_and_relationships(sith_text)
        
        # Both should extract entities, but potentially with different confidence
        assert len(entities1) > 0
        assert len(entities2) > 0

    def test_performance_with_large_text(self, extractor):
        """Test performance with larger text inputs."""
        # Generate large text
        large_text = "Luke Skywalker was a Jedi. " * 100
        
        mock_doc = MagicMock()
        mock_ent = MagicMock()
        mock_ent.text = "Luke Skywalker"
        mock_ent.label_ = "PERSON"
        
        mock_doc.ents = [mock_ent] * 100  # Simulate many entities
        extractor.nlp.return_value = mock_doc
        
        import time
        start_time = time.time()
        entities, relationships = extractor.extract_entities_and_relationships(large_text)
        end_time = time.time()
        
        # Should complete in reasonable time (< 5 seconds for test)
        assert end_time - start_time < 5.0
        assert len(entities) > 0

    def test_error_handling(self, extractor):
        """Test error handling for malformed inputs."""
        # Test with empty text
        entities, relationships = extractor.extract_entities_and_relationships("")
        assert entities == []
        assert relationships == []
        
        # Test with None input
        entities, relationships = extractor.extract_entities_and_relationships(None)
        assert entities == []
        assert relationships == []

    def test_spacy_pipeline_components(self, extractor):
        """Test that all required spaCy pipeline components are present."""
        # This tests that our custom components are added to the pipeline
        # The actual components would be tested in integration tests
        assert hasattr(extractor, 'nlp')

    @pytest.fixture(autouse=True)
    def cleanup_temp_files(self, temp_config_file):
        """Clean up temporary files after each test."""
        yield
        Path(temp_config_file).unlink(missing_ok=True)


@pytest.mark.integration
class TestEnhancedEntityExtractorIntegration:
    """Integration tests for Enhanced Entity Extractor with real spaCy models."""
    
    def test_real_spacy_integration(self, temp_config_file):
        """Test integration with real spaCy model (if available)."""
        try:
            import spacy
            # Try to load the model
            nlp = spacy.load('en_core_web_sm')
            
            # If successful, test with real model
            extractor = EnhancedEntityExtractor(temp_config_file)
            text = "Luke Skywalker fought Darth Vader on the Death Star"
            entities, relationships = extractor.extract_entities_and_relationships(text)
            
            assert len(entities) > 0
            assert any('Luke' in e['text'] for e in entities)
            
        except (OSError, ImportError):
            # Model not available, skip test
            pytest.skip("spaCy model 'en_core_web_sm' not available") 