"""
Test utilities and helper functions for ARC system testing.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
import json
import subprocess

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))


class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_sample_meeting_file(filename: str, temp_dir: Path) -> Path:
        """Create a sample meeting markdown file."""
        content = f"""# Meeting Notes - {filename.replace('.md', '').replace('_', ' ').title()}

**Date:** 2024-01-15
**Participants:** Luke Skywalker (Rebel Alliance), Admiral Ackbar (Mon Calamari Fleet)
**Type:** Strategic Alliance Discussion

## Summary

Productive meeting to discuss potential alliance opportunities between the Rebel Alliance and Mon Calamari Fleet.

## Key Points

- Discussed fleet tactical capabilities
- Reviewed integration possibilities for joint operations
- Identified next steps for coordinated resistance

## People Mentioned

- **Luke Skywalker** - Jedi Knight with Rebel Alliance
- **Admiral Ackbar** - Fleet Admiral of Mon Calamari Fleet
- **Princess Leia** - Alliance Leader (to be included in follow-up)

## Action Items

- [ ] Ackbar to send fleet technical specifications
- [ ] Schedule follow-up strategic meeting
- [ ] Review tactical requirements

## Status

[PENDING] - Awaiting fleet documentation
"""
        
        file_path = temp_dir / filename
        file_path.write_text(content)
        return file_path
    
    @staticmethod
    def create_sample_standup_file(filename: str, temp_dir: Path) -> Path:
        """Create a sample standup markdown file."""
        content = f"""# Weekly Rebel Cell Briefing

**Date:** 2024-01-20
**Participants:** Cell Alpha members

## Updates

### Luke Skywalker
- Worked on Force training protocols
- Meeting with Mon Calamari fleet commanders
- Strategic planning for upcoming missions

### Princess Leia
- Intelligence gathering operations
- Diplomatic outreach to neutral systems
- Coordination with other rebel cells

### Han Solo
- Smuggling route optimization
- Falcon maintenance and upgrades
- Reconnaissance mission planning

## Blockers

- Waiting for new equipment delivery
- Need approval for expanded operations

## Next Week Goals

- Complete mission preparations
- Establish new supply lines
- Prepare for quarterly strategy review
"""
        
        file_path = temp_dir / filename
        file_path.write_text(content)
        return file_path
    
    @staticmethod
    def create_test_entities() -> List[Dict[str, Any]]:
        """Create sample entities for testing."""
        return [
            {
                'text': 'Luke Skywalker',
                'label': 'PERSON',
                'type': 'person',
                'properties': {
                    'role': 'Senior Jedi Knight',
                    'company': 'Rebel Alliance',
                    'email': 'david.curlewis@canva.com'
                }
            },
            {
                'text': 'Admiral Ackbar',
                'label': 'PERSON',
                'type': 'person',
                'properties': {
                    'role': 'Fleet Admiral',
                    'company': 'Mon Calamari Fleet'
                }
            },
            {
                'text': 'Princess Leia',
                'label': 'PERSON',
                'type': 'person',
                'properties': {
                    'role': 'Rebel Technicianing Lead',
                    'company': 'Rebel Alliance'
                }
            },
            {
                'text': 'Rebel Alliance',
                'label': 'ORG',
                'type': 'company',
                'properties': {
                    'industry': 'Galactic Communications',
                    'size': 'Large'
                }
            },
            {
                'text': 'Mon Calamari Fleet',
                'label': 'ORG',
                'type': 'company',
                'properties': {
                    'industry': 'Fleet Tactical Systems',
                    'size': 'Medium'
                }
            }
        ]
    
    @staticmethod
    def create_test_relationships() -> List[Dict[str, Any]]:
        """Create sample relationships for testing."""
        return [
            {
                'source': 'Luke Skywalker',
                'target': 'Rebel Alliance',
                'type': 'WORKS_AT',
                'properties': {'since': '2022-01-01'}
            },
            {
                'source': 'Admiral Ackbar',
                'target': 'Mon Calamari Fleet',
                'type': 'WORKS_AT',
                'properties': {'since': '2021-06-01'}
            },
            {
                'source': 'Princess Leia',
                'target': 'Rebel Alliance',
                'type': 'WORKS_AT',
                'properties': {'since': '2023-03-01'}
            },
            {
                'source': 'Luke Skywalker',
                'target': 'Admiral Ackbar',
                'type': 'MET_WITH',
                'properties': {
                    'date': '2024-01-15',
                    'context': 'Partnership Discussion'
                }
            }
        ]


class DatabaseTestHelper:
    """Helper for database-related testing."""
    
    @staticmethod
    def create_mock_neo4j_driver():
        """Create a comprehensive Neo4j driver mock."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        
        # Default query results
        mock_result = Mock()
        mock_result.single.return_value = {"count": 0}
        mock_result.data.return_value = []
        mock_session.run.return_value = mock_result
        
        return mock_driver, mock_session
    
    @staticmethod
    def create_mock_chromadb_client():
        """Create a comprehensive ChromaDB client mock."""
        mock_client = Mock()
        mock_collection = Mock()
        
        # Default collection behavior
        mock_collection.count.return_value = 0
        mock_collection.query.return_value = {
            'ids': [],
            'distances': [],
            'documents': [],
            'metadatas': []
        }
        mock_collection.add.return_value = None
        mock_collection.get.return_value = {
            'ids': [],
            'documents': [],
            'metadatas': []
        }
        
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.delete_collection = Mock()
        
        return mock_client, mock_collection
    
    @staticmethod
    def setup_neo4j_query_response(mock_session: Mock, query_type: str, response_data: List[Dict]):
        """Setup specific Neo4j query responses."""
        mock_result = Mock()
        
        if query_type == 'single':
            mock_result.single.return_value = response_data[0] if response_data else None
        else:
            mock_result.data.return_value = response_data
        
        mock_session.run.return_value = mock_result
    
    @staticmethod
    def setup_chromadb_query_response(mock_collection: Mock, documents: List[str], 
                                    metadatas: List[Dict], distances: Optional[List[float]] = None):
        """Setup specific ChromaDB query responses."""
        if distances is None:
            distances = [0.1 * i for i in range(len(documents))]
        
        mock_collection.query.return_value = {
            'ids': [f'doc_{i}' for i in range(len(documents))],
            'distances': distances,
            'documents': documents,
            'metadatas': metadatas
        }


class FileTestHelper:
    """Helper for file-related testing."""
    
    @staticmethod
    def create_test_import_directory(temp_dir: Path, num_files: int = 5) -> Path:
        """Create a test import directory with sample files."""
        import_dir = temp_dir / "import"
        import_dir.mkdir()
        
        for i in range(num_files):
            if i % 3 == 0:
                filename = f"2024011{i+5}-meeting-test{i}.md"
                TestDataFactory.create_sample_meeting_file(filename, import_dir)
            elif i % 3 == 1:
                filename = f"2024012{i}-standup-weekly.md"
                TestDataFactory.create_sample_standup_file(filename, import_dir)
            else:
                filename = f"2024020{i}-review-strategy.md"
                content = f"""# Strategic Review {i}

**Date:** 2024-02-0{i}

Alliance strategic review notes.

## Attendees
- Command Leader
- Strategic Coordinator  
- Operations Team

## Decisions
- Approved mission parameters
- Timeline confirmed
"""
                (import_dir / filename).write_text(content)
        
        return import_dir
    
    @staticmethod
    def create_config_file(temp_dir: Path, config_data: Optional[Dict] = None) -> Path:
        """Create a test configuration file."""
        if config_data is None:
            config_data = {
                'chromadb': {
                    'path': str(temp_dir / 'chromadb'),
                    'document_collection': 'test_documents',
                    'summary_collection': 'test_summaries'
                },
                'neo4j': {
                    'uri': 'bolt://localhost:7687',
                    'auth': None
                },
                'spacy': {
                    'model': 'en_core_web_sm'
                },
                'import': {
                    'source_dir': str(temp_dir / 'import'),
                    'pattern': '*.md'
                }
            }
        
        config_file = temp_dir / "config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        return config_file


class TestRunner:
    """Helper for running tests programmatically."""
    
    @staticmethod
    def run_unit_tests(verbose: bool = True) -> bool:
        """Run unit tests."""
        cmd = ["python", "-m", "pytest", "tests/unit/"]
        if verbose:
            cmd.append("-v")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return result.returncode == 0
        except Exception as e:
            print(f"Error running unit tests: {e}")
            return False
    
    @staticmethod
    def run_integration_tests(verbose: bool = True) -> bool:
        """Run integration tests."""
        cmd = ["python", "-m", "pytest", "tests/integration/", "-m", "integration"]
        if verbose:
            cmd.append("-v")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return result.returncode == 0
        except Exception as e:
            print(f"Error running integration tests: {e}")
            return False
    
    @staticmethod
    def run_coverage_report() -> bool:
        """Run tests with coverage report."""
        cmd = ["python", "-m", "pytest", "--cov=tools", "--cov-report=html", "--cov-report=term"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return result.returncode == 0
        except Exception as e:
            print(f"Error running coverage tests: {e}")
            return False


class MockHelper:
    """Helper for creating comprehensive mocks."""
    
    @staticmethod
    def create_spacy_mock(entities: Optional[List[Dict]] = None):
        """Create a comprehensive spaCy NLP mock."""
        if entities is None:
            entities = [
                {'text': 'Luke Skywalker', 'label': 'PERSON', 'start': 0, 'end': 2},
                {'text': 'Rebel Alliance', 'label': 'ORG', 'start': 5, 'end': 7}
            ]
        
        mock_nlp = Mock()
        mock_doc = Mock()
        
        mock_entities = []
        for ent_data in entities:
            mock_ent = Mock()
            mock_ent.text = ent_data['text']
            mock_ent.label_ = ent_data['label']
            mock_ent.start = ent_data.get('start', 0)
            mock_ent.end = ent_data.get('end', 1)
            mock_entities.append(mock_ent)
        
        mock_doc.ents = mock_entities
        mock_nlp.return_value = mock_doc
        
        return mock_nlp
    
    @staticmethod
    def create_sentence_transformer_mock(embedding_size: int = 384):
        """Create a sentence transformer mock."""
        mock_model = Mock()
        
        import numpy as np
        # Return consistent embeddings for testing
        mock_model.encode.return_value = np.random.rand(embedding_size)
        
        return mock_model


class ValidationHelper:
    """Helper for validation and assertions."""
    
    @staticmethod
    def validate_entity_structure(entity: Dict[str, Any]) -> bool:
        """Validate that an entity has the required structure."""
        required_fields = ['text', 'label', 'type']
        return all(field in entity for field in required_fields)
    
    @staticmethod
    def validate_relationship_structure(relationship: Dict[str, Any]) -> bool:
        """Validate that a relationship has the required structure."""
        required_fields = ['source', 'target', 'type']
        return all(field in relationship for field in required_fields)
    
    @staticmethod
    def validate_neo4j_query(query: str) -> bool:
        """Basic validation of Neo4j query syntax."""
        query_upper = query.upper()
        # Check for basic Cypher keywords
        valid_keywords = ['MATCH', 'CREATE', 'MERGE', 'WHERE', 'RETURN', 'WITH']
        return any(keyword in query_upper for keyword in valid_keywords)
    
    @staticmethod
    def validate_chromadb_metadata(metadata: Dict[str, Any]) -> bool:
        """Validate ChromaDB metadata structure."""
        # Basic validation - metadata should be JSON serializable
        try:
            json.dumps(metadata)
            return True
        except (TypeError, ValueError):
            return False


class PerformanceHelper:
    """Helper for performance testing."""
    
    @staticmethod
    def measure_function_time(func, *args, **kwargs):
        """Measure execution time of a function."""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    
    @staticmethod
    def create_large_test_data(num_entities: int = 1000) -> List[Dict[str, Any]]:
        """Create large test datasets for performance testing."""
        entities = []
        for i in range(num_entities):
            entities.append({
                'text': f'Person {i}',
                'label': 'PERSON',
                'type': 'person',
                'properties': {
                    'role': f'Role {i % 10}',
                    'company': f'Company {i % 50}'
                }
            })
        return entities


# Pytest fixtures that can be imported in test files
def pytest_fixtures():
    """Return commonly used pytest fixtures as functions."""
    
    def create_temp_test_environment():
        """Create a temporary test environment."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create directory structure
        (temp_dir / "chromadb").mkdir()
        (temp_dir / "neo4j").mkdir()
        import_dir = FileTestHelper.create_test_import_directory(temp_dir)
        config_file = FileTestHelper.create_config_file(temp_dir)
        
        return {
            'temp_dir': temp_dir,
            'import_dir': import_dir,
            'config_file': config_file,
            'cleanup': lambda: shutil.rmtree(temp_dir)
        }
    
    return {
        'create_temp_test_environment': create_temp_test_environment
    }


# Quick test commands for development
def quick_test():
    """Run quick smoke tests."""
    print("Running quick smoke tests...")
    
    # Test imports
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))
        from arc_core import get_config
        from arc_query import query_person
        print("✓ Imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test configuration
    try:
        config = get_config()
        print("✓ Configuration loading successful")
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False
    
    print("✓ Quick tests passed!")
    return True


if __name__ == "__main__":
    # Run quick tests if called directly
    quick_test() 