"""
Pytest configuration and shared fixtures for ARC tests.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, MagicMock, patch

import pytest
import json
from datetime import datetime, timedelta

# Add tools directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Provide test configuration."""
    return {
        'chromadb': {
            'path': './test_data/chromadb',
            'document_collection': 'test_documents',
            'summary_collection': 'test_summaries'
        },
        'neo4j': {
            'uri': 'bolt://localhost:7687',
            'auth': None
        },
        'spacy': {
            'model': 'en_core_web_sm'  # Use smaller model for tests
        },
        'sentence_transformers': {
            'model': 'sentence-transformers/all-MiniLM-L6-v2'
        },
        'import': {
            'source_dir': './test_data/import',
            'pattern': '*.md'
        },
        'entity': {
            'disambiguation_rules': {
                'Luke': 'Luke Skywalker',
                'Obi-Wan': 'Obi-Wan Kenobi',  # Never abbreviate Jedi names
                'Vader': 'Darth Vader'
            }
        }
    }


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    mock_driver = Mock()
    mock_session = Mock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_driver.session.return_value.__exit__.return_value = None
    
    # Mock query results
    mock_result = Mock()
    mock_result.single.return_value = {"count": 0}
    mock_result.data.return_value = []
    mock_session.run.return_value = mock_result
    
    return mock_driver


@pytest.fixture
def mock_chromadb_client():
    """Mock ChromaDB client for testing."""
    mock_client = Mock()
    mock_collection = Mock()
    
    # Configure collection behavior
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
    
    return mock_client


@pytest.fixture
def mock_spacy_nlp():
    """Mock spaCy NLP model for testing."""
    mock_nlp = Mock()
    
    # Create mock doc with entities
    mock_doc = Mock()
    mock_ent1 = Mock()
    mock_ent1.text = "Luke Skywalker"
    mock_ent1.label_ = "PERSON"
    mock_ent1.start = 0
    mock_ent1.end = 2
    
    mock_ent2 = Mock()
    mock_ent2.text = "Rebel Alliance"
    mock_ent2.label_ = "ORG"
    mock_ent2.start = 3
    mock_ent2.end = 5
    
    mock_doc.ents = [mock_ent1, mock_ent2]
    mock_doc.text = "Luke Skywalker works with the Rebel Alliance"
    
    mock_nlp.return_value = mock_doc
    
    return mock_nlp


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model for testing."""
    mock_model = Mock()
    
    # Return predictable embeddings
    import numpy as np
    mock_model.encode.return_value = np.random.rand(384)  # Standard embedding size
    
    return mock_model


@pytest.fixture
def sample_markdown_content() -> str:
    """Provide sample markdown content for testing."""
    return """# Meeting with Admiral Ackbar from Mon Calamari Fleet

**Date:** 2024-01-15
**Participants:** Luke Skywalker (Rebel Alliance), Admiral Ackbar (Mon Calamari Fleet)
**Type:** Strategic Alliance Discussion

## Summary

Had a productive meeting with Admiral Ackbar from the Mon Calamari Fleet to discuss potential alliance opportunities. Ackbar is the Fleet Admiral and was very interested in our tactical intelligence systems.

## Key Points

- Mon Calamari's ship designs could help with our fleet modernization
- They have experience with large-scale operations against the Empire
- Potential joint training program starting next quarter

## Action Items

- [ ] Ackbar to send technical specifications for MC80 cruisers
- [ ] Schedule follow-up with engineering corps
- [ ] Review resource allocation strategy

## People Mentioned

- **Admiral Ackbar** - Fleet Admiral of Mon Calamari Fleet
- **Luke Skywalker** - Jedi Knight with Rebel Alliance
- **Princess Leia** - Rebel Leader (to be included in next meeting)

## Status

[PENDING] - Awaiting technical documentation from Mon Calamari Fleet
"""


@pytest.fixture
def sample_entities() -> list:
    """Provide sample entities for testing."""
    return [
        {
            'text': 'Luke Skywalker',
            'label': 'PERSON',
            'type': 'person',
            'properties': {'role': 'Jedi Knight', 'affiliation': 'Rebel Alliance'}
        },
        {
            'text': 'Admiral Ackbar',
            'label': 'PERSON', 
            'type': 'person',
            'properties': {'role': 'Fleet Admiral', 'affiliation': 'Mon Calamari Fleet'}
        },
        {
            'text': 'Rebel Alliance',
            'label': 'ORG',
            'type': 'organization',
            'properties': {'type': 'Military Alliance'}
        },
        {
            'text': 'Mon Calamari Fleet',
            'label': 'ORG',
            'type': 'organization',
            'properties': {'type': 'Naval Force'}
        }
    ]


@pytest.fixture
def sample_import_files(temp_dir: Path) -> Path:
    """Create sample import files for testing."""
    import_dir = temp_dir / "import"
    import_dir.mkdir()
    
    # Create several test markdown files
    files_content = {
        "20240115-meeting-mon-calamari.md": """# Meeting with Mon Calamari Fleet
        
**Date:** 2024-01-15
**Participants:** Luke Skywalker, Admiral Ackbar

Discussion about alliance with Mon Calamari Fleet tactical systems.
""",
        "20240120-weekly-standup.md": """# Weekly Rebel Cell Briefing

**Date:** 2024-01-20

Cell updates and mission planning for the week.

## Attendees
- Luke Skywalker
- Princess Leia
- Han Solo
""",
        "20240201-strategy-review.md": """# Strategic Review Meeting

**Date:** 2024-02-01

Quarterly strategy review with Alliance leadership.

## Key Decisions
- Approved new intelligence network architecture
- Resource allocation for next quarter
"""
    }
    
    for filename, content in files_content.items():
        (import_dir / filename).write_text(content)
    
    return import_dir


@pytest.fixture
def mock_arc_config(test_config):
    """Mock ARC configuration."""
    with patch('arc_core.get_config') as mock_get_config:
        mock_get_config.return_value = test_config
        yield test_config


@pytest.fixture
def sample_neo4j_data():
    """Sample Neo4j graph data for testing."""
    return {
        'nodes': [
            {'id': 'person_1', 'labels': ['Person'], 'properties': {'name': 'Luke Skywalker', 'role': 'Jedi Knight'}},
            {'id': 'person_2', 'labels': ['Person'], 'properties': {'name': 'Admiral Ackbar', 'role': 'Fleet Admiral'}},
            {'id': 'company_1', 'labels': ['Company'], 'properties': {'name': 'Rebel Alliance'}},
            {'id': 'company_2', 'labels': ['Company'], 'properties': {'name': 'Mon Calamari Fleet'}},
            {'id': 'meeting_1', 'labels': ['Meeting'], 'properties': {'date': '2024-01-15', 'title': 'Partnership Discussion'}}
        ],
        'relationships': [
            {'start': 'person_1', 'end': 'company_1', 'type': 'WORKS_AT'},
            {'start': 'person_2', 'end': 'company_2', 'type': 'WORKS_AT'},
            {'start': 'person_1', 'end': 'meeting_1', 'type': 'ATTENDED'},
            {'start': 'person_2', 'end': 'meeting_1', 'type': 'ATTENDED'}
        ]
    }


@pytest.fixture
def mock_database_managers(mock_neo4j_driver, mock_chromadb_client):
    """Mock both database managers."""
    with patch('arc_core.Neo4jManager') as mock_neo4j_class, \
         patch('arc_core.ChromaDBManager') as mock_chroma_class:
        
        # Configure Neo4j manager
        mock_neo4j_instance = Mock()
        mock_neo4j_instance.driver = mock_neo4j_driver
        mock_neo4j_instance.test_connection.return_value = True
        mock_neo4j_instance.setup_constraints.return_value = None
        mock_neo4j_class.return_value = mock_neo4j_instance
        
        # Configure ChromaDB manager
        mock_chroma_instance = Mock()
        mock_chroma_instance.client = mock_chromadb_client
        mock_chroma_instance.test_connection.return_value = True
        mock_chroma_instance.setup_collections.return_value = None
        mock_chroma_class.return_value = mock_chroma_instance
        
        yield {
            'neo4j': mock_neo4j_instance,
            'chromadb': mock_chroma_instance
        }


# Test data factories using factory_boy
try:
    import factory
    from factory import Faker
    
    class PersonFactory(factory.Factory):
        class Meta:
            model = dict
        
        name = Faker('name')
        role = Faker('job')
        company = Faker('company')
        email = Faker('email')
    
    class CompanyFactory(factory.Factory):
        class Meta:
            model = dict
        
        name = Faker('company')
        domain = Faker('domain_name')
        industry = Faker('bs')
    
    class MeetingFactory(factory.Factory):
        class Meta:
            model = dict
        
        title = Faker('sentence', nb_words=4)
        date = Faker('date_between', start_date='-1y', end_date='today')
        participants = factory.List([Faker('name') for _ in range(3)])
        
except ImportError:
    # Fallback if factory_boy not available
    PersonFactory = None
    CompanyFactory = None
    MeetingFactory = None


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Automatically cleanup test data after each test."""
    yield
    
    # Cleanup any test files or directories
    test_dirs = ['./test_data', './htmlcov']
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
            except:
                pass  # Ignore cleanup errors


# Pytest hooks for better test organization
def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    for item in items:
        # Add 'unit' marker to all tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add 'integration' marker to all tests in integration/ directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration) 