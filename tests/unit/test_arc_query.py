"""
Unit tests for arc_query module.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))

from arc_query import (
    find_people,
    find_companies,
    find_relationships,
    find_meetings,
    search_documents,
    get_person_details,
    get_company_details,
    get_meeting_details,
    semantic_search,
    find_interactions_between_people,
    get_recent_meetings,
    find_people_by_company,
    search_by_topic
)


class TestPeopleQueries:
    """Test people-related query functions."""
    
    def test_find_people_basic(self, mock_database_managers):
        """Test basic people search."""
        # Setup mock data
        neo4j_manager = mock_database_managers['neo4j']
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.data.return_value = [
            {'p': {'name': 'Luke Skywalker', 'role': 'Jedi Knight'}},
            {'p': {'name': 'Admiral Ackbar', 'role': 'Fleet Admiral'}}
        ]
        mock_session.run.return_value = mock_result
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {}
            mock_neo4j_class.return_value = neo4j_manager
            
            people = find_people("Luke")
            
            assert len(people) == 2
            assert people[0]['name'] == 'Luke Skywalker'
            assert people[1]['name'] == 'Admiral Ackbar'
    
    def test_find_people_empty_result(self, mock_database_managers):
        """Test people search with no results."""
        neo4j_manager = mock_database_managers['neo4j']
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.data.return_value = []
        mock_session.run.return_value = mock_result
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {}
            mock_neo4j_class.return_value = neo4j_manager
            
            people = find_people("NonExistentPerson")
            
            assert len(people) == 0
    
    def test_get_person_details(self, mock_database_managers):
        """Test getting detailed person information."""
        neo4j_manager = mock_database_managers['neo4j']
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.single.return_value = {
            'person': {
                'name': 'Luke Skywalker',
                'role': 'Jedi Knight',
                'affiliation': 'Rebel Alliance'
            },
            'relationships': [
                {'type': 'MEMBER_OF', 'target': 'Rebel Alliance'},
                {'type': 'ATTENDED', 'target': 'Strategic Alliance Meeting'}
            ]
        }
        mock_session.run.return_value = mock_result
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {}
            mock_neo4j_class.return_value = neo4j_manager
            
            details = get_person_details("Luke Skywalker")
            
            assert details['person']['name'] == 'Luke Skywalker'
            assert details['person']['role'] == 'Jedi Knight'
            assert len(details['relationships']) == 2
    
    def test_find_people_by_company(self, mock_database_managers):
        """Test finding people by company."""
        neo4j_manager = mock_database_managers['neo4j']
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.data.return_value = [
            {'p': {'name': 'Luke Skywalker', 'role': 'Jedi Knight'}},
            {'p': {'name': 'Princess Leia', 'role': 'Rebel Technician'}}
        ]
        mock_session.run.return_value = mock_result
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {}
            mock_neo4j_class.return_value = neo4j_manager
            
            people = find_people_by_company("Rebel Alliance")
            
            assert len(people) == 2
            assert all('name' in person for person in people)


class TestCompanyQueries:
    """Test company-related query functions."""
    
    def test_find_companies(self, mock_database_managers):
        """Test basic company search."""
        neo4j_manager = mock_database_managers['neo4j']
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.data.return_value = [
            {'c': {'name': 'Rebel Alliance', 'industry': 'Galactic Communications'}},
            {'c': {'name': 'Mon Calamari Fleet', 'industry': 'Fleet Tactical Systems'}}
        ]
        mock_session.run.return_value = mock_result
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {}
            mock_neo4j_class.return_value = neo4j_manager
            
            companies = find_companies("Rebel Alliance")
            
            assert len(companies) == 2
            assert companies[0]['name'] == 'Rebel Alliance'
            assert companies[1]['name'] == 'Mon Calamari Fleet'
    
    def test_get_company_details(self, mock_database_managers):
        """Test getting detailed company information."""
        neo4j_manager = mock_database_managers['neo4j']
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.single.return_value = {
            'company': {
                'name': 'Rebel Alliance',
                'industry': 'Galactic Communications',
                'size': 'Large'
            },
            'employees': [
                {'name': 'Luke Skywalker', 'role': 'Jedi Knight'},
                {'name': 'Princess Leia', 'role': 'Rebel Technician'}
            ],
            'partnerships': [
                {'partner': 'Mon Calamari Fleet', 'type': 'Strategic Alliance'}
            ]
        }
        mock_session.run.return_value = mock_result
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {}
            mock_neo4j_class.return_value = neo4j_manager
            
            details = get_company_details("Rebel Alliance")
            
            assert details['company']['name'] == 'Rebel Alliance'
            assert len(details['employees']) == 2
            assert len(details['partnerships']) == 1


class TestMeetingQueries:
    """Test meeting-related query functions."""
    
    def test_find_meetings(self, mock_database_managers):
        """Test basic meeting search."""
        neo4j_manager = mock_database_managers['neo4j']
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.data.return_value = [
            {
                'm': {
                    'title': 'Partnership Discussion',
                    'date': '2024-01-15',
                    'participants': ['Luke Skywalker', 'Admiral Ackbar']
                }
            },
            {
                'm': {
                    'title': 'Weekly Standup',
                    'date': '2024-01-20',
                    'participants': ['Luke Skywalker', 'Princess Leia']
                }
            }
        ]
        mock_session.run.return_value = mock_result
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {}
            mock_neo4j_class.return_value = neo4j_manager
            
            meetings = find_meetings("partnership")
            
            assert len(meetings) == 2
            assert meetings[0]['title'] == 'Partnership Discussion'
    
    def test_get_recent_meetings(self, mock_database_managers):
        """Test getting recent meetings."""
        neo4j_manager = mock_database_managers['neo4j']
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.data.return_value = [
            {
                'm': {
                    'title': 'Recent Meeting 1',
                    'date': '2024-01-20',
                    'participants': ['Luke Skywalker']
                }
            },
            {
                'm': {
                    'title': 'Recent Meeting 2',
                    'date': '2024-01-19',
                    'participants': ['Princess Leia']
                }
            }
        ]
        mock_session.run.return_value = mock_result
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {}
            mock_neo4j_class.return_value = neo4j_manager
            
            meetings = get_recent_meetings(days=7)
            
            assert len(meetings) == 2
            assert all('date' in meeting for meeting in meetings)
    
    def test_get_meeting_details(self, mock_database_managers):
        """Test getting detailed meeting information."""
        neo4j_manager = mock_database_managers['neo4j']
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.single.return_value = {
            'meeting': {
                'title': 'Partnership Discussion',
                'date': '2024-01-15',
                'type': 'Business',
                'status': 'PENDING'
            },
            'participants': [
                {'name': 'Luke Skywalker', 'role': 'Jedi Knight'},
                {'name': 'Admiral Ackbar', 'role': 'Fleet Admiral'}
            ],
            'action_items': [
                {'item': 'Send technical overview', 'assigned_to': 'Admiral Ackbar'},
                {'item': 'Review pricing', 'assigned_to': 'Luke Skywalker'}
            ]
        }
        mock_session.run.return_value = mock_result
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {}
            mock_neo4j_class.return_value = neo4j_manager
            
            details = get_meeting_details("Partnership Discussion")
            
            assert details['meeting']['title'] == 'Partnership Discussion'
            assert len(details['participants']) == 2
            assert len(details['action_items']) == 2


class TestRelationshipQueries:
    """Test relationship query functions."""
    
    def test_find_relationships(self, mock_database_managers):
        """Test finding relationships between entities."""
        neo4j_manager = mock_database_managers['neo4j']
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.data.return_value = [
            {
                'person1': {'name': 'Luke Skywalker'},
                'relationship': {'type': 'WORKS_WITH'},
                'person2': {'name': 'Princess Leia'}
            },
            {
                'person1': {'name': 'Luke Skywalker'},
                'relationship': {'type': 'MET_WITH'},
                'person2': {'name': 'Admiral Ackbar'}
            }
        ]
        mock_session.run.return_value = mock_result
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {}
            mock_neo4j_class.return_value = neo4j_manager
            
            relationships = find_relationships("Luke Skywalker")
            
            assert len(relationships) == 2
            assert relationships[0]['relationship']['type'] == 'WORKS_WITH'
    
    def test_find_interactions_between_people(self, mock_database_managers):
        """Test finding interactions between specific people."""
        neo4j_manager = mock_database_managers['neo4j']
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.data.return_value = [
            {
                'interaction': {
                    'type': 'ATTENDED_MEETING',
                    'date': '2024-01-15',
                    'context': 'Partnership Discussion'
                }
            },
            {
                'interaction': {
                    'type': 'EMAIL_EXCHANGE',
                    'date': '2024-01-10',
                    'context': 'Technical Questions'
                }
            }
        ]
        mock_session.run.return_value = mock_result
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {}
            mock_neo4j_class.return_value = neo4j_manager
            
            interactions = find_interactions_between_people(
                "Luke Skywalker", 
                "Admiral Ackbar"
            )
            
            assert len(interactions) == 2
            assert interactions[0]['type'] == 'ATTENDED_MEETING'


class TestSemanticSearch:
    """Test semantic search functionality."""
    
    def test_search_documents(self, mock_database_managers):
        """Test document search."""
        chromadb_manager = mock_database_managers['chromadb']
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'ids': ['doc1', 'doc2'],
            'distances': [0.2, 0.4],
            'documents': ['Meeting notes about AI', 'Partnership discussion'],
            'metadatas': [
                {'source': 'meeting-notes.md', 'date': '2024-01-15'},
                {'source': 'partnership.md', 'date': '2024-01-10'}
            ]
        }
        chromadb_manager.client.get_or_create_collection.return_value = mock_collection
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.ChromaDBManager') as mock_chroma_class:
            mock_config.return_value = {'chromadb': {'document_collection': 'documents'}}
            mock_chroma_class.return_value = chromadb_manager
            
            results = search_documents("AI partnership")
            
            assert len(results) == 2
            assert results[0]['content'] == 'Meeting notes about AI'
            assert results[0]['metadata']['source'] == 'meeting-notes.md'
    
    def test_semantic_search(self, mock_database_managers):
        """Test general semantic search."""
        chromadb_manager = mock_database_managers['chromadb']
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'ids': ['summary1', 'summary2'],
            'distances': [0.1, 0.3],
            'documents': [
                'Weekly team meeting summary',
                'Partnership strategy discussion'
            ],
            'metadatas': [
                {'type': 'summary', 'date': '2024-01-20'},
                {'type': 'summary', 'date': '2024-01-15'}
            ]
        }
        chromadb_manager.client.get_or_create_collection.return_value = mock_collection
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.ChromaDBManager') as mock_chroma_class:
            mock_config.return_value = {'chromadb': {'summary_collection': 'summaries'}}
            mock_chroma_class.return_value = chromadb_manager
            
            results = semantic_search("team meeting", collection_type="summaries")
            
            assert len(results) == 2
            assert results[0]['content'] == 'Weekly team meeting summary'
    
    def test_search_by_topic(self, mock_database_managers):
        """Test search by topic combining multiple sources."""
        # Mock both ChromaDB and Neo4j responses
        chromadb_manager = mock_database_managers['chromadb']
        neo4j_manager = mock_database_managers['neo4j']
        
        # Mock ChromaDB response
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'ids': ['doc1'],
            'distances': [0.2],
            'documents': ['AI partnership discussion'],
            'metadatas': [{'source': 'partnership.md', 'date': '2024-01-15'}]
        }
        chromadb_manager.client.get_or_create_collection.return_value = mock_collection
        
        # Mock Neo4j response
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.data.return_value = [
            {
                'm': {
                    'title': 'AI Strategy Meeting',
                    'date': '2024-01-15',
                    'participants': ['Luke Skywalker', 'Princess Leia']
                }
            }
        ]
        mock_session.run.return_value = mock_result
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.ChromaDBManager') as mock_chroma_class, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {
                'chromadb': {'document_collection': 'documents'},
                'neo4j': {}
            }
            mock_chroma_class.return_value = chromadb_manager
            mock_neo4j_class.return_value = neo4j_manager
            
            results = search_by_topic("AI strategy")
            
            assert 'documents' in results
            assert 'meetings' in results
            assert len(results['documents']) == 1
            assert len(results['meetings']) == 1


class TestErrorHandling:
    """Test error handling in query functions."""
    
    def test_database_connection_error(self):
        """Test handling of database connection errors."""
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {}
            mock_neo4j_class.side_effect = Exception("Database connection failed")
            
            # Should handle gracefully
            people = find_people("Luke")
            assert people == []
    
    def test_chromadb_connection_error(self):
        """Test handling of ChromaDB connection errors."""
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.ChromaDBManager') as mock_chroma_class:
            mock_config.return_value = {}
            mock_chroma_class.side_effect = Exception("ChromaDB connection failed")
            
            # Should handle gracefully
            results = search_documents("test query")
            assert results == []
    
    def test_malformed_query_handling(self, mock_database_managers):
        """Test handling of malformed queries."""
        neo4j_manager = mock_database_managers['neo4j']
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        mock_session.run.side_effect = Exception("Query syntax error")
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {}
            mock_neo4j_class.return_value = neo4j_manager
            
            # Should handle gracefully
            people = find_people("Luke")
            assert people == []


class TestQueryOptimization:
    """Test query optimization and performance considerations."""
    
    def test_limited_results(self, mock_database_managers):
        """Test that queries respect result limits."""
        neo4j_manager = mock_database_managers['neo4j']
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        
        # Create 100 mock results
        mock_data = [
            {'p': {'name': f'Person {i}', 'role': 'Role'}} 
            for i in range(100)
        ]
        mock_result.data.return_value = mock_data
        mock_session.run.return_value = mock_result
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {}
            mock_neo4j_class.return_value = neo4j_manager
            
            people = find_people("Person", limit=10)
            
            # Should be limited even if more results available
            assert len(people) <= 10
    
    def test_query_caching(self, mock_database_managers):
        """Test query result caching (if implemented)."""
        # This would test caching functionality if we implement it
        pass
    
    def test_efficient_relationship_queries(self, mock_database_managers):
        """Test that relationship queries are efficient."""
        neo4j_manager = mock_database_managers['neo4j']
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {}
            mock_neo4j_class.return_value = neo4j_manager
            
            find_relationships("Luke Skywalker", max_depth=2)
            
            # Verify that query was called with appropriate parameters
            mock_session.run.assert_called()
            call_args = mock_session.run.call_args
            query = call_args[0][0]
            
            # Should include depth limitation
            assert "LIMIT" in query.upper() or any(
                "max_depth" in str(arg) or "depth" in str(arg) 
                for arg in call_args[1].values()
            )


class TestQueryIntegration:
    """Test integration between different query types."""
    
    @pytest.mark.integration
    def test_cross_reference_search(self, mock_database_managers):
        """Test searching across multiple data sources."""
        # This would test combining Neo4j and ChromaDB results
        chromadb_manager = mock_database_managers['chromadb']
        neo4j_manager = mock_database_managers['neo4j']
        
        # Setup mocks for both databases
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'ids': ['doc1'],
            'distances': [0.2],
            'documents': ['Meeting with Luke about Force training'],
            'metadatas': [{'source': 'meeting.md'}]
        }
        chromadb_manager.client.get_or_create_collection.return_value = mock_collection
        
        mock_session = neo4j_manager.driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.data.return_value = [
            {'p': {'name': 'Luke Skywalker', 'role': 'Jedi Knight'}}
        ]
        mock_session.run.return_value = mock_result
        
        with patch('arc_query.get_config') as mock_config, \
             patch('arc_query.ChromaDBManager') as mock_chroma_class, \
             patch('arc_query.Neo4jManager') as mock_neo4j_class:
            mock_config.return_value = {
                'chromadb': {'document_collection': 'documents'}
            }
            mock_chroma_class.return_value = chromadb_manager
            mock_neo4j_class.return_value = neo4j_manager
            
            # Test combined search
            results = search_by_topic("Luke Force")
            
            assert 'documents' in results
            assert 'people' in results or 'meetings' in results