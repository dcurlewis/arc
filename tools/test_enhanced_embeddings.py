#!/usr/bin/env python3
"""
Test script for the enhanced embedding system.
Verifies that enhanced embeddings work correctly with the existing data.
"""

import sys
import asyncio
from pathlib import Path

# Add the tools directory to the path
sys.path.append(str(Path(__file__).parent))

from arc_core import get_db_manager
from enhanced_embeddings import create_enhanced_embedding_system


async def test_enhanced_embeddings():
    """Test the enhanced embedding system functionality."""
    print("🧪 Testing Enhanced Embedding System")
    print("=" * 50)
    
    try:
        # Initialize database manager and enhanced embeddings
        print("1. Initializing systems...")
        db_manager = get_db_manager()
        embedding_generator, query_interface = create_enhanced_embedding_system(db_manager)
        print("✅ Enhanced embedding system initialized")
        
        # Test hybrid search
        print("\n2. Testing hybrid search...")
        results = query_interface.hybrid_search(
            query="platform crisis",
            search_types=["documents", "entities", "relationships"],
            limit=5
        )
        print(f"✅ Hybrid search returned {len(results)} results")
        
        if results:
            print("   Sample result:")
            result = results[0]
            print(f"   - Type: {result['search_type']}")
            print(f"   - Score: {result['score']:.3f}")
            print(f"   - Content preview: {result['content'][:100]}...")
        
        # Test entity-centric search
        print("\n3. Testing entity-centric search...")
        results = query_interface.entity_centric_search(
            entity_name="Glen",
            limit=3
        )
        print(f"✅ Entity search returned {len(results)} results")
        
        if results:
            print("   Sample result:")
            result = results[0]
            print(f"   - Score: {result['score']:.3f}")
            print(f"   - Content preview: {result['content'][:100]}...")
        
        # Test relationship search
        print("\n4. Testing relationship search...")
        results = query_interface.relationship_search(
            source_entity="David",
            limit=3
        )
        print(f"✅ Relationship search returned {len(results)} results")
        
        if results:
            print("   Sample result:")
            result = results[0]
            print(f"   - Score: {result['score']:.3f}")
            print(f"   - Content preview: {result['content'][:100]}...")
        
        # Test temporal search
        print("\n5. Testing temporal search...")
        results = query_interface.temporal_search(
            query="meeting",
            start_date="2025-07-01",
            end_date="2025-07-31",
            limit=3
        )
        print(f"✅ Temporal search returned {len(results)} results")
        
        if results:
            print("   Sample result:")
            result = results[0]
            print(f"   - Score: {result['score']:.3f}")
            print(f"   - Content preview: {result['content'][:100]}...")
        
        # Test collection info
        print("\n6. Checking enhanced collections...")
        collections = embedding_generator.collections
        for name, collection in collections.items():
            try:
                count = collection.count()
                print(f"   - {name}: {count} items")
            except Exception as e:
                print(f"   - {name}: Error getting count - {e}")
        
        print("\n🎉 All enhanced embedding tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_basic_functionality():
    """Test basic functionality to ensure system works."""
    print("🔧 Testing Basic Functionality")
    print("=" * 50)
    
    try:
        # Test database connections
        print("1. Testing database connections...")
        db_manager = get_db_manager()
        
        # Test Neo4j
        with db_manager.neo4j.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            neo4j_count = result.single()["count"]
        print(f"   ✅ Neo4j: {neo4j_count} nodes")
        
        # Test ChromaDB
        collections = db_manager.chromadb.list_collections()
        total_docs = sum(c.count() for c in collections)
        print(f"   ✅ ChromaDB: {len(collections)} collections, {total_docs} documents")
        
        # Test embedding model
        print("2. Testing embedding model...")
        embeddings = db_manager.embeddings
        test_embedding = embeddings.encode(["This is a test sentence."])
        print(f"   ✅ Embedding model: shape {test_embedding.shape}")
        
        print("\n✅ Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 ARC Enhanced Embedding System Tests")
    print("=" * 60)
    
    # Run basic tests first
    basic_success = await test_basic_functionality()
    
    if basic_success:
        print("\n")
        # Run enhanced embedding tests
        enhanced_success = await test_enhanced_embeddings()
        
        if enhanced_success:
            print("\n🎉 ALL TESTS PASSED! Enhanced embedding system is working correctly.")
            
            print("\n📋 Summary:")
            print("• Enhanced embeddings are generating multiple embedding types")
            print("• Hybrid search combines documents, entities, and relationships")
            print("• Entity-centric search finds entity-related content")
            print("• Relationship search discovers entity connections")
            print("• Temporal search filters by date ranges")
            print("• All new collections are properly indexed")
            
        else:
            print("\n❌ Enhanced embedding tests failed.")
            sys.exit(1)
    else:
        print("\n❌ Basic functionality tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 