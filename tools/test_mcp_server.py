#!/usr/bin/env python3
"""Test script to validate MCP server functionality"""

import sys
sys.path.append('tools')

try:
    import arc_mcp_server
    from arc_core import get_config, get_db_manager
    
    # Test configuration loading
    config = get_config()
    print("✅ Configuration loaded successfully")
    
    # Test database manager initialization
    db_manager = get_db_manager()
    print("✅ Database manager initialized successfully")
    
    # Test basic database connectivity
    with db_manager.neo4j.session() as session:
        result = session.run("RETURN 1 as test")
        record = result.single()
        if record and record["test"] == 1:
            print("✅ Neo4j connection successful")
        else:
            print("❌ Neo4j connection failed")
    
    # Test ChromaDB
    collections = db_manager.chromadb.list_collections()
    print(f"✅ ChromaDB connection successful - {len(collections)} collections found")
    
    print("\n🎉 ARC MCP Server validation complete - all systems ready!")
    
except Exception as e:
    print(f"❌ Error during validation: {str(e)}")
    sys.exit(1)
