"""
Diagnostic script to investigate the "*" relationship issue in Neo4j.
"""

import asyncio
from clients.graph.Neo4jClient import Neo4jClient

async def diagnose_graph_relationships():
    """Diagnose relationship issues in the Neo4j graph database."""
    
    print("🔍 Diagnosing Neo4j Graph Relationships...")
    
    # Initialize Neo4j client
    client = Neo4jClient()
    
    try:
        # Connect to Neo4j
        await client.connect()
        print("✅ Connected to Neo4j")
        
        # Run diagnostics
        await client.diagnose_relationships()
        
        # Ask user if they want to clean up invalid relationships
        print("\n" + "="*50)
        response = input("Do you want to clean up any invalid relationships? (y/n): ")
        
        if response.lower() == 'y':
            await client.cleanup_invalid_relationships()
            print("✅ Cleanup completed")
            
            # Run diagnostics again to see the result
            print("\n📊 Post-cleanup diagnostics:")
            await client.diagnose_relationships()
        
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")
    
    finally:
        # Close connection
        await client.close()
        print("🔌 Disconnected from Neo4j")

async def test_relationship_creation():
    """Test creating specific relationships to ensure they work correctly."""
    
    print("\n🧪 Testing Relationship Creation...")
    
    client = Neo4jClient()
    
    try:
        await client.connect()
        
        # Test creating a simple relationship manually
        async with client.driver.session() as session:
            # Create test nodes and relationship
            test_query = '''
            CREATE (p:TestPaper {id: "test_paper_1"})
            CREATE (i:TestInstitution {id: "test_inst_1"})
            CREATE (p)-[:TEST_RELATIONSHIP]->(i)
            RETURN p, i
            '''
            
            await session.run(test_query)
            print("✅ Created test relationship")
            
            # Verify the relationship
            verify_query = '''
            MATCH (p:TestPaper)-[r]->(i:TestInstitution)
            RETURN type(r) as relationship_type
            '''
            
            result = await session.run(verify_query)
            record = await result.single()
            
            if record:
                print(f"✅ Test relationship type: {record['relationship_type']}")
            else:
                print("❌ No test relationship found")
            
            # Clean up test nodes
            cleanup_query = '''
            MATCH (n:TestPaper), (m:TestInstitution)
            DETACH DELETE n, m
            '''
            await session.run(cleanup_query)
            print("🧹 Cleaned up test nodes")
    
    except Exception as e:
        print(f"❌ Error during relationship testing: {e}")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(diagnose_graph_relationships())
    asyncio.run(test_relationship_creation())