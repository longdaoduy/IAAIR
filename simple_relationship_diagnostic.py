"""
Simple diagnostic script to investigate the '*' relationship issue in Neo4j.
"""

from neo4j import GraphDatabase
import asyncio

# Neo4j connection details from GraphDBConfig
NEO4J_URI = "neo4j+s://4d600688.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "LkIC3FNqbGTGHtkxiYujARXY6IUZnvyvyrMFJBeagCI"

def diagnose_relationships_sync():
    """Synchronous version to diagnose relationship issues."""
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            print("🔍 Diagnosing Neo4j Relationships...")
            
            # 1. Check all relationship types
            result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType")
            relationships = [record["relationshipType"] for record in result]
            
            print("\n=== All Relationship Types ===")
            for rel in relationships:
                print(f"  - '{rel}'")
            
            # 2. Look for any "*" relationships specifically
            star_query = """
            MATCH ()-[r]-()
            WHERE type(r) = '*'
            RETURN type(r) as rel_type, count(*) as count
            """
            result = session.run(star_query)
            star_records = list(result)
            
            if star_records:
                print("\n❌ Found '*' relationships:")
                for record in star_records:
                    print(f"  - Count: {record['count']}")
            else:
                print("\n✅ No '*' relationships found in database")
            
            # 3. Check for empty or null relationship types
            null_query = """
            MATCH ()-[r]-()
            WHERE type(r) = '' OR type(r) IS NULL
            RETURN type(r) as rel_type, count(*) as count
            """
            result = session.run(null_query)
            null_records = list(result)
            
            if null_records:
                print("\n❌ Found relationships with empty/null types:")
                for record in null_records:
                    print(f"  - Type: '{record['rel_type']}', Count: {record['count']}")
            else:
                print("\n✅ No empty/null relationship types found")
            
            # 4. Show sample relationships to see what's actually in the database
            sample_query = """
            MATCH (a)-[r]->(b)
            RETURN labels(a)[0] as from_label, type(r) as rel_type, labels(b)[0] as to_label, count(*) as count
            ORDER BY count DESC
            LIMIT 10
            """
            result = session.run(sample_query)
            sample_records = list(result)
            
            print("\n=== Sample Relationships (Top 10 by count) ===")
            for record in sample_records:
                print(f"  {record['from_label']} -[{record['rel_type']}]-> {record['to_label']} ({record['count']})")
            
            # 5. Check if there are any unusual relationship patterns
            unusual_query = """
            MATCH ()-[r]-()
            WITH type(r) as rel_type, count(*) as count
            WHERE size(rel_type) > 50 OR rel_type CONTAINS '*' OR rel_type CONTAINS ' ' OR rel_type STARTS WITH '_'
            RETURN rel_type, count
            ORDER BY count DESC
            """
            result = session.run(unusual_query)
            unusual_records = list(result)
            
            if unusual_records:
                print("\n⚠️ Unusual relationship types found:")
                for record in unusual_records:
                    print(f"  - '{record['rel_type']}': {record['count']}")
            else:
                print("\n✅ No unusual relationship types found")
    
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
        print("\n💡 Possible solutions:")
        print("  1. Check that Neo4j is running")
        print("  2. Verify the connection details (URI, username, password)")
        print("  3. Make sure the Neo4j Python driver is installed: pip install neo4j")
        
    finally:
        driver.close()

if __name__ == "__main__":
    print("Note: You may need to update the NEO4J_PASSWORD variable in this script")
    print("You can find the password in your GraphDBConfig.py file or Neo4j configuration")
    print()
    diagnose_relationships_sync()