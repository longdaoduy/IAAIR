"""
Simple queries to verify institution-paper relationships and investigate '*' relationships.
"""

# Cypher queries to run in Neo4j Browser:

QUERIES = {
    "Check all relationship types": """
CALL db.relationshipTypes() YIELD relationshipType
RETURN relationshipType
ORDER BY relationshipType
""",

    "Count relationships by type": """
MATCH ()-[r]-()
RETURN type(r) as relationship_type, count(*) as count
ORDER BY count DESC
""",

    "Check institution-paper relationships": """
MATCH (p:Paper)-[r:ASSOCIATED_WITH_INSTITUTION]->(i:Institution)
RETURN p.title, i.name, type(r) as relationship_type
LIMIT 10
""",

    "Check for any '*' relationships": """
MATCH ()-[r]-()
WHERE type(r) = '*'
RETURN type(r), count(*) as count
""",

    "Sample all relationships": """
MATCH (a)-[r]->(b)
RETURN labels(a)[0] as from_label, type(r) as rel_type, labels(b)[0] as to_label, count(*) as count
ORDER BY count DESC
LIMIT 20
""",

    "Check specific paper relationships": """
MATCH (p:Paper)
WHERE p.id CONTAINS "openalex"
MATCH (p)-[r]-(connected)
RETURN p.title, type(r) as relationship_type, labels(connected)[0] as connected_type, count(*) as count
ORDER BY count DESC
LIMIT 10
""",

    "Check institutions": """
MATCH (i:Institution)
RETURN i.id, i.name, i.country
LIMIT 10
""",

    "Verify paper-institution connections": """
MATCH (p:Paper)-[:ASSOCIATED_WITH_INSTITUTION]->(i:Institution)
RETURN count(*) as total_connections
"""
}

if __name__ == "__main__":
    print("🔍 Neo4j Diagnostic Queries")
    print("="*50)
    print("Copy and paste these queries into Neo4j Browser to diagnose the relationship issues:")
    print()
    
    for description, query in QUERIES.items():
        print(f"## {description}")
        print("```cypher")
        print(query.strip())
        print("```")
        print()
    
    print("📋 Instructions:")
    print("1. Open Neo4j Browser in your web browser")
    print("2. Connect to your database")
    print("3. Run each query one by one")
    print("4. Look for any relationships showing as '*' in the results")
    print("5. Pay attention to the 'Count relationships by type' results")
    print()
    print("💡 If you see '*' relationships:")
    print("- It might just be a browser display issue")
    print("- Try refreshing the browser or restarting Neo4j")
    print("- Check if the same relationships show proper names in the query results")