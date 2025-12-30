"""
PrimeKG (Precision Medicine Knowledge Graph) Integration

This module provides integration with PrimeKG, a large-scale multimodal knowledge graph_store
designed for precision medicine. PrimeKG integrates 20 high-quality biomedical resources
to describe 17,080 diseases with 4,050,249 relationships across ten biological scales.

Paper: "Building a knowledge graph_store to enable precision medicine"
Authors: Payal Chandak, Kexin Huang, Marinka Zitnik
Published: Nature Scientific Data (2023)
DOI: https://doi.org/10.1038/s41597-023-01960-3
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import networkx as nx
import json

from models.schemas.schemas import Document

logger = logging.getLogger(__name__)


class PrimeKGIntegrator:
    """
    Integration module for PrimeKG (Precision Medicine Knowledge Graph)
    
    Provides access to a comprehensive biomedical knowledge graph_store with
    17K+ diseases, 4M+ relationships, and multimodal clinical information.
    
    Data Access:
    - Harvard Dataverse: https://doi.org/10.7910/DVN/IXA7BM
    - GitHub: https://github.com/mims-harvard/PrimeKG
    """
    
    # Known entity types in PrimeKG
    ENTITY_TYPES = {
        'disease': 'Disease/Medical Condition',
        'drug': 'Drug/Pharmaceutical',
        'protein': 'Protein/Gene',
        'pathway': 'Biological Pathway',
        'anatomy': 'Anatomical Entity',
        'phenotype': 'Phenotype/Clinical Finding',
        'biological_process': 'GO Biological Process',
        'molecular_function': 'GO Molecular Function',
        'cellular_component': 'GO Cellular Component',
        'exposure': 'Environmental Exposure'
    }
    
    # Known relation types in PrimeKG
    RELATION_TYPES = {
        'indication': 'Drug indicated for disease',
        'contraindication': 'Drug contraindicated for disease',
        'off_label_use': 'Off-label drug use for disease',
        'protein_protein': 'Protein-protein interaction',
        'disease_phenotype_positive': 'Disease has phenotype',
        'disease_phenotype_negative': 'Disease does not have phenotype',
        'pathway_pathway': 'Pathway hierarchy',
        'protein_pathway': 'Protein in pathway',
        'anatomy_anatomy': 'Anatomical hierarchy',
        'protein_go_term': 'Protein has GO annotation'
    }
    
    def __init__(self, data_path: str = "/home/dnhoa/IAAIR/data/primekg"):
        """
        Initialize PrimeKG integrator
        
        Args:
            data_path: Path to PrimeKG data files
        """
        self.data_path = Path(data_path)
        self.kg_data = None
        self.node_data = None
        self.drug_features = None
        self.disease_features = None
        
        # Graph analysis
        self.entity_types = {}
        self.relation_types = {}
        self.graph = None
        
        # Statistics
        self.stats = {}
        
        logger.info(f"Initializing PrimeKG integrator with data path: {data_path}")
    
    def download_data(self) -> bool:
        """
        Download PrimeKG data from Harvard Dataverse
        
        Returns:
            True if successful, False otherwise
        """
        import requests

        logger.info("Downloading PrimeKG data from Harvard Dataverse...")
        
        # Create data directory
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Harvard Dataverse API URLs (these are example URLs - check actual dataverse for current links)
            base_url = "https://dataverse.harvard.edu/api/access/datafile"
            
            # Note: These file IDs are examples - you'll need to get actual ones from the dataverse
            files_to_download = {
                'kg.csv': f"{base_url}/6180620",  # Main knowledge graph_store
                'nodes.csv': f"{base_url}/6180621",  # Node information
                'drug_features.csv': f"{base_url}/6180622",  # Drug features
                'disease_features.csv': f"{base_url}/6180623"  # Disease features
            }
            
            for filename, url in files_to_download.items():
                file_path = self.data_path / filename
                
                if file_path.exists():
                    logger.info(f"File {filename} already exists, skipping download")
                    continue
                
                logger.info(f"Downloading {filename}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Downloaded {filename} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)")
            
            logger.info("PrimeKG data download completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download PrimeKG data: {e}")
            logger.info("Please manually download from: https://doi.org/10.7910/DVN/IXA7BM")
            return False
    
    def load_data(self) -> bool:
        """Load PrimeKG data files"""
        try:
            logger.info("Loading PrimeKG data files...")
            
            # Load main knowledge graph_store
            kg_file = self.data_path / "kg.csv"
            if not kg_file.exists():
                # Try alternative names
                for alt_name in ["kg_giant.csv", "primekg.csv", "knowledge_graph.csv"]:
                    alt_file = self.data_path / alt_name
                    if alt_file.exists():
                        kg_file = alt_file
                        break
            
            if kg_file.exists():
                self.kg_data = pd.read_csv(kg_file)
                logger.info(f"Loaded knowledge graph_store: {len(self.kg_data)} edges")
            else:
                logger.error(f"Knowledge graph_store file not found in {self.data_path}")
                logger.info("Expected files: kg.csv, kg_giant.csv, or primekg.csv")
                return False
            
            # Load node information if available
            for node_file_name in ["nodes.csv", "node_info.csv", "entities.csv"]:
                nodes_file = self.data_path / node_file_name
                if nodes_file.exists():
                    self.node_data = pd.read_csv(nodes_file)
                    logger.info(f"Loaded node data: {len(self.node_data)} nodes")
                    break
            
            # Load drug features if available
            for drug_file_name in ["drug_features.csv", "drugs.csv"]:
                drug_file = self.data_path / drug_file_name
                if drug_file.exists():
                    self.drug_features = pd.read_csv(drug_file)
                    logger.info(f"Loaded drug features: {len(self.drug_features)} drugs")
                    break
            
            # Load disease features if available
            for disease_file_name in ["disease_features.csv", "diseases.csv"]:
                disease_file = self.data_path / disease_file_name
                if disease_file.exists():
                    self.disease_features = pd.read_csv(disease_file)
                    logger.info(f"Loaded disease features: {len(self.disease_features)} diseases")
                    break
            
            # Analyze graph_store structure
            self._analyze_graph_structure()
            self._compute_statistics()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load PrimeKG data: {e}")
            return False
    
    def _analyze_graph_structure(self):
        """Analyze the structure of the knowledge graph_store"""
        if self.kg_data is not None:
            # Determine column names (may vary across versions)
            relation_col = None
            source_col = None
            target_col = None
            
            for col in self.kg_data.columns:
                if 'relation' in col.lower() or 'edge' in col.lower():
                    relation_col = col
                elif any(x in col.lower() for x in ['source', 'x_id', 'head']):
                    source_col = col
                elif any(x in col.lower() for x in ['target', 'y_id', 'tail']):
                    target_col = col
            
            # Count relation types
            if relation_col and relation_col in self.kg_data.columns:
                self.relation_types = self.kg_data[relation_col].value_counts().to_dict()
            
            # Count entity types if node data available
            if self.node_data is not None:
                type_cols = [col for col in self.node_data.columns if 'type' in col.lower()]
                if type_cols:
                    self.entity_types = self.node_data[type_cols[0]].value_counts().to_dict()
            
            logger.info(f"Found {len(self.relation_types)} relation types")
            logger.info(f"Found {len(self.entity_types)} entity types")
            
            # Store column mappings for later use
            self.column_mapping = {
                'relation': relation_col,
                'source': source_col,
                'target': target_col
            }
    
    def _compute_statistics(self):
        """Compute comprehensive statistics about PrimeKG"""
        if self.kg_data is None:
            return
        
        source_col = self.column_mapping.get('source')
        target_col = self.column_mapping.get('target')
        
        if source_col and target_col:
            # Basic graph_store statistics
            all_nodes = set(list(self.kg_data[source_col].unique()) + 
                           list(self.kg_data[target_col].unique()))
            
            self.stats = {
                'total_edges': len(self.kg_data),
                'total_nodes': len(all_nodes),
                'relation_types': len(self.relation_types),
                'entity_types': len(self.entity_types),
                'relation_counts': self.relation_types,
                'entity_counts': self.entity_types,
                'avg_degree': 2 * len(self.kg_data) / len(all_nodes) if all_nodes else 0
            }
            
            # Disease-specific statistics
            disease_relations = [k for k in self.relation_types.keys() 
                               if any(term in k.lower() for term in ['indication', 'disease', 'phenotype'])]
            disease_edge_count = sum(self.relation_types.get(rel, 0) for rel in disease_relations)
            self.stats['disease_related_edges'] = disease_edge_count
            
            logger.info(f"Computed statistics: {self.stats['total_nodes']} nodes, {self.stats['total_edges']} edges")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about PrimeKG"""
        return self.stats.copy()
    
    def search_entities(
        self, 
        query: str, 
        entity_type: Optional[str] = None,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for entities in PrimeKG
        
        Args:
            query: Search query
            entity_type: Optional entity type filter ('disease', 'drug', 'protein', etc.)
            max_results: Maximum number of results
            
        Returns:
            List of matching entities
        """
        if self.node_data is None:
            logger.warning("Node data not available for entity search")
            return []
        
        # Prepare search data
        search_data = self.node_data.copy()
        
        # Filter by entity type if specified
        if entity_type and 'node_type' in search_data.columns:
            search_data = search_data[search_data['node_type'] == entity_type]
        
        # Search in text columns
        query_lower = query.lower()
        text_columns = [col for col in search_data.columns 
                       if search_data[col].dtype == 'object']
        
        # Search across relevant text columns
        search_cols = []
        for col in text_columns:
            if any(term in col.lower() for term in ['name', 'label', 'description', 'id']):
                search_cols.append(col)
        
        if not search_cols:
            search_cols = text_columns[:3]  # Use first 3 text columns as fallback
        
        # Combine search across columns
        mask = pd.Series(False, index=search_data.index)
        for col in search_cols:
            mask |= search_data[col].astype(str).str.lower().str.contains(query_lower, na=False)
        
        matches = search_data[mask]
        
        # Convert to list of dictionaries
        results = []
        for _, row in matches.head(max_results).iterrows():
            result = row.to_dict()
            # Clean up NaN values
            result = {k: v for k, v in result.items() if pd.notna(v)}
            results.append(result)
        
        return results
    
    def search_diseases(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search specifically for diseases"""
        return self.search_entities(query, entity_type='disease', max_results=max_results)
    
    def search_drugs(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search specifically for drugs"""
        return self.search_entities(query, entity_type='drug', max_results=max_results)
    
    def search_proteins(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search specifically for proteins"""
        return self.search_entities(query, entity_type='protein', max_results=max_results)
    
    def get_entity_relationships(
        self, 
        entity_id: str,
        relation_filter: Optional[List[str]] = None,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all relationships for a specific entity
        
        Args:
            entity_id: Entity identifier
            relation_filter: Optional list of relation types to include
            max_results: Optional limit on number of results
            
        Returns:
            List of relationships involving the entity
        """
        if self.kg_data is None:
            return []
        
        source_col = self.column_mapping.get('source')
        target_col = self.column_mapping.get('target')
        relation_col = self.column_mapping.get('relation')
        
        if not all([source_col, target_col]):
            logger.warning("Cannot determine source/target columns in knowledge graph_store")
            return []
        
        # Find edges involving this entity
        entity_edges = self.kg_data[
            (self.kg_data[source_col] == entity_id) | 
            (self.kg_data[target_col] == entity_id)
        ].copy()
        
        # Apply relation filter
        if relation_filter and relation_col:
            entity_edges = entity_edges[entity_edges[relation_col].isin(relation_filter)]
        
        # Apply result limit
        if max_results:
            entity_edges = entity_edges.head(max_results)
        
        relationships = []
        for _, edge in entity_edges.iterrows():
            rel = {
                'source': edge[source_col],
                'target': edge[target_col],
                'source_type': self._get_entity_type(edge[source_col]),
                'target_type': self._get_entity_type(edge[target_col])
            }
            
            if relation_col:
                rel['relation'] = edge[relation_col]
            
            # Add other edge attributes
            for col, val in edge.items():
                if col not in [source_col, target_col, relation_col] and pd.notna(val):
                    rel[col] = val
            
            relationships.append(rel)
        
        return relationships
    
    def _get_entity_type(self, entity_id: str) -> Optional[str]:
        """Get the type of an entity by its ID"""
        if self.node_data is None:
            return None
        
        # Find node by ID (try different possible ID column names)
        id_columns = [col for col in self.node_data.columns 
                     if 'id' in col.lower()]
        
        for id_col in id_columns:
            entity_info = self.node_data[self.node_data[id_col] == entity_id]
            if not entity_info.empty:
                type_cols = [col for col in entity_info.columns if 'type' in col.lower()]
                if type_cols:
                    return entity_info.iloc[0][type_cols[0]]
        
        return None
    
    def get_drug_disease_relationships(self) -> pd.DataFrame:
        """
        Extract drug-disease relationships for therapeutic insights
        
        Returns:
            DataFrame with drug-disease relationships including indications,
            contraindications, and off-label uses
        """
        if self.kg_data is None:
            logger.warning("Knowledge graph_store not loaded")
            return pd.DataFrame()
        
        relation_col = self.column_mapping.get('relation')
        if not relation_col:
            logger.warning("Cannot determine relation column")
            return pd.DataFrame()
        
        # Filter for drug-disease relations
        drug_disease_relations = ['indication', 'contraindication', 'off_label_use']
        
        # Find relations that contain these terms
        drug_disease_mask = self.kg_data[relation_col].str.lower().str.contains(
            '|'.join(drug_disease_relations), na=False
        )
        
        drug_disease_edges = self.kg_data[drug_disease_mask].copy()
        
        logger.info(f"Found {len(drug_disease_edges)} drug-disease relationships")
        return drug_disease_edges
    
    def create_subgraph(
        self, 
        central_entities: List[str], 
        max_hops: int = 2,
        relation_filter: Optional[List[str]] = None,
        max_nodes: int = 1000
    ) -> Dict[str, Any]:
        """
        Create a subgraph around central entities
        
        Args:
            central_entities: List of central entity IDs
            max_hops: Maximum number of hops from central entities
            relation_filter: Optional list of relation types to include
            max_nodes: Maximum number of nodes to include
            
        Returns:
            Subgraph data structure with nodes and edges
        """
        if self.kg_data is None:
            return {'nodes': [], 'edges': []}
        
        source_col = self.column_mapping.get('source')
        target_col = self.column_mapping.get('target')
        relation_col = self.column_mapping.get('relation')
        
        if not all([source_col, target_col]):
            return {'nodes': [], 'edges': []}
        
        visited_nodes = set()
        current_level = set(central_entities)
        all_edges = []
        
        for hop in range(max_hops):
            if len(visited_nodes) >= max_nodes:
                break
                
            next_level = set()
            
            # Get edges for current level nodes
            level_edges = self.kg_data[
                (self.kg_data[source_col].isin(current_level)) |
                (self.kg_data[target_col].isin(current_level))
            ].copy()
            
            # Apply relation filter if specified
            if relation_filter and relation_col:
                level_edges = level_edges[level_edges[relation_col].isin(relation_filter)]
            
            # Add edges and collect next level nodes
            for _, edge in level_edges.iterrows():
                if len(visited_nodes) >= max_nodes:
                    break
                
                edge_dict = edge.to_dict()
                # Clean NaN values
                edge_dict = {k: v for k, v in edge_dict.items() if pd.notna(v)}
                all_edges.append(edge_dict)
                
                next_level.add(edge[source_col])
                next_level.add(edge[target_col])
            
            visited_nodes.update(current_level)
            current_level = next_level - visited_nodes
        
        # Collect unique nodes
        all_nodes = set()
        for edge in all_edges:
            all_nodes.add(edge[source_col])
            all_nodes.add(edge[target_col])
        
        # Add node information if available
        nodes_info = []
        for node_id in list(all_nodes)[:max_nodes]:
            node_info = {'id': node_id, 'type': self._get_entity_type(node_id)}
            
            # Add detailed node information if available
            if self.node_data is not None:
                id_columns = [col for col in self.node_data.columns if 'id' in col.lower()]
                for id_col in id_columns:
                    node_data = self.node_data[self.node_data[id_col] == node_id]
                    if not node_data.empty:
                        node_dict = node_data.iloc[0].to_dict()
                        # Clean NaN values
                        node_dict = {k: v for k, v in node_dict.items() if pd.notna(v)}
                        node_info.update(node_dict)
                        break
            
            nodes_info.append(node_info)
        
        return {
            'nodes': nodes_info,
            'edges': all_edges,
            'central_entities': central_entities,
            'max_hops': max_hops,
            'total_nodes': len(nodes_info),
            'total_edges': len(all_edges)
        }
    
    def build_networkx_graph(self, relation_filter: Optional[List[str]] = None) -> nx.Graph:
        """
        Build a NetworkX graph_store from PrimeKG data
        
        Args:
            relation_filter: Optional list of relation types to include
            
        Returns:
            NetworkX graph_store object
        """
        if self.kg_data is None:
            return nx.Graph()
        
        source_col = self.column_mapping.get('source')
        target_col = self.column_mapping.get('target')
        relation_col = self.column_mapping.get('relation')
        
        if not all([source_col, target_col]):
            return nx.Graph()
        
        # Filter data if needed
        data = self.kg_data.copy()
        if relation_filter and relation_col:
            data = data[data[relation_col].isin(relation_filter)]
        
        # Create graph_store
        G = nx.Graph()
        
        # Add edges
        for _, row in data.iterrows():
            edge_attrs = {}
            if relation_col:
                edge_attrs['relation'] = row[relation_col]
            
            # Add other edge attributes
            for col, val in row.items():
                if col not in [source_col, target_col] and pd.notna(val):
                    edge_attrs[col] = val
            
            G.add_edge(row[source_col], row[target_col], **edge_attrs)
        
        # Add node attributes if available
        if self.node_data is not None:
            for _, node_row in self.node_data.iterrows():
                id_columns = [col for col in node_row.index if 'id' in col.lower()]
                if id_columns:
                    node_id = node_row[id_columns[0]]
                    if node_id in G.nodes:
                        node_attrs = {k: v for k, v in node_row.items() if pd.notna(v)}
                        G.nodes[node_id].update(node_attrs)
        
        return G
    
    def enhance_documents_with_primekg(
        self, 
        documents: List[Document]
    ) -> List[Document]:
        """
        Enhance documents with PrimeKG knowledge
        
        Args:
            documents: Documents to enhance
            
        Returns:
            Enhanced documents with PrimeKG metadata
        """
        enhanced_docs = []
        
        for doc in documents:
            logger.info(f"Enhancing document {doc.id} with PrimeKG knowledge")
            
            # Combine title and abstract for search
            search_text = f"{doc.title} {doc.abstract or ''}"
            
            # Search for relevant entities
            diseases = self.search_diseases(search_text, max_results=5)
            drugs = self.search_drugs(search_text, max_results=5)
            proteins = self.search_proteins(search_text, max_results=5)
            
            # Create enhanced metadata
            enhanced_metadata = doc.metadata.copy()
            enhanced_metadata.update({
                'primekg_diseases': diseases,
                'primekg_drugs': drugs,
                'primekg_proteins': proteins,
                'primekg_enhanced': True,
                'primekg_enhancement_timestamp': datetime.utcnow().isoformat(),
                'primekg_total_entities': len(diseases) + len(drugs) + len(proteins)
            })
            
            # Create enhanced document
            enhanced_doc = Document(
                id=doc.id,
                title=doc.title,
                authors=doc.authors,
                abstract=doc.abstract,
                content=doc.content,
                doi=doc.doi,
                url=doc.url,
                publication_date=doc.publication_date,
                venue=doc.venue,
                citations=doc.citations,
                embeddings=doc.embeddings,
                metadata=enhanced_metadata,
                version=doc.version + 1,
                created_at=doc.created_at,
                updated_at=datetime.utcnow()
            )
            
            enhanced_docs.append(enhanced_doc)
        
        return enhanced_docs
    
    def export_subgraph(
        self, 
        subgraph_data: Dict[str, Any], 
        output_path: str,
        format: str = 'json'
    ):
        """
        Export subgraph data to file
        
        Args:
            subgraph_data: Subgraph data from create_subgraph
            output_path: Output file path
            format: Export format ('json', 'csv', 'graphml')
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_file, 'w') as f:
                json.dump(subgraph_data, f, indent=2, default=str)
        
        elif format.lower() == 'csv':
            # Export nodes and edges as separate CSV files
            base_name = output_file.stem
            
            nodes_df = pd.DataFrame(subgraph_data['nodes'])
            nodes_df.to_csv(output_file.parent / f"{base_name}_nodes.csv", index=False)
            
            edges_df = pd.DataFrame(subgraph_data['edges'])
            edges_df.to_csv(output_file.parent / f"{base_name}_edges.csv", index=False)
        
        elif format.lower() == 'graphml':
            # Create NetworkX graph_store and export
            G = nx.Graph()
            
            # Add nodes
            for node in subgraph_data['nodes']:
                node_id = node['id']
                node_attrs = {k: v for k, v in node.items() if k != 'id'}
                G.add_node(node_id, **node_attrs)
            
            # Add edges
            source_col = self.column_mapping.get('source', 'source')
            target_col = self.column_mapping.get('target', 'target')
            
            for edge in subgraph_data['edges']:
                source = edge[source_col]
                target = edge[target_col]
                edge_attrs = {k: v for k, v in edge.items() 
                             if k not in [source_col, target_col]}
                G.add_edge(source, target, **edge_attrs)
            
            nx.write_graphml(G, output_file)
        
        logger.info(f"Exported subgraph to {output_file} in {format} format")


# Utility functions for integration

def download_primekg_data(data_path: str = "/home/dnhoa/IAAIR/data/primekg") -> bool:
    """
    Download PrimeKG data from Harvard Dataverse
    
    Args:
        data_path: Path to store data
        
    Returns:
        True if successful
    """
    integrator = PrimeKGIntegrator(data_path)
    return integrator.download_data()


def load_primekg_for_knowledge_fabric(
    data_path: str = "/home/dnhoa/IAAIR/data/primekg"
) -> Optional[PrimeKGIntegrator]:
    """
    Load PrimeKG data for Knowledge Fabric integration
    
    Args:
        data_path: Path to PrimeKG data
        
    Returns:
        Configured PrimeKG integrator or None if failed
    """
    integrator = PrimeKGIntegrator(data_path)
    
    if integrator.load_data():
        logger.info("PrimeKG successfully loaded for Knowledge Fabric")
        return integrator
    else:
        logger.error("Failed to load PrimeKG data")
        return None


def analyze_primekg_coverage(
    integrator: PrimeKGIntegrator,
    documents: List[Document]
) -> Dict[str, Any]:
    """
    Analyze how well PrimeKG covers the entities mentioned in documents
    
    Args:
        integrator: PrimeKG integrator
        documents: Documents to analyze
        
    Returns:
        Coverage analysis results
    """
    total_docs = len(documents)
    docs_with_diseases = 0
    docs_with_drugs = 0
    docs_with_proteins = 0
    
    all_diseases = set()
    all_drugs = set()
    all_proteins = set()
    
    for doc in documents:
        search_text = f"{doc.title} {doc.abstract or ''}"
        
        diseases = integrator.search_diseases(search_text, max_results=3)
        drugs = integrator.search_drugs(search_text, max_results=3)
        proteins = integrator.search_proteins(search_text, max_results=3)
        
        if diseases:
            docs_with_diseases += 1
            all_diseases.update(d.get('node_id', d.get('id', '')) for d in diseases)
        
        if drugs:
            docs_with_drugs += 1
            all_drugs.update(d.get('node_id', d.get('id', '')) for d in drugs)
        
        if proteins:
            docs_with_proteins += 1
            all_proteins.update(p.get('node_id', p.get('id', '')) for p in proteins)
    
    return {
        'total_documents': total_docs,
        'disease_coverage': {
            'documents_with_diseases': docs_with_diseases,
            'percentage': (docs_with_diseases / total_docs * 100) if total_docs > 0 else 0,
            'unique_diseases_found': len(all_diseases)
        },
        'drug_coverage': {
            'documents_with_drugs': docs_with_drugs,
            'percentage': (docs_with_drugs / total_docs * 100) if total_docs > 0 else 0,
            'unique_drugs_found': len(all_drugs)
        },
        'protein_coverage': {
            'documents_with_proteins': docs_with_proteins,
            'percentage': (docs_with_proteins / total_docs * 100) if total_docs > 0 else 0,
            'unique_proteins_found': len(all_proteins)
        }
    }