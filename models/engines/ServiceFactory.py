import logging

# Import handlers
from pipelines.ingestions.IngestionHandler import IngestionHandler
from pipelines.ingestions.GraphNeo4jHandler import GraphNeo4jHandler
from clients.vector.MilvusClient import MilvusClient
from clients.huggingface.SciBERTClient import SciBERTClient
from clients.huggingface.DeepseekClient import DeepseekClient
from clients.huggingface.CLIPClient import CLIPClient
from pipelines.ingestions.EmbeddingSciBERTHandler import EmbeddingSciBERTHandler
from pipelines.retrievals.HybridRetrievalHandler import HybridRetrievalHandler
from pipelines.retrievals.GraphQueryHandler import GraphQueryHandler
from models.engines.RoutingDecisionEngine import RoutingDecisionEngine
from models.engines.ResultFusion import ResultFusion
from models.engines.ScientificReranker import ScientificReranker
from models.engines.AttributionTracker import AttributionTracker

logger = logging.getLogger(__name__)


class ServiceFactory:
    def __init__(self):
        # Clients
        self.neo4j_handler = GraphNeo4jHandler()
        self.vector_handler = MilvusClient()
        self.scibert_client = SciBERTClient()
        self.deepseek_client = None

        # Pipelines & Engines
        self.query_handler = GraphQueryHandler()
        self.routing_engine = RoutingDecisionEngine()
        self.result_fusion = ResultFusion()
        self.scientific_reranker = ScientificReranker()
        self.attribution_tracker = AttributionTracker()

        # Complex Handlers
        self.ingestion_handler = IngestionHandler(self.scibert_client)
        self.embedding_handler = EmbeddingSciBERTHandler(self.scibert_client)
        self.retrieval_handler = HybridRetrievalHandler(
            self.vector_handler,
            self.query_handler,
            self.deepseek_client,
            self.scibert_client
        )

    async def connect_all(self):
        """Standardize startup for all database clients"""
        self.vector_handler.connect()
        # await self.neo4j_handler.connect() if async

    async def disconnect_all(self):
        """Cleanup connections on shutdown"""
        self.vector_handler.disconnect()
