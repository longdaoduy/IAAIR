from typing import Dict, List, Optional
from datetime import datetime
import logging
import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

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
from models.entities.ingestion.PaperRequest import PaperRequest
from models.entities.ingestion.PaperResponse import PaperResponse
from models.entities.retrieval.HybridSearchRequest import HybridSearchRequest
from models.entities.retrieval.GraphQueryRequest import GraphQueryRequest
from models.entities.retrieval.GraphQueryResponse import GraphQueryResponse
from models.entities.retrieval.SearchRequest import SearchRequest
from models.entities.retrieval.SearchResponse import SearchResponse
from models.entities.retrieval.HybridSearchResponse import HybridSearchResponse
from models.entities.retrieval.RoutingStrategy import RoutingStrategy
from models.entities.retrieval.RoutingPerformanceResponse import RoutingPerformanceResponse
from models.entities.retrieval.QueryAnalysisResponse import QueryAnalysisResponse
from models.entities.retrieval.QueryAnalysisRequest import QueryAnalysisRequest
from models.entities.retrieval.QueryType import QueryType
from models.entities.retrieval.AttributionStatsResponse import AttributionStatsResponse

logger = logging.getLogger(__name__)


class ServiceFactory:
    def __init__(self):
        # Clients
        self.neo4j_handler = GraphNeo4jHandler()
        self.vector_handler = MilvusClient()
        self.scibert_client = SciBERTClient()
        self.deepseek_client = DeepseekClient()

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
