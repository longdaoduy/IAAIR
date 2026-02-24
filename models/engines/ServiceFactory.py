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
from pipelines.evaluation.SciMMIRBenchmarkIntegration import (
    SciMMIRBenchmarkResult,
    SciMMIRResultAnalyzer,
    SciMMIRDataLoader,
    SciMMIRBenchmarkRunner
)

logger = logging.getLogger(__name__)


class ServiceFactory:
    def __init__(self):
        # Clients
        self.neo4j_handler = GraphNeo4jHandler()
        self.vector_handler = MilvusClient()
        self.scibert_client = SciBERTClient()
        self.clip_client = None
        self.deepseek_client = DeepseekClient()

        # Pipelines & Engines
        self.query_handler = GraphQueryHandler()
        self.routing_engine = RoutingDecisionEngine(self.deepseek_client)
        self.result_fusion = ResultFusion()
        self.scientific_reranker = ScientificReranker()
        self.attribution_tracker = AttributionTracker()

        # Complex Handlers
        self.ingestion_handler = IngestionHandler(self.scibert_client, self.clip_client)
        self.embedding_handler = EmbeddingSciBERTHandler(self.scibert_client)
        self.retrieval_handler = HybridRetrievalHandler(
            self.vector_handler,
            self.query_handler,
            self.deepseek_client,
            self.scibert_client
        )

    def run_scimmir_benchmark_suite(
            self,
            limit_samples: int = 50,
            cache_dir: str = "./data/scimmir_cache",
            report_path: str = "./data/scimmir_benchmark_report.md",
    ) -> SciMMIRBenchmarkResult:
        """
        Complete SciMMIR benchmark evaluation workflow with memory management.

        Args:
            limit_samples: Number of test samples to evaluate (default: 50 for memory efficiency)
            cache_dir: Directory to cache SciMMIR dataset (ignored if use_streaming=True)
            report_path: Path to save evaluation report
            use_streaming: Use streaming mode to avoid downloading entire dataset
            use_mock: Use mock samples for quick testing (no download required)
            memory_efficient: Use memory-efficient loading (default: True, caps at 100 samples)

        Returns:
            Benchmark results with comparison to baselines
        """
        # Load data with memory management
        data_loader = SciMMIRDataLoader(cache_dir)
        samples = data_loader.load_test_samples(
            limit=None
        )

        if not samples:
            raise ValueError("Failed to load SciMMIR samples")

        # Run benchmark
        benchmark_runner = SciMMIRBenchmarkRunner(self.clip_client, self.scibert_client, self.vector_handler)
        result = benchmark_runner.run_benchmark(samples, model_name="IAAIR-SciBERT-CLIP")

        # Generate report
        analyzer = SciMMIRResultAnalyzer()
        report = analyzer.generate_report(result, save_path=report_path)

        print("=" * 80)
        print("🎯 SciMMIR Benchmark Completed!")
        print("=" * 80)
        print(report)

        return result

    async def connect_all(self):
        """Standardize startup for all database clients"""
        self.vector_handler.connect()
        # self.clip_client.initialize()
        # await self.neo4j_handler.connect() if async

    async def disconnect_all(self):
        """Cleanup connections on shutdown"""
        self.vector_handler.disconnect()
