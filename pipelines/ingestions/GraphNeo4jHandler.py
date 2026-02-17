import asyncio
import logging
from typing import Dict, List, Optional
from clients.graph.Neo4jClient import Neo4jClient

logger = logging.getLogger(__name__)


class GraphNeo4jHandler:
    """Handler for Neo4j database operations with concurrent task management."""

    def __init__(self, batch_size: int = 20):
        self.neo4j_client: Optional[Neo4jClient] = None
        self.batch_size = batch_size

    async def _get_client(self) -> Neo4jClient:
        """Lazy initialization of the client."""
        if not self.neo4j_client:
            self.neo4j_client = Neo4jClient()
            await self.neo4j_client.connect()
        return self.neo4j_client

    async def upload_papers_to_neo4j(self, papers_data: List[Dict]) -> bool:
        """
        Upload papers data using concurrent task batching.
        """
        if not papers_data:
            logger.warning("No papers data provided for upload.")
            return False

        client = await self._get_client()
        dataset_ids = {pd["paper"].id for pd in papers_data}  # O(1) lookup

        # Use a semaphore to limit concurrency (prevents overwhelming the DB pool)
        semaphore = asyncio.Semaphore(self.batch_size)

        async def upload_task(paper_item):
            async with semaphore:
                try:
                    paper = paper_item["paper"]
                    # Filter citations only if they exist in our current processing set
                    citations = [cid for cid in paper_item.get("citations", [])
                                 if cid in dataset_ids]

                    await client.store_paper_with_content(
                        paper=paper,
                        authors=paper_item.get("authors", []),
                        venue=paper_item.get("venue"),
                        citations=citations,
                        institutions=paper_item.get("institutions", []),
                        figures=paper_item.get("figures", []),
                        tables=paper_item.get("tables", [])
                    )
                    return True
                except Exception as e:
                    logger.error(f"Failed task for paper {paper.id}: {e}")
                    return False

        # Create concurrent tasks
        tasks = [upload_task(pd) for pd in papers_data]
        results = await asyncio.gather(*tasks)

        success_count = sum(1 for r in results if r)
        logger.info(f"Upload complete. Success: {success_count}/{len(papers_data)}")

        return success_count > 0

    async def close(self):
        if self.neo4j_client:
            await self.neo4j_client.close()
            self.neo4j_client = None