"""
Main entry point for the Knowledge Fabric system.

Provides CLI interface for common operations like ingestions,
search, and system management.
"""

import asyncio
import click
import logging
from typing import Optional

from src.knowledge_fabric import KnowledgeFabric
from src.config import settings
from ..knowledge_fabric.schemas import QueryPlan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--configurators', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(config: Optional[str], verbose: bool):
    """Knowledge Fabric - Multi-modal Scientific Literature Retrieval System"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option('--source', type=click.Choice(['openalex', 'semantic-scholar']), required=True)
@click.option('--query', required=True, help='Search query for papers')
@click.option('--max-docs', default=1000, help='Maximum documents to ingest')
async def ingest(source: str, query: str, max_docs: int):
    """Ingest documents from external sources."""
    fabric = KnowledgeFabric(settings)
    
    try:
        if source == 'openalex':
            results = await fabric.ingestion_pipeline.ingest_from_openalex(query, max_docs)
        else:
            results = await fabric.ingestion_pipeline.ingest_from_semantic_scholar(query, max_docs)
        
        click.echo(f"Ingestion completed: {results}")
        
    except Exception as e:
        click.echo(f"Ingestion failed: {e}", err=True)
    finally:
        await fabric.close()


@cli.command()
@click.argument('query')
@click.option('--max-results', default=10, help='Maximum search results')
@click.option('--vector_store-weight', default=0.6, help='Vector search weight')
@click.option('--graph_store-weight', default=0.4, help='Graph search weight')
async def search(query: str, max_results: int, vector_weight: float, graph_weight: float):
    """Search the knowledge fabric."""
    fabric = KnowledgeFabric(settings)
    
    try:
        query_plan = QueryPlan(
            query=query,
            vector_weight=vector_weight,
            graph_weight=graph_weight,
            max_results=max_results
        )
        
        results = await fabric.search(query, max_results, query_plan)
        
        click.echo(f"Found {len(results)} results for: {query}\n")
        for i, result in enumerate(results, 1):
            click.echo(f"{i}. {result.document.title}")
            click.echo(f"   Score: {result.score:.3f} | Method: {result.retrieval_method}")
            click.echo(f"   Authors: {', '.join(author.name for author in result.document.authors)}")
            click.echo(f"   DOI: {result.document.doi or 'N/A'}\n")
            
    except Exception as e:
        click.echo(f"Search failed: {e}", err=True)
    finally:
        await fabric.close()


@cli.command()
@click.argument('query')
@click.option('--max-sources', default=5, help='Maximum source documents')
async def evidence(query: str, max_sources: int):
    """Generate evidence bundle for a query."""
    fabric = KnowledgeFabric(settings)
    
    try:
        bundle = await fabric.generate_evidence_bundle(query, max_sources)
        
        click.echo(f"Evidence Bundle for: {query}\n")
        click.echo(f"Overall Confidence: {bundle.confidence:.3f}")
        click.echo(f"Sources: {len(bundle.sources)}")
        click.echo(f"Attribution Spans: {len(bundle.attributions)}")
        
        click.echo("\nSources:")
        for i, source in enumerate(bundle.sources, 1):
            click.echo(f"{i}. {source.title}")
            click.echo(f"   {', '.join(author.name for author in source.authors)}")
            
    except Exception as e:
        click.echo(f"Evidence generation failed: {e}", err=True)
    finally:
        await fabric.close()


@cli.command()
async def health():
    """Check system health."""
    fabric = KnowledgeFabric(settings)
    
    try:
        health_status = await fabric.health_check()
        
        click.echo("System Health Check:")
        for component, status in health_status.items():
            status_icon = "✓" if status else "✗"
            click.echo(f"  {status_icon} {component}: {'OK' if status else 'FAILED'}")
            
    except Exception as e:
        click.echo(f"Health check failed: {e}", err=True)
    finally:
        await fabric.close()


@cli.command()
@click.option('--host', default='0.0.0.0', help='API host')
@click.option('--port', default=8000, help='API port')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host: str, port: int, reload: bool):
    """Start the API server."""
    import uvicorn
    uvicorn.run(
        "src.pipelines.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


def main():
    """Main entry point that handles async commands."""
    import sys
    
    # Check if command is async
    async_commands = ['ingest', 'search', 'evidence', 'health']
    
    if len(sys.argv) > 1 and sys.argv[1] in async_commands:
        # Run async command
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Patch click to support async
        original_invoke = click.Command.invoke
        
        def async_invoke(self, ctx):
            if asyncio.iscoroutinefunction(self.callback):
                return loop.run_until_complete(self.callback(**ctx.params))
            return original_invoke(self, ctx)
        
        click.Command.invoke = async_invoke
    
    cli()


if __name__ == "__main__":
    main()