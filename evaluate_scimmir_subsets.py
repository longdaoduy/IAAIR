#!/usr/bin/env python3
"""
SciMMIR Subset Evaluation Demo

This script demonstrates how to evaluate your model on SciMMIR subsets 
following the CLIP-BERT methodology with hierarchical subset breakdowns.

Based on the SciMMIR benchmark methodology:
• Figure Subset (11,491 total test samples):
    ◦ Figure Result: 9,488 samples
    ◦ Figure Illustration: 1,536 samples
    ◦ Figure Architecture: 467 samples
• Table Subset (4,772 total test samples):
    ◦ Table Result: 4,229 samples
    ◦ Table Parameter: 543 samples
"""

import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pipelines.evaluation.SciMMIRBenchmarkIntegration import (
    SciMMIRDataLoader,
    SciMMIRBenchmarkRunner,
    SciMMIRResultAnalyzer
)
from clients.huggingface.CLIPClient import CLIPClient
from clients.huggingface.SciBERTClient import SciBERTClient
from clients.vector.MilvusClient import MilvusClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run SciMMIR subset evaluation like CLIP-BERT."""
    
    logger.info("🚀 Starting SciMMIR Subset Evaluation (CLIP-BERT Methodology)")
    
    try:
        # Initialize data loader
        logger.info("📁 Initializing SciMMIR data loader...")
        data_loader = SciMMIRDataLoader(cache_dir="./data/scimmir_cache")
        
        # Load test samples (limit for demo purposes)
        logger.info("📊 Loading SciMMIR test samples...")
        # For full evaluation, remove the limit parameter
        samples = data_loader.load_test_samples(limit=1000)  # Remove limit for full evaluation
        
        if not samples:
            logger.error("❌ No samples loaded. Please check data availability.")
            return
            
        logger.info(f"✅ Loaded {len(samples)} samples for evaluation")
        
        # Analyze subset distribution
        subset_distribution = {}
        for sample in samples:
            subset_category = data_loader.get_subset_category(sample.class_label, sample.text)
            subset_distribution[subset_category] = subset_distribution.get(subset_category, 0) + 1
        
        logger.info("📋 Subset distribution in loaded samples:")
        for subset, count in sorted(subset_distribution.items()):
            logger.info(f"  {subset}: {count} samples")
        
        # Initialize model clients
        logger.info("🤖 Initializing model clients...")
        # Note: Make sure these clients are properly configured in your environment
        try:
            clip_client = CLIPClient()
            scibert_client = SciBERTClient()
            vector_client = MilvusClient()
        except Exception as e:
            logger.error(f"❌ Failed to initialize clients: {e}")
            logger.info("💡 Please ensure your model clients are properly configured")
            return
        
        # Initialize benchmark runner
        logger.info("⚡ Initializing SciMMIR benchmark runner...")
        benchmark_runner = SciMMIRBenchmarkRunner(
            clip_client=clip_client,
            scibert_client=scibert_client,
            vector_client=vector_client
        )
        
        # Run benchmark with subset evaluation and memory optimization
        logger.info("🏃 Running SciMMIR benchmark with subset evaluation and memory optimization...")
        logger.info("💾 Using automatic batch size optimization to prevent OOM errors...")
        
        result = benchmark_runner.run_benchmark(
            samples=samples,
            model_name="IAAIR-Hybrid-CLIP-BERT-Style",
            top_k=10,
            evaluate_subsets=True,  # Enable subset evaluation
            batch_size=None  # Auto-optimize batch size based on available memory
        )
        
        # Analyze results
        logger.info("📈 Analyzing results...")
        analyzer = SciMMIRResultAnalyzer()
        
        # Generate comprehensive report
        logger.info("📝 Generating comprehensive evaluation report...")
        report_path = "data/scimmir_subset_evaluation_report.md"
        report = analyzer.generate_report(result, save_path=report_path)
        
        # Print key results
        print("\n" + "="*80)
        print("🎯 SciMMIR SUBSET EVALUATION RESULTS (CLIP-BERT Methodology)")
        print("="*80)
        
        print(f"\n📊 Overall Performance:")
        print(f"   Text-to-Image MRR: {result.text2img_mrr:.4f} ({result.text2img_mrr*100:.2f}%)")
        print(f"   Image-to-Text MRR: {result.img2text_mrr:.4f} ({result.img2text_mrr*100:.2f}%)")
        
        if result.subset_results:
            print(f"\n📋 Subset Performance:")
            
            # Figure subsets
            figure_subsets = {k: v for k, v in result.subset_results.items() if k.startswith('figure_')}
            if figure_subsets:
                total_figure = sum(v['sample_count'] for v in figure_subsets.values())
                print(f"\n   🖼️ Figure Subset ({total_figure:,} samples):")
                for subset, metrics in figure_subsets.items():
                    name = subset.replace('_', ' ').title()
                    print(f"     {name}: {metrics['sample_count']:,} samples")
                    print(f"       T2I MRR: {metrics['text2img_mrr']:.4f} ({metrics['text2img_mrr']*100:.2f}%)")
                    print(f"       I2T MRR: {metrics['img2text_mrr']:.4f} ({metrics['img2text_mrr']*100:.2f}%)")
            
            # Table subsets
            table_subsets = {k: v for k, v in result.subset_results.items() if k.startswith('table_')}
            if table_subsets:
                total_table = sum(v['sample_count'] for v in table_subsets.values())
                print(f"\n   📊 Table Subset ({total_table:,} samples):")
                for subset, metrics in table_subsets.items():
                    name = subset.replace('_', ' ').title()
                    print(f"     {name}: {metrics['sample_count']:,} samples")
                    print(f"       T2I MRR: {metrics['text2img_mrr']:.4f} ({metrics['text2img_mrr']*100:.2f}%)")
                    print(f"       I2T MRR: {metrics['img2text_mrr']:.4f} ({metrics['img2text_mrr']*100:.2f}%)")
        
        print(f"\n📄 Full report saved to: {report_path}")
        
        # Compare with CLIP-BERT expected distribution
        print(f"\n📋 Expected CLIP-BERT Distribution vs Actual:")
        expected = {
            'figure_result': 9488,
            'figure_illustration': 1536,
            'figure_architecture': 467,
            'table_result': 4229,
            'table_parameter': 543
        }
        
        for subset, expected_count in expected.items():
            actual_count = result.subset_results.get(subset, {}).get('sample_count', 0) if result.subset_results else 0
            print(f"   {subset.replace('_', ' ').title()}: {actual_count:,} actual vs {expected_count:,} expected")
        
        print("\n✅ SciMMIR subset evaluation completed successfully!")
        print(f"🎉 Your model evaluated on {result.total_samples:,} samples with subset breakdowns")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()