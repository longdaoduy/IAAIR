#!/usr/bin/env python3
"""
SciMMIR Subset Evaluation Configuration

This script helps configure and test the SciMMIR subset evaluation system
before running the full evaluation.
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from pipelines.evaluation.SciMMIRBenchmarkIntegration import SciMMIRDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_scimmir_data():
    """Analyze SciMMIR data structure and subset distribution."""
    
    logger.info("🔍 Analyzing SciMMIR data structure...")
    
    data_loader = SciMMIRDataLoader(cache_dir="./data/scimmir_cache")
    
    # Check if parquet files exist
    dataset_dir = Path("./data/scimmir_cache/scimmir_dataset")
    parquet_files = list(dataset_dir.glob("*.parquet")) if dataset_dir.exists() else []
    
    if not parquet_files:
        logger.info("📥 No parquet files found. Attempting to download...")
        success = data_loader.download_parquet_files()
        if not success:
            logger.error("❌ Failed to download SciMMIR data")
            return
        parquet_files = list(dataset_dir.glob("*.parquet"))
    
    logger.info(f"✅ Found {len(parquet_files)} parquet files")
    
    # Analyze first file to understand structure
    if parquet_files:
        first_file = parquet_files[0]
        logger.info(f"📊 Analyzing structure of {first_file.name}...")
        
        try:
            df = pd.read_parquet(first_file)
            logger.info(f"   Shape: {df.shape}")
            logger.info(f"   Columns: {list(df.columns)}")
            
            # Check class distribution
            if 'class' in df.columns:
                class_counts = df['class'].value_counts()
                logger.info(f"   Class distribution in {first_file.name}:")
                for class_name, count in class_counts.head(10).items():
                    logger.info(f"     {class_name}: {count}")
                    
                # Test subset mapping
                logger.info(f"   Testing subset mapping:")
                subset_counts = {}
                for class_label in class_counts.index[:10]:  # Test first 10 classes
                    subset = data_loader.get_subset_category(class_label, "")
                    subset_counts[subset] = subset_counts.get(subset, 0) + class_counts[class_label]
                
                for subset, count in sorted(subset_counts.items()):
                    logger.info(f"     {subset}: {count} samples")
            
            # Check text data
            text_columns = [col for col in df.columns if any(term in col.lower() for term in ['text', 'caption', 'description'])]
            if text_columns:
                logger.info(f"   Text columns: {text_columns}")
                for col in text_columns[:2]:  # Show first 2 text columns
                    sample_text = df[col].dropna().iloc[0] if not df[col].dropna().empty else "No text found"
                    logger.info(f"     {col} sample: {sample_text[:100]}...")
            
            # Check image data
            if 'image' in df.columns:
                logger.info(f"   Image column found with type: {type(df['image'].iloc[0])}")
                
        except Exception as e:
            logger.error(f"❌ Failed to analyze {first_file.name}: {e}")

def test_subset_mapping():
    """Test the subset mapping logic."""
    
    logger.info("🧪 Testing subset mapping logic...")
    
    data_loader = SciMMIRDataLoader()
    
    # Test cases based on expected SciMMIR categories
    test_cases = [
        # Figure categories
        ('fig_result', 'Performance comparison of different models'),
        ('fig_chart', 'Bar chart showing accuracy results'),
        ('fig_plot', 'Line plot of training progress'),
        ('fig_diagram', 'System architecture diagram'),
        ('fig_architecture', 'Neural network architecture'),
        ('fig_flowchart', 'Algorithm workflow diagram'),
        
        # Table categories
        ('tab_result', 'Comparison table of model performance'),
        ('tab_parameter', 'Hyperparameter configuration table'),
        ('tab_config', 'Experimental setup parameters'),
        
        # Edge cases
        ('unknown_fig', 'Some figure description'),
        ('unknown_tab', 'Some table description'),
    ]
    
    logger.info("   Subset mapping results:")
    for class_label, text in test_cases:
        subset = data_loader.get_subset_category(class_label, text)
        logger.info(f"     {class_label} + '{text[:30]}...' → {subset}")

def estimate_evaluation_time():
    """Estimate evaluation time based on sample count."""
    
    logger.info("⏱️  Estimating evaluation time...")
    
    # Check available samples
    data_loader = SciMMIRDataLoader(cache_dir="./data/scimmir_cache")
    
    # Load a small sample to test
    try:
        samples = data_loader.load_test_samples(limit=100)
        if samples:
            logger.info(f"   Successfully loaded {len(samples)} test samples")
            
            # Estimate for full dataset
            full_dataset_size = 16263  # Based on CLIP-BERT evaluation
            time_per_sample = 0.5  # seconds (rough estimate)
            
            estimated_time_minutes = (full_dataset_size * time_per_sample) / 60
            logger.info(f"   Estimated time for full evaluation: {estimated_time_minutes:.1f} minutes")
            logger.info(f"   Recommended: Start with 1000 samples for testing")
        else:
            logger.warning("   Could not load test samples")
    except Exception as e:
        logger.error(f"   Failed to load samples: {e}")

def main():
    """Run configuration analysis."""
    
    print("🛠️  SciMMIR Subset Evaluation Configuration")
    print("=" * 50)
    
    # Run analysis steps
    analyze_scimmir_data()
    print()
    test_subset_mapping()
    print()
    estimate_evaluation_time()
    
    print("\n✅ Configuration analysis complete!")
    print("\n📝 Next steps:")
    print("   1. Run: python evaluate_scimmir_subsets.py")
    print("   2. Check the generated report in data/scimmir_subset_evaluation_report.md")
    print("   3. Compare results with CLIP-BERT baselines")

if __name__ == "__main__":
    main()