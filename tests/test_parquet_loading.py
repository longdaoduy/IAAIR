#!/usr/bin/env python3
"""
Test script to load SciMMIR data from your Parquet file using the updated integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.evaluation.SciMMIRBenchmarkIntegration import SciMMIRDataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_parquet_loading():
    """Test loading SciMMIR data from Parquet file."""
    
    # Initialize the data loader
    loader = SciMMIRDataLoader()
    
    # Path to your Parquet file
    parquet_path = "/data/scimmir_cache/scimmir_dataset/test-00000-of-00004-758f4fffbab26e7d.parquet"
    
    print("🚀 Testing Parquet file loading...")
    
    # Load a small sample first (10 samples for quick testing)
    samples = loader.load_test_samples(
        limit=10,
    )
    
    print(f"✅ Loaded {len(samples)} samples from Parquet file")
    
    # Display sample information
    for i, sample in enumerate(samples[:3]):
        print(f"\n📄 Sample {i+1}:")
        print(f"   ID: {sample.sample_id}")
        print(f"   Text: {sample.text[:100]}...")
        print(f"   Class: {sample.class_label}")
        print(f"   Domain: {sample.domain}")
        print(f"   Has Image: {'Yes' if sample.image else 'No'}")
        if sample.image:
            print(f"   Image Size: {sample.image.size}")
    
    print(f"\n📊 Class distribution:")
    class_counts = {}
    for sample in samples:
        class_counts[sample.class_label] = class_counts.get(sample.class_label, 0) + 1
    
    for class_name, count in sorted(class_counts.items()):
        print(f"   {class_name}: {count}")
    
    return samples

def test_larger_batch():
    """Test loading a larger batch (100 samples)."""
    
    loader = SciMMIRDataLoader()
    parquet_path = "/data/scimmir_cache/scimmir_dataset/test-00000-of-00004-758f4fffbab26e7d.parquet"
    
    print(f"\n🎯 Loading larger batch (100 samples)...")
    
    samples = loader.load_test_samples(
        limit=100,
    )
    
    print(f"✅ Loaded {len(samples)} samples from Parquet file")
    
    # Statistics
    total_with_images = sum(1 for s in samples if s.image)
    print(f"📷 Samples with images: {total_with_images}/{len(samples)} ({total_with_images/len(samples)*100:.1f}%)")
    
    # Class distribution
    class_counts = {}
    for sample in samples:
        class_counts[sample.class_label] = class_counts.get(sample.class_label, 0) + 1
    
    print(f"\n📊 Class distribution (100 samples):")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {class_name}: {count} ({count/len(samples)*100:.1f}%)")
    
    return samples

if __name__ == "__main__":
    print("🔬 SciMMIR Parquet Loading Test\n" + "="*50)
    
    try:
        # Test small batch
        small_samples = test_parquet_loading()
        
        # Test larger batch
        large_samples = test_larger_batch()
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"💡 You can now use your Parquet file directly in the SciMMIR benchmark!")
        
        # Show how to use it in the main benchmark
        print(f"\n🛠️  To use in your benchmark, call:")
        print(f"   samples = loader.load_test_samples(parquet_path='your_parquet_file.parquet')")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()