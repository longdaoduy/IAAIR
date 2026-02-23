#!/usr/bin/env python3
"""
Quick script to load and explore the SciMMIR Parquet data file.
"""

import pandas as pd
import json
from pathlib import Path

def load_scimmir_parquet(parquet_path: str):
    """Load SciMMIR data from Parquet file."""
    try:
        # Load the parquet file
        df = pd.read_parquet(parquet_path)
        
        print(f"Loaded Parquet file: {parquet_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nData types:")
        print(df.dtypes)
        
        print("\nSample text content:")
        if 'text' in df.columns:
            print(df['text'].iloc[0][:200] + "...")
        
        print(f"\nUnique classes:")
        if 'class' in df.columns:
            print(df['class'].value_counts())
        
        return df
        
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return None

def convert_to_scimmir_samples(df):
    """Convert DataFrame to SciMMIR sample format."""
    samples = []
    
    for idx, row in df.iterrows():
        sample = {
            'sample_id': f"parquet_{idx:06d}",
            'text': row.get('text', ''),
            'image': row.get('image', None),  # May be binary or path
            'class_label': row.get('class', 'figure'),
            'domain': 'general'  # You can infer this from text
        }
        samples.append(sample)
    
    return samples

if __name__ == "__main__":
    parquet_path = "/data/scimmir_cache/scimmir_dataset/test-00000-of-00004-758f4fffbab26e7d.parquet"
    
    # Load and explore the data
    df = load_scimmir_parquet(parquet_path)
    
    if df is not None:
        print(f"\n🎉 Successfully loaded {len(df)} samples from Parquet file!")
        
        # Convert to your format
        samples = convert_to_scimmir_samples(df)
        print(f"Converted to {len(samples)} SciMMIR samples")
        
        # Save a preview
        preview_path = "/data/scimmir_preview.json"
        with open(preview_path, 'w') as f:
            json.dump(samples[:5], f, indent=2, default=str)  # Save first 5 samples
        print(f"Preview saved to: {preview_path}")