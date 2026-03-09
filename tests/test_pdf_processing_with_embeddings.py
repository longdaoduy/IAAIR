#!/usr/bin/env python3
"""
Test script for PDF processing with description embeddings and Milvus integration.

This script demonstrates:
1. Creating a mock paper with PDF URL
2. Processing the PDF to extract figures and tables
3. Generating SciBERT embeddings for descriptions
4. Saving both image and description embeddings to Milvus
"""

import os
import sys
from datetime import datetime
from models.schemas.nodes import Paper
from pipelines.ingestions.PDFProcessingHandler import PDFProcessingHandler
from clients.huggingface.CLIPClient import CLIPClient
from clients.huggingface.SciBERTClient import SciBERTClient
from clients.vector.MilvusClient import MilvusClient

def test_pdf_processing_with_embeddings():
    """Test the enhanced PDF processing pipeline."""
    
    print("🧪 Testing PDF Processing with Description Embeddings and Milvus Integration")
    print("=" * 80)
    
    # Initialize clients
    print("📡 Initializing clients...")
    clip_client = CLIPClient()
    scibert_client = SciBERTClient()  
    milvus_client = MilvusClient()
    
    # Initialize PDF handler with all clients
    pdf_handler = PDFProcessingHandler(
        clip_client=clip_client,
        scibert_client=scibert_client,
        milvus_client=milvus_client
    )
    
    print("✅ Clients initialized successfully")
    
    # Create a mock paper (you can replace this with a real paper with PDF URL)
    test_paper = Paper(
        id="test_paper_001",
        title="Test Paper for PDF Processing",
        abstract="This is a test paper to demonstrate PDF processing with embeddings.",
        doi="10.1000/test.001",
        publication_date=datetime.now(),
        source="test",
        metadata={},
        # Replace with an actual PDF URL for testing
        pdf_url="https://example.com/test_paper.pdf"  # This should be replaced with a real PDF URL
    )
    
    print(f"📄 Created test paper: {test_paper.id}")
    
    # Process the PDF (this will download, extract figures/tables, generate embeddings, and save to Milvus)
    try:
        print("🔄 Processing PDF...")
        figures, tables = pdf_handler.process_paper_pdf(test_paper)
        
        print(f"\n📊 Processing Results:")
        print(f"   • Figures extracted: {len(figures)}")
        print(f"   • Tables extracted: {len(tables)}")
        
        # Display details about extracted content
        if figures:
            print(f"\n🖼️ Figure Details:")
            for i, figure in enumerate(figures, 1):
                print(f"   {i}. Figure {figure.figure_number} (Page {figure.page_number})")
                print(f"      Description: {figure.description[:100] + '...' if figure.description and len(figure.description) > 100 else figure.description}")
                print(f"      Has image embedding: {figure.image_embedding is not None}")
                print(f"      Has description embedding: {figure.description_embedding is not None}")
                print()
        
        if tables:
            print(f"📋 Table Details:")
            for i, table in enumerate(tables, 1):
                print(f"   {i}. Table {table.table_number} (Page {table.page_number})")
                print(f"      Description: {table.description[:100] + '...' if table.description and len(table.description) > 100 else table.description}")
                print(f"      Has description embedding: {table.description_embedding is not None}")
                print(f"      Headers: {table.headers}")
                print()
        
        print("✅ PDF processing completed successfully!")
        
        # Summary of Milvus uploads
        figures_with_embeddings = sum(1 for f in figures if f.description_embedding or f.image_embedding)
        tables_with_embeddings = sum(1 for t in tables if t.description_embedding)
        
        print(f"\n📤 Milvus Upload Summary:")
        print(f"   • Figures uploaded: {figures_with_embeddings}")
        print(f"   • Tables uploaded: {tables_with_embeddings}")
        print(f"   • Total items in vector database: {figures_with_embeddings + tables_with_embeddings}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during PDF processing: {e}")
        return False

def test_embedding_generation():
    """Test SciBERT embedding generation separately."""
    
    print("\n🧪 Testing SciBERT Embedding Generation")
    print("=" * 50)
    
    try:
        # Initialize SciBERT client
        scibert_client = SciBERTClient()
        
        # Test description texts
        test_descriptions = [
            "Figure 1 shows the experimental setup with temperature measurements over time.",
            "Table 2 presents the statistical analysis results for different treatment groups.",
            "This diagram illustrates the neural network architecture used in our model."
        ]
        
        for i, desc in enumerate(test_descriptions, 1):
            print(f"\nGenerating embedding for description {i}:")
            print(f"Text: {desc}")
            
            # Generate embedding
            embedding = scibert_client.generate_text_embedding(desc)
            
            print(f"Embedding dimension: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")
        
        print("\n✅ SciBERT embedding generation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during embedding generation test: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting PDF Processing with Embeddings Test Suite")
    print("=" * 80)
    
    # Test embedding generation first (doesn't require PDF)
    embedding_success = test_embedding_generation()
    
    if embedding_success:
        print("\n" + "=" * 80)
        # Test full PDF processing (requires valid PDF URL)
        # Note: This will only work if you provide a real PDF URL
        pdf_success = test_pdf_processing_with_embeddings()
        
        if pdf_success:
            print("\n🎉 All tests passed successfully!")
        else:
            print("\n⚠️ PDF processing test failed (likely due to missing/invalid PDF URL)")
    else:
        print("\n❌ Embedding generation test failed")
    
    print("\n" + "=" * 80)
    print("ℹ️ To test with actual PDFs:")
    print("  1. Replace the pdf_url in test_paper with a real PDF URL")
    print("  2. Ensure Milvus/Zilliz connection is configured")
    print("  3. Run this script again")