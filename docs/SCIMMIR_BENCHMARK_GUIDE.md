# SciMMIR Benchmark Integration Guide

This guide shows how to benchmark your IAAIR Scientific RAG system using the SciMMIR multi-modal information retrieval benchmark.

## 🎯 What is SciMMIR?

SciMMIR is a benchmark for evaluating **Scientific Multi-modal Information Retrieval** systems. It tests how well your system can:

- **Text-to-Image Retrieval**: Find relevant scientific figures based on text queries
- **Image-to-Text Retrieval**: Find relevant text descriptions for scientific figures  
- **Cross-Modal Understanding**: Bridge the gap between scientific text and visual content

The benchmark uses 537K scientific image-text pairs from ArXiv papers (May-Oct 2023) across multiple domains and figure categories.

## 🏗️ Integration Architecture

```
Your IAAIR System                  SciMMIR Benchmark
├── SciBERT (text embeddings)  ←→  Text queries & descriptions
├── CLIP (image embeddings)    ←→  Scientific figures  
├── Milvus (vector search)     ←→  Similarity retrieval
└── Evaluation metrics         ←→  MRR, Recall@k comparison
```

## 🚀 Quick Start

### 1. Install Dependencies

Your system already includes the required packages:
- `datasets` - For loading SciMMIR data from HuggingFace
- `torch` & `transformers` - For model inference
- `Pillow` - For image processing
- `numpy` - For numerical computations

### 2. Run Basic Benchmark

```python
from pipelines.evaluation.SciMMIRBenchmarkIntegration import run_scimmir_benchmark_suite

# Run benchmark with 1000 samples
result = run_scimmir_benchmark_suite(
    limit_samples=1000,
    cache_dir="./data/scimmir_cache",
    report_path="./data/scimmir_benchmark_report.md"
)

# View results
print(f"🎯 Text→Image MRR: {result.text2img_mrr*100:.2f}%")
print(f"🎯 Image→Text MRR: {result.img2text_mrr*100:.2f}%") 
print(f"🏆 Total Samples: {result.total_samples}")
```

### 3. Use API Endpoint

```bash
# Start the API
python main.py

# Run SciMMIR benchmark
curl -X POST "http://localhost:8000/evaluation/scimmir-benchmark" \
     -H "Content-Type: application/json" \
     -d '{
       "limit_samples": 500,
       "generate_report": true
     }'
```

## 📊 Understanding Results

### Key Metrics

1. **MRR (Mean Reciprocal Rank)**: Average of 1/rank for correct matches
   - Higher is better (max = 1.0)
   - Measures how quickly correct results are found

2. **Recall@k**: Percentage of queries with correct result in top-k
   - Recall@1: Correct result is #1
   - Recall@5: Correct result in top-5  
   - Recall@10: Correct result in top-10

### Baseline Comparison

Your results are automatically compared against published SciMMIR baselines:

| Model | Text→Image MRR | Image→Text MRR |
|-------|----------------|----------------|
| **BLIP-base+BERT** | 11.15% | 12.69% |
| **CLIP-base** | ~8.5% | ~9.2% |
| **Your Model** | ? | ? |

## 🎯 Benchmarking Your Data

### Option 1: Use Existing Data

If you have scientific papers with figures already ingested:

```python
from pipelines.evaluation.SciMMIRBenchmarkIntegration import (
    SciMMIRBenchmarkRunner, 
    SciMMIRSample
)

# Convert your data to benchmark format
your_samples = []
for paper in your_papers:
    if paper.get('figures'):
        for figure in paper['figures']:
            sample = SciMMIRSample(
                text=figure['description'],
                image=figure.get('image'),
                class_label=figure.get('class', 'figure'),
                paper_id=paper['id'],
                domain=paper.get('domain', 'general')
            )
            your_samples.append(sample)

# Run benchmark on your data
from clients.huggingface.CLIPClient import CLIPClient
from clients.huggingface.SciBERTClient import SciBERTClient
from clients.vector.MilvusClient import MilvusClient

clip_client = CLIPClient()
scibert_client = SciBERTClient()
vector_client = MilvusClient()

runner = SciMMIRBenchmarkRunner(clip_client, scibert_client, vector_client)
result = runner.run_benchmark(your_samples)
```

### Option 2: Load SciMMIR Dataset

The integration automatically downloads and caches the SciMMIR dataset:

```python
from pipelines.evaluation.SciMMIRBenchmarkIntegration import SciMMIRDataLoader

loader = SciMMIRDataLoader("./data/scimmir_cache")
samples = loader.load_test_samples(limit=1000)

print(f"Loaded {len(samples)} samples")
print(f"Categories: {set(s.class_label for s in samples)}")
print(f"Domains: {set(s.domain for s in samples)}")
```

## 📈 Performance Analysis

### Category Breakdown

SciMMIR includes 5 figure categories:
- `fig_architecture`: System/model diagrams
- `fig_natural`: Natural images
- `fig_chart`: Charts and graphs
- `fig_equation`: Mathematical equations
- `fig_table`: Tables and data

Example analysis:
```python
for category, metrics in result.by_category.items():
    print(f"{category}:")
    print(f"  - Samples: {metrics['sample_count']}")
    print(f"  - T2I MRR: {metrics['text2img_mrr']:.4f}")
    print(f"  - I2T MRR: {metrics['img2text_mrr']:.4f}")
```

### Domain Analysis

Performance across scientific domains:
```python
for domain, metrics in result.by_domain.items():
    print(f"{domain}: {metrics['sample_count']} samples")
    print(f"  - T2I: {metrics['text2img_mrr']:.4f}")
    print(f"  - I2T: {metrics['img2text_mrr']:.4f}")
```

## 🔧 Advanced Usage

### Custom Evaluation Setup

```python
from pipelines.evaluation.SciMMIRBenchmarkIntegration import SciMMIRBenchmarkRunner

# Initialize with custom configurations
runner = SciMMIRBenchmarkRunner(
    clip_client=your_clip_client,
    scibert_client=your_scibert_client, 
    vector_client=your_vector_client
)

# Run with custom parameters
result = runner.run_benchmark(
    samples=your_samples,
    model_name="IAAIR-Custom-v1.0",
    top_k=20  # Evaluate up to top-20 results
)
```

### Batch Processing

For large datasets, process in batches:

```python
# Process 10K samples in batches of 1K
total_samples = 10000
batch_size = 1000
results = []

for i in range(0, total_samples, batch_size):
    batch_result = run_scimmir_benchmark_suite(
        limit_samples=batch_size,
        offset=i  # If supported
    )
    results.append(batch_result)

# Aggregate results
combined_result = aggregate_benchmark_results(results)
```

## 📊 Expected Performance

Based on your system architecture:

### Predicted Baseline Performance
- **Text→Image MRR**: 8-15% (competitive with CLIP-based systems)
- **Image→Text MRR**: 10-18% (SciBERT text understanding advantage)
- **Strong Categories**: fig_chart, fig_table (structured content)
- **Challenge Categories**: fig_natural, fig_equation (visual complexity)

### Optimization Opportunities

1. **Hybrid Embeddings**: Combine SciBERT + CLIP features
2. **Domain Adaptation**: Fine-tune on scientific domains
3. **Multi-scale Features**: Use different image resolutions
4. **Cross-attention**: Text-image interaction modeling

## 🚨 Troubleshooting

### Common Issues

1. **Memory Errors**:
   ```bash
   # Reduce batch size
   result = run_scimmir_benchmark_suite(limit_samples=100)
   ```

2. **Dataset Download Fails**:
   ```python
   # Manual dataset download
   import datasets
   ds = datasets.load_dataset("m-a-p/SciMMIR")
   ds.save_to_disk("./data/scimmir_cache/scimmir_dataset")
   ```

3. **Model Loading Issues**:
   ```python
   # Initialize models explicitly
   clip_client = CLIPClient()
   if not clip_client.initialize():
       print("CLIP model failed to load")
   ```

### Performance Issues

- **Slow evaluation**: Reduce `limit_samples` or use CPU-only mode
- **High memory usage**: Process in smaller batches
- **Poor results**: Check image preprocessing and embedding normalization

## 📚 References

- **SciMMIR Paper**: [Benchmarking Scientific Multi-modal Information Retrieval](https://arxiv.org/abs/2401.13478)
- **Dataset**: [HuggingFace - m-a-p/SciMMIR](https://huggingface.co/datasets/m-a-p/SciMMIR)
- **Code Repository**: [GitHub - SciMMIR](https://github.com/Wusiwei0410/SciMMIR)

## 🎉 Next Steps

1. **Run Initial Benchmark**: Test with 500-1000 samples
2. **Analyze Results**: Identify strengths and weaknesses by category/domain  
3. **Optimize Performance**: Fine-tune embeddings or retrieval strategy
4. **Scale Up**: Run full benchmark on complete dataset
5. **Compare Models**: Test different embedding combinations

Your IAAIR system is now ready to benchmark against state-of-the-art multi-modal scientific retrieval systems! 🚀