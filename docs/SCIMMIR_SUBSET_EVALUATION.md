# SciMMIR Subset Evaluation (CLIP-BERT Methodology)

This document describes how to evaluate your model on SciMMIR subsets following the CLIP-BERT methodology with hierarchical subset breakdowns.

## Overview

Based on the SciMMIR benchmark methodology, when evaluating models like CLIP-base+BERT, researchers used a test set consisting of 16,263 samples broken down into specific hierarchical subsets:

### Figure Subset (11,491 total test samples)
- **Figure Result**: 9,488 samples - Performance charts, comparison graphs, experimental results
- **Figure Illustration**: 1,536 samples - Diagrams, drawings, illustrations, schemas  
- **Figure Architecture**: 467 samples - System architectures, flowcharts, pipelines, workflows

### Table Subset (4,772 total test samples)
- **Table Result**: 4,229 samples - Performance comparisons, experimental data, results
- **Table Parameter**: 543 samples - Hyperparameters, configurations, settings

## Quick Start

### 1. Configure the Evaluation Environment

First, run the configuration script to check your setup:

```bash
cd /path/to/IAAIR
python configure_scimmir_evaluation.py
```

This will:
- Check if SciMMIR data is available
- Download data if needed
- Analyze subset distribution
- Estimate evaluation time

### 2. Run Subset Evaluation

Execute the main evaluation script:

```bash
python evaluate_scimmir_subsets.py
```

For a quick test with limited samples:
```bash
# Edit the script and set limit=1000 in load_test_samples()
python evaluate_scimmir_subsets.py
```

For full evaluation (16,263 samples):
```bash
# Edit the script and remove the limit parameter
python evaluate_scimmir_subsets.py
```

### 3. View Results

The evaluation generates a comprehensive report saved to:
```
data/scimmir_subset_evaluation_report.md
```

## Implementation Details

### Subset Classification

The system automatically classifies samples into subsets using:

1. **Direct Mapping**: Based on SciMMIR class labels
2. **Text Analysis**: Fallback inference from text content
3. **Hierarchical Structure**: Groups into figure/table → specific subtypes

```python
# Example subset mapping
subset_mapping = {
    'fig_result': 'figure_result',
    'fig_chart': 'figure_result', 
    'fig_plot': 'figure_result',
    'fig_diagram': 'figure_illustration',
    'fig_architecture': 'figure_architecture',
    'tab_result': 'table_result',
    'tab_parameter': 'table_parameter'
}
```

### Evaluation Metrics

For each subset, the system calculates:
- **MRR** (Mean Reciprocal Rank)
- **Recall@1, Recall@5, Recall@10**
- **Bi-directional retrieval** (Text-to-Image and Image-to-Text)

### Code Usage

```python
from pipelines.evaluation.SciMMIRBenchmarkIntegration import (
    SciMMIRDataLoader,
    SciMMIRBenchmarkRunner,
    SciMMIRResultAnalyzer
)

# Load data
data_loader = SciMMIRDataLoader()
samples = data_loader.load_test_samples(limit=1000)

# Run evaluation with subsets
benchmark_runner = SciMMIRBenchmarkRunner(clip_client, scibert_client, vector_client)
result = benchmark_runner.run_benchmark(
    samples=samples,
    model_name="Your-Model-Name",
    evaluate_subsets=True  # Enable subset evaluation
)

# Generate report
analyzer = SciMMIRResultAnalyzer()
report = analyzer.generate_report(result, save_path="evaluation_report.md")
```

## Expected Results Format

### Overall Performance
```
Text-to-Image MRR: 0.1234 (12.34%)
Image-to-Text MRR: 0.1456 (14.56%)
```

### Subset Performance
```
Figure Subset (11,491 samples):
  Figure Result: 9,488 samples
    T2I MRR: 0.1234 (12.34%)
    I2T MRR: 0.1456 (14.56%)
  
  Figure Illustration: 1,536 samples
    T2I MRR: 0.1123 (11.23%)
    I2T MRR: 0.1334 (13.34%)
  
  Figure Architecture: 467 samples
    T2I MRR: 0.0987 (9.87%)
    I2T MRR: 0.1098 (10.98%)

Table Subset (4,772 samples):
  Table Result: 4,229 samples
    T2I MRR: 0.1567 (15.67%)
    I2T MRR: 0.1789 (17.89%)
  
  Table Parameter: 543 samples
    T2I MRR: 0.1123 (11.23%)
    I2T MRR: 0.1234 (12.34%)
```

## Comparison with Baselines

The system compares your results with published baselines:

- **CLIP-base+BERT**: T2I MRR: 11.15%, I2T MRR: 12.69%
- **BLIP-base+BERT**: T2I MRR: 11.15%, I2T MRR: 12.69%
- **CLIP-base**: T2I MRR: 8.5%, I2T MRR: 9.2%

## Performance Optimization

### Memory Management
- Images are resized to 224x224 for memory efficiency
- Batch processing for large datasets
- Garbage collection between evaluations

### Speed Optimization
- Parallel embedding generation (when supported)
- Efficient similarity calculations
- Progress tracking and logging

### Sample Size Recommendations
- **Development/Testing**: 1,000 samples (~8 minutes)
- **Validation**: 5,000 samples (~40 minutes)
- **Full Evaluation**: 16,263 samples (~2 hours)

## Troubleshooting

### Data Loading Issues
1. Check internet connection for data download
2. Verify sufficient disk space (>2GB)
3. Clear cache: `rm -rf data/scimmir_cache`

### Memory Issues
1. Reduce sample limit
2. Use batch processing
3. Increase system memory/swap

### Model Client Issues
1. Verify model clients are properly configured
2. Check API keys and endpoints
3. Test individual client functionality

## File Structure

```
IAAIR/
├── pipelines/evaluation/
│   └── SciMMIRBenchmarkIntegration.py  # Main evaluation code
├── configure_scimmir_evaluation.py      # Configuration script
├── evaluate_scimmir_subsets.py          # Evaluation script
├── data/
│   ├── scimmir_cache/                   # Downloaded data
│   └── scimmir_subset_evaluation_report.md  # Generated report
└── docs/
    └── SCIMMIR_SUBSET_EVALUATION.md     # This documentation
```

## Advanced Usage

### Custom Subset Definitions

You can modify the subset mapping in `SciMMIRDataLoader`:

```python
self.subset_mapping = {
    # Add your custom mappings
    'custom_fig_type': 'figure_custom',
    'custom_tab_type': 'table_custom'
}
```

### Custom Evaluation Metrics

Add custom metrics to the evaluation:

```python
def _evaluate_custom_metric(self, samples, embeddings):
    # Your custom evaluation logic
    return custom_metrics
```

### Integration with Other Benchmarks

The subset evaluation framework can be extended for other benchmarks:

```python
# Extend for other datasets
class CustomBenchmarkRunner(SciMMIRBenchmarkRunner):
    def evaluate_custom_subsets(self, samples):
        # Your custom subset evaluation
        pass
```

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@article{scimmir2024,
  title={SciMMIR: Benchmarking Scientific Multi-Modal Information Retrieval},
  author={SciMMIR Team},
  year={2024}
}
```

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run `configure_scimmir_evaluation.py` for diagnostics
3. Review logs for detailed error information