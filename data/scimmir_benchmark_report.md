
# SciMMIR Benchmark Evaluation Report

**Model**: IAAIR-SciBERT-CLIP  
**Evaluation Date**: 2026-02-23 15:59:08  
**Total Samples**: 50

## 📊 Overall Performance

### Text-to-Image Retrieval
- **MRR**: 0.1270 (12.70%)
- **Recall@1**: 0.0600 (6.00%)
- **Recall@5**: 0.1000 (10.00%)
- **Recall@10**: 0.2800 (28.00%)

### Image-to-Text Retrieval
- **MRR**: 0.1536 (15.36%)
- **Recall@1**: 0.0600 (6.00%)
- **Recall@5**: 0.1600 (16.00%)
- **Recall@10**: 0.3000 (30.00%)

## 🏆 Comparison with Baselines

**Your Rank**: #2 out of 6 models  
**Percentile**: 83.3th percentile

### Performance vs. Published Baselines:
- **CLIP-base**: T2I MRR: 0.65%, I2T MRR: 0.64%
- **BLIP2-OPT-2.7B**: T2I MRR: 0.10%, I2T MRR: 0.03%
- **BLIP2-FLAN-T5-XLL**: T2I MRR: 0.04%, I2T MRR: 0.00%
- **mPLUG-Owl2-LLaMA2-7B**: T2I MRR: 0.07%, I2T MRR: 0.00%
- **Kosmos-2**: T2I MRR: 0.03%, I2T MRR: 0.00%
- **IAAIR-SciBERT-CLIP (Your Model)**: T2I MRR: 12.70%, I2T MRR: 15.36%

## ✨ Key Improvements

- Strong Recall@1 performance: 6.00%
- Superior figure result text-to-image performance
- Superior figure result image-to-text performance
- Superior table result text-to-image performance
- Superior table result image-to-text performance
- Superior figure illustration text-to-image performance
- Superior figure illustration image-to-text performance

## 📋 Subset Performance (CLIP-BERT Methodology)

### 🏆 Subset Performance vs. Baselines

#### Figure Result

**Your Model Rank**: #1 out of 6 models

| Model | Text→Image MRR | Image→Text MRR | Text→Image Hits@10 | Image→Text Hits@10 |
|-------|----------------|----------------|--------------------|--------------------|
| **IAAIR-SciBERT-CLIP** | **15.292%** | **22.088%** | **33.333%** | **37.037%** |
| CLIP-base | 0.373% | 0.386% | 0.643% | 0.738% |
| BLIP2-FLAN-T5-XLL | 0.062% | 0.004% | 0.105% | 0.000% |
| BLIP2-OPT-2.7B | 0.031% | 0.014% | 0.042% | 0.032% |
| mPLUG-Owl2-LLaMA2-7B | 0.019% | 0.002% | 0.021% | 0.000% |
| Kosmos-2 | 0.006% | 0.002% | 0.011% | 0.000% |

#### Table Result

**Your Model Rank**: #1 out of 6 models

| Model | Text→Image MRR | Image→Text MRR | Text→Image Hits@10 | Image→Text Hits@10 |
|-------|----------------|----------------|--------------------|--------------------|
| **IAAIR-SciBERT-CLIP** | **27.118%** | **34.655%** | **87.500%** | **75.000%** |
| CLIP-base | 0.281% | 0.177% | 0.544% | 0.284% |
| BLIP2-OPT-2.7B | 0.076% | 0.010% | 0.213% | 0.024% |
| BLIP2-FLAN-T5-XLL | 0.041% | 0.003% | 0.095% | 0.000% |
| mPLUG-Owl2-LLaMA2-7B | 0.001% | 0.004% | 0.000% | 0.000% |
| Kosmos-2 | 0.000% | 0.001% | 0.000% | 0.000% |

#### Figure Illustration

**Your Model Rank**: #1 out of 6 models

| Model | Text→Image MRR | Image→Text MRR | Text→Image Hits@10 | Image→Text Hits@10 |
|-------|----------------|----------------|--------------------|--------------------|
| **IAAIR-SciBERT-CLIP** | **47.333%** | **44.000%** | **100.000%** | **100.000%** |
| CLIP-base | 0.750% | 0.458% | 1.237% | 0.716% |
| mPLUG-Owl2-LLaMA2-7B | 0.302% | 0.003% | 0.521% | 0.000% |
| BLIP2-FLAN-T5-XLL | 0.037% | 0.005% | 0.065% | 0.000% |
| BLIP2-OPT-2.7B | 0.033% | 0.006% | 0.130% | 0.000% |
| Kosmos-2 | 0.011% | 0.004% | 0.000% | 0.000% |

### Figure Subset (32 total samples):

**Figure Result**: 27 samples
- T2I MRR: 0.1529 (15.29%)
- I2T MRR: 0.2209 (22.09%)
- T2I Recall@1: 0.0741 (7.41%)
- I2T Recall@1: 0.1111 (11.11%)

**Figure Illustration**: 5 samples
- T2I MRR: 0.4733 (47.33%)
- I2T MRR: 0.4400 (44.00%)
- T2I Recall@1: 0.2000 (20.00%)
- I2T Recall@1: 0.2000 (20.00%)

### Table Subset (16 total samples):

**Table Result**: 16 samples
- T2I MRR: 0.2712 (27.12%)
- I2T MRR: 0.3466 (34.66%)
- T2I Recall@1: 0.1250 (12.50%)
- I2T Recall@1: 0.1875 (18.75%)

### Expected SciMMIR Subset Distribution:
- **Figure Result**: ~9,488 samples (Expected from CLIP-BERT evaluation)
- **Figure Illustration**: ~1,536 samples
- **Figure Architecture**: ~467 samples
- **Table Result**: ~4,229 samples
- **Table Parameter**: ~543 samples
- **Total**: 16,263 samples

