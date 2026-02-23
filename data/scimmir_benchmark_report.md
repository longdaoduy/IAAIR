
# SciMMIR Benchmark Evaluation Report

**Model**: IAAIR-SciBERT-CLIP  
**Evaluation Date**: 2026-02-23 16:56:01  
**Total Samples**: 500

## 📊 Overall Performance

### Text-to-Image Retrieval
- **MRR**: 0.0165 (1.65%)
- **Recall@1**: 0.0040 (0.40%)
- **Recall@5**: 0.0140 (1.40%)
- **Recall@10**: 0.0300 (3.00%)

### Image-to-Text Retrieval
- **MRR**: 0.0206 (2.06%)
- **Recall@1**: 0.0060 (0.60%)
- **Recall@5**: 0.0180 (1.80%)
- **Recall@10**: 0.0320 (3.20%)

## 🏆 Comparison with Baselines

**Your Rank**: #5 out of 6 models  
**Percentile**: 33.3th percentile

### Performance vs. Published Baselines:
- **CLIP-base**: T2I MRR: 0.65%, I2T MRR: 0.64%
- **BLIP2-OPT-2.7B**: T2I MRR: 0.10%, I2T MRR: 0.03%
- **BLIP2-FLAN-T5-XLL**: T2I MRR: 0.04%, I2T MRR: 0.00%
- **mPLUG-Owl2-LLaMA2-7B**: T2I MRR: 0.07%, I2T MRR: 0.00%
- **Kosmos-2**: T2I MRR: 0.03%, I2T MRR: 0.00%
- **IAAIR-SciBERT-CLIP (Your Model)**: T2I MRR: 1.65%, I2T MRR: 2.06%

## ✨ Key Improvements

- Superior figure result text-to-image performance
- Superior figure result image-to-text performance
- Superior table result text-to-image performance
- Superior table result image-to-text performance
- Superior figure architecture text-to-image performance
- Superior figure architecture image-to-text performance
- Superior table parameter text-to-image performance
- Superior table parameter image-to-text performance
- Superior figure illustration text-to-image performance
- Superior figure illustration image-to-text performance

## 📋 Subset Performance (CLIP-BERT Methodology)

### 🏆 Subset Performance vs. Baselines

#### Figure Result

**Your Model Rank**: #1 out of 6 models

| Model | Text→Image MRR | Image→Text MRR | Text→Image Hits@10 | Image→Text Hits@10 |
|-------|----------------|----------------|--------------------|--------------------|
| **IAAIR-SciBERT-CLIP** | **2.746%** | **3.071%** | **5.536%** | **4.498%** |
| CLIP-base | 0.373% | 0.386% | 0.643% | 0.738% |
| BLIP2-FLAN-T5-XLL | 0.062% | 0.004% | 0.105% | 0.000% |
| BLIP2-OPT-2.7B | 0.031% | 0.014% | 0.042% | 0.032% |
| mPLUG-Owl2-LLaMA2-7B | 0.019% | 0.002% | 0.021% | 0.000% |
| Kosmos-2 | 0.006% | 0.002% | 0.011% | 0.000% |

#### Table Result

**Your Model Rank**: #1 out of 6 models

| Model | Text→Image MRR | Image→Text MRR | Text→Image Hits@10 | Image→Text Hits@10 |
|-------|----------------|----------------|--------------------|--------------------|
| **IAAIR-SciBERT-CLIP** | **5.644%** | **4.288%** | **9.836%** | **11.475%** |
| CLIP-base | 0.281% | 0.177% | 0.544% | 0.284% |
| BLIP2-OPT-2.7B | 0.076% | 0.010% | 0.213% | 0.024% |
| BLIP2-FLAN-T5-XLL | 0.041% | 0.003% | 0.095% | 0.000% |
| mPLUG-Owl2-LLaMA2-7B | 0.001% | 0.004% | 0.000% | 0.000% |
| Kosmos-2 | 0.000% | 0.001% | 0.000% | 0.000% |

#### Figure Architecture

**Your Model Rank**: #1 out of 6 models

| Model | Text→Image MRR | Image→Text MRR | Text→Image Hits@10 | Image→Text Hits@10 |
|-------|----------------|----------------|--------------------|--------------------|
| **IAAIR-SciBERT-CLIP** | **25.499%** | **29.840%** | **60.000%** | **55.000%** |
| CLIP-base | 1.351% | 1.074% | 1.927% | 2.141% |
| BLIP2-OPT-2.7B | 0.130% | 0.005% | 0.214% | 0.000% |
| Kosmos-2 | 0.123% | 0.008% | 0.428% | 0.000% |
| BLIP2-FLAN-T5-XLL | 0.056% | 0.003% | 0.214% | 0.000% |
| mPLUG-Owl2-LLaMA2-7B | 0.022% | 0.003% | 0.000% | 0.000% |

#### Table Parameter

**Your Model Rank**: #1 out of 6 models

| Model | Text→Image MRR | Image→Text MRR | Text→Image Hits@10 | Image→Text Hits@10 |
|-------|----------------|----------------|--------------------|--------------------|
| **IAAIR-SciBERT-CLIP** | **21.199%** | **25.998%** | **81.250%** | **75.000%** |
| CLIP-base | 0.545% | 0.558% | 0.921% | 1.105% |
| BLIP2-OPT-2.7B | 0.228% | 0.101% | 0.368% | 0.184% |
| BLIP2-FLAN-T5-XLL | 0.030% | 0.003% | 0.184% | 0.000% |
| mPLUG-Owl2-LLaMA2-7B | 0.002% | 0.005% | 0.000% | 0.000% |
| Kosmos-2 | 0.000% | 0.003% | 0.000% | 0.000% |

#### Figure Illustration

**Your Model Rank**: #1 out of 6 models

| Model | Text→Image MRR | Image→Text MRR | Text→Image Hits@10 | Image→Text Hits@10 |
|-------|----------------|----------------|--------------------|--------------------|
| **IAAIR-SciBERT-CLIP** | **7.249%** | **8.491%** | **18.868%** | **13.208%** |
| CLIP-base | 0.750% | 0.458% | 1.237% | 0.716% |
| mPLUG-Owl2-LLaMA2-7B | 0.302% | 0.003% | 0.521% | 0.000% |
| BLIP2-FLAN-T5-XLL | 0.037% | 0.005% | 0.065% | 0.000% |
| BLIP2-OPT-2.7B | 0.033% | 0.006% | 0.130% | 0.000% |
| Kosmos-2 | 0.011% | 0.004% | 0.000% | 0.000% |

### Figure Subset (362 total samples):

**Figure Result**: 289 samples
- T2I MRR: 0.0275 (2.75%)
- I2T MRR: 0.0307 (3.07%)
- T2I Recall@1: 0.0069 (0.69%)
- I2T Recall@1: 0.0104 (1.04%)

**Figure Architecture**: 20 samples
- T2I MRR: 0.2550 (25.50%)
- I2T MRR: 0.2984 (29.84%)
- T2I Recall@1: 0.1000 (10.00%)
- I2T Recall@1: 0.1500 (15.00%)

**Figure Illustration**: 53 samples
- T2I MRR: 0.0725 (7.25%)
- I2T MRR: 0.0849 (8.49%)
- T2I Recall@1: 0.0000 (0.00%)
- I2T Recall@1: 0.0189 (1.89%)

### Table Subset (138 total samples):

**Table Result**: 122 samples
- T2I MRR: 0.0564 (5.64%)
- I2T MRR: 0.0429 (4.29%)
- T2I Recall@1: 0.0164 (1.64%)
- I2T Recall@1: 0.0000 (0.00%)

**Table Parameter**: 16 samples
- T2I MRR: 0.2120 (21.20%)
- I2T MRR: 0.2600 (26.00%)
- T2I Recall@1: 0.0625 (6.25%)
- I2T Recall@1: 0.0625 (6.25%)

### Expected SciMMIR Subset Distribution:
- **Figure Result**: ~9,488 samples (Expected from CLIP-BERT evaluation)
- **Figure Illustration**: ~1,536 samples
- **Figure Architecture**: ~467 samples
- **Table Result**: ~4,229 samples
- **Table Parameter**: ~543 samples
- **Total**: 16,263 samples

