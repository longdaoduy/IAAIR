
# SciMMIR Benchmark Evaluation Report

**Model**: IAAIR-SciBERT-CLIP  
**Evaluation Date**: 2026-02-22 10:46:01  
**Total Samples**: 50

## 📊 Overall Performance

### Text-to-Image Retrieval
- **MRR**: 0.1145 (11.45%)
- **Recall@1**: 0.0400 (4.00%)
- **Recall@5**: 0.1600 (16.00%)
- **Recall@10**: 0.2600 (26.00%)

### Image-to-Text Retrieval
- **MRR**: 0.1503 (15.03%)
- **Recall@1**: 0.0800 (8.00%)
- **Recall@5**: 0.1600 (16.00%)
- **Recall@10**: 0.3000 (30.00%)

## 🏆 Comparison with Baselines

**Your Rank**: #1 out of 4 models  
**Percentile**: 100.0th percentile

### Performance vs. Published Baselines:
- **BLIP-base+BERT**: T2I MRR: 11.15%, I2T MRR: 12.69%
- **CLIP-base**: T2I MRR: 8.50%, I2T MRR: 9.20%
- **Random**: T2I MRR: 0.10%, I2T MRR: 0.10%
- **IAAIR-SciBERT-CLIP (Your Model)**: T2I MRR: 11.45%, I2T MRR: 15.03%

## ✨ Key Improvements

- Text-to-Image retrieval improved by 0.30% over best baseline
- Image-to-Text retrieval improved by 2.34% over best baseline

## 📈 Performance by Category

### Table Parameter
- Samples: 2
- T2I MRR: 0.7500
- I2T MRR: 0.7500

### Fig Illustration
- Samples: 5
- T2I MRR: 0.5567
- I2T MRR: 0.4567

### Fig Result
- Samples: 27
- T2I MRR: 0.1649
- I2T MRR: 0.2246

### Table Result
- Samples: 15
- T2I MRR: 0.3143
- I2T MRR: 0.3450

### Fig Architecture
- Samples: 1
- T2I MRR: 1.0000
- I2T MRR: 1.0000

## 🔬 Performance by Domain

### Physics
- Samples: 2
- T2I MRR: 0.7500
- I2T MRR: 1.0000

### Cs
- Samples: 5
- T2I MRR: 0.3067
- I2T MRR: 0.4567

### Biomedical
- Samples: 3
- T2I MRR: 0.6111
- I2T MRR: 0.6667

### General
- Samples: 40
- T2I MRR: 0.1164
- I2T MRR: 0.1641

