
# SciMMIR Benchmark Evaluation Report

**Model**: IAAIR-SciBERT-CLIP  
**Evaluation Date**: 2026-02-22 21:21:37  
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

**Your Rank**: #1 out of 4 models  
**Percentile**: 100.0th percentile

### Performance vs. Published Baselines:
- **BLIP-base+BERT**: T2I MRR: 11.15%, I2T MRR: 12.69%
- **CLIP-base**: T2I MRR: 8.50%, I2T MRR: 9.20%
- **Random**: T2I MRR: 0.10%, I2T MRR: 0.10%
- **IAAIR-SciBERT-CLIP (Your Model)**: T2I MRR: 12.70%, I2T MRR: 15.36%

## ✨ Key Improvements

- Text-to-Image retrieval improved by 1.55% over best baseline
- Image-to-Text retrieval improved by 2.67% over best baseline

## 📈 Performance by Category

### Fig Illustration
- Samples: 5
- T2I MRR: 0.4733
- I2T MRR: 0.4400

### Fig Result
- Samples: 27
- T2I MRR: 0.1529
- I2T MRR: 0.2209

### Fig Architecture
- Samples: 1
- T2I MRR: 1.0000
- I2T MRR: 1.0000

### Table Parameter
- Samples: 2
- T2I MRR: 1.0000
- I2T MRR: 0.7500

### Table Result
- Samples: 15
- T2I MRR: 0.2675
- I2T MRR: 0.3821

## 🔬 Performance by Domain

### Physics
- Samples: 2
- T2I MRR: 0.7500
- I2T MRR: 0.7500

### General
- Samples: 40
- T2I MRR: 0.1437
- I2T MRR: 0.1853

### Biomedical
- Samples: 3
- T2I MRR: 0.6111
- I2T MRR: 0.6111

### Cs
- Samples: 5
- T2I MRR: 0.4567
- I2T MRR: 0.4467

