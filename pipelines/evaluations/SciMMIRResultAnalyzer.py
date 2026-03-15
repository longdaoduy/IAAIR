import logging
from typing import List, Dict, Optional, Tuple, Any
from models.entities.evaluations.SciMMIRBenchmarkResult import SciMMIRBenchmarkResult

class SciMMIRResultAnalyzer:
    """Analyze and compare SciMMIR benchmark results."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compare_with_baselines(self, result: SciMMIRBenchmarkResult) -> Dict[str, Any]:
        """Compare results with published SciMMIR baselines."""

        # Published baselines from SciMMIR paper with subset-specific performance
        baselines = {
            "CLIP-base": {
                # Overall performance (estimated averages)
                "text2img_mrr": 0.652,  # Average across all subsets
                "img2text_mrr": 0.641,
                # Subset-specific performance
                "subsets": {
                    "figure_architecture": {
                        "text2img_mrr": 1.351, "text2img_hits_at_10": 1.927,
                        "img2text_mrr": 1.074, "img2text_hits_at_10": 2.141
                    },
                    "figure_illustration": {
                        "text2img_mrr": 0.750, "text2img_hits_at_10": 1.237,
                        "img2text_mrr": 0.458, "img2text_hits_at_10": 0.716
                    },
                    "figure_result": {
                        "text2img_mrr": 0.373, "text2img_hits_at_10": 0.643,
                        "img2text_mrr": 0.386, "img2text_hits_at_10": 0.738
                    },
                    "table_result": {
                        "text2img_mrr": 0.281, "text2img_hits_at_10": 0.544,
                        "img2text_mrr": 0.177, "img2text_hits_at_10": 0.284
                    },
                    "table_parameter": {
                        "text2img_mrr": 0.545, "text2img_hits_at_10": 0.921,
                        "img2text_mrr": 0.558, "img2text_hits_at_10": 1.105
                    }
                }
            },
            "BLIP2-OPT-2.7B": {
                "text2img_mrr": 0.100,  # Average across all subsets
                "img2text_mrr": 0.027,
                "subsets": {
                    "figure_architecture": {
                        "text2img_mrr": 0.130, "text2img_hits_at_10": 0.214,
                        "img2text_mrr": 0.005, "img2text_hits_at_10": 0.000
                    },
                    "figure_illustration": {
                        "text2img_mrr": 0.033, "text2img_hits_at_10": 0.130,
                        "img2text_mrr": 0.006, "img2text_hits_at_10": 0.000
                    },
                    "figure_result": {
                        "text2img_mrr": 0.031, "text2img_hits_at_10": 0.042,
                        "img2text_mrr": 0.014, "img2text_hits_at_10": 0.032
                    },
                    "table_result": {
                        "text2img_mrr": 0.076, "text2img_hits_at_10": 0.213,
                        "img2text_mrr": 0.010, "img2text_hits_at_10": 0.024
                    },
                    "table_parameter": {
                        "text2img_mrr": 0.228, "text2img_hits_at_10": 0.368,
                        "img2text_mrr": 0.101, "img2text_hits_at_10": 0.184
                    }
                }
            },
            "BLIP2-FLAN-T5-XLL": {
                "text2img_mrr": 0.045,
                "img2text_mrr": 0.004,
                "subsets": {
                    "figure_architecture": {
                        "text2img_mrr": 0.056, "text2img_hits_at_10": 0.214,
                        "img2text_mrr": 0.003, "img2text_hits_at_10": 0.000
                    },
                    "figure_illustration": {
                        "text2img_mrr": 0.037, "text2img_hits_at_10": 0.065,
                        "img2text_mrr": 0.005, "img2text_hits_at_10": 0.000
                    },
                    "figure_result": {
                        "text2img_mrr": 0.062, "text2img_hits_at_10": 0.105,
                        "img2text_mrr": 0.004, "img2text_hits_at_10": 0.000
                    },
                    "table_result": {
                        "text2img_mrr": 0.041, "text2img_hits_at_10": 0.095,
                        "img2text_mrr": 0.003, "img2text_hits_at_10": 0.000
                    },
                    "table_parameter": {
                        "text2img_mrr": 0.030, "text2img_hits_at_10": 0.184,
                        "img2text_mrr": 0.003, "img2text_hits_at_10": 0.000
                    }
                }
            },
            "mPLUG-Owl2-LLaMA2-7B": {
                "text2img_mrr": 0.070,
                "img2text_mrr": 0.003,
                "subsets": {
                    "figure_architecture": {
                        "text2img_mrr": 0.022, "text2img_hits_at_10": 0.000,
                        "img2text_mrr": 0.003, "img2text_hits_at_10": 0.000
                    },
                    "figure_illustration": {
                        "text2img_mrr": 0.302, "text2img_hits_at_10": 0.521,
                        "img2text_mrr": 0.003, "img2text_hits_at_10": 0.000
                    },
                    "figure_result": {
                        "text2img_mrr": 0.019, "text2img_hits_at_10": 0.021,
                        "img2text_mrr": 0.002, "img2text_hits_at_10": 0.000
                    },
                    "table_result": {
                        "text2img_mrr": 0.001, "text2img_hits_at_10": 0.000,
                        "img2text_mrr": 0.004, "img2text_hits_at_10": 0.000
                    },
                    "table_parameter": {
                        "text2img_mrr": 0.002, "text2img_hits_at_10": 0.000,
                        "img2text_mrr": 0.005, "img2text_hits_at_10": 0.000
                    }
                }
            },
            "Kosmos-2": {
                "text2img_mrr": 0.028,
                "img2text_mrr": 0.004,
                "subsets": {
                    "figure_architecture": {
                        "text2img_mrr": 0.123, "text2img_hits_at_10": 0.428,
                        "img2text_mrr": 0.008, "img2text_hits_at_10": 0.000
                    },
                    "figure_illustration": {
                        "text2img_mrr": 0.011, "text2img_hits_at_10": 0.000,
                        "img2text_mrr": 0.004, "img2text_hits_at_10": 0.000
                    },
                    "figure_result": {
                        "text2img_mrr": 0.006, "text2img_hits_at_10": 0.011,
                        "img2text_mrr": 0.002, "img2text_hits_at_10": 0.000
                    },
                    "table_result": {
                        "text2img_mrr": 0.000, "text2img_hits_at_10": 0.000,
                        "img2text_mrr": 0.001, "img2text_hits_at_10": 0.000
                    },
                    "table_parameter": {
                        "text2img_mrr": 0.000, "text2img_hits_at_10": 0.000,
                        "img2text_mrr": 0.003, "img2text_hits_at_10": 0.000
                    }
                }
            }
        }

        comparison = {
            "your_model": {
                "name": result.model_name,
                "text2img_mrr": result.text2img_mrr * 100,  # Convert to percentage
                "img2text_mrr": result.img2text_mrr * 100,
                "text2img_recall_at_1": result.text2img_recall_at_1 * 100,
                "img2text_recall_at_1": result.img2text_recall_at_1 * 100
            },
            "baselines": baselines,
            "performance_ranking": self._rank_performance(result, baselines),
            "improvement_analysis": self._analyze_improvements(result, baselines)
        }

        # Add subset comparisons if available
        if result.subset_results:
            comparison["subset_performance"] = {
                "figure_subset": {
                    subset: metrics for subset, metrics in result.subset_results.items()
                    if subset.startswith('figure_')
                },
                "table_subset": {
                    subset: metrics for subset, metrics in result.subset_results.items()
                    if subset.startswith('table_')
                }
            }

            # Calculate subset totals (like CLIP-BERT methodology)
            figure_total_samples = sum(
                metrics['sample_count'] for subset, metrics in result.subset_results.items()
                if subset.startswith('figure_')
            )
            table_total_samples = sum(
                metrics['sample_count'] for subset, metrics in result.subset_results.items()
                if subset.startswith('table_')
            )

            comparison["subset_totals"] = {
                "figure_subset_samples": figure_total_samples,
                "table_subset_samples": table_total_samples,
                "total_subset_samples": figure_total_samples + table_total_samples
            }

            # Add detailed subset-specific comparisons with baselines
            comparison["subset_baseline_comparison"] = self._compare_subset_performance(result, baselines)

        return comparison

    def _compare_subset_performance(self, result: SciMMIRBenchmarkResult, baselines: Dict) -> Dict[str, Any]:
        """Compare subset-specific performance with baseline models."""
        subset_comparisons = {}

        if not result.subset_results:
            return subset_comparisons

        for subset_name, your_metrics in result.subset_results.items():
            subset_comparisons[subset_name] = {
                "your_performance": {
                    "text2img_mrr": your_metrics['text2img_mrr'] * 100,
                    "img2text_mrr": your_metrics['img2text_mrr'] * 100,
                    "text2img_recall_at_10": your_metrics.get('text2img_recall_at_10', 0) * 100,
                    "img2text_recall_at_10": your_metrics.get('img2text_recall_at_10', 0) * 100,
                    "sample_count": your_metrics['sample_count']
                },
                "baseline_comparison": {},
                "ranking": {}
            }

            # Compare with each baseline that has subset data
            baseline_scores = []
            for model_name, baseline_data in baselines.items():
                if "subsets" in baseline_data and subset_name in baseline_data["subsets"]:
                    baseline_subset = baseline_data["subsets"][subset_name]
                    subset_comparisons[subset_name]["baseline_comparison"][model_name] = {
                        "text2img_mrr": baseline_subset["text2img_mrr"],
                        "img2text_mrr": baseline_subset["img2text_mrr"],
                        "text2img_hits_at_10": baseline_subset.get("text2img_hits_at_10", 0),
                        "img2text_hits_at_10": baseline_subset.get("img2text_hits_at_10", 0)
                    }

                    # Calculate average MRR for ranking
                    avg_mrr = (baseline_subset["text2img_mrr"] + baseline_subset["img2text_mrr"]) / 2
                    baseline_scores.append((model_name, avg_mrr))

            # Add your model to ranking
            your_avg_mrr = (your_metrics['text2img_mrr'] + your_metrics['img2text_mrr']) / 2 * 100
            baseline_scores.append((result.model_name, your_avg_mrr))
            baseline_scores.sort(key=lambda x: x[1], reverse=True)

            # Find your rank
            your_rank = next(i for i, (name, _) in enumerate(baseline_scores, 1) if name == result.model_name)

            subset_comparisons[subset_name]["ranking"] = {
                "your_rank": your_rank,
                "total_models": len(baseline_scores),
                "all_rankings": baseline_scores
            }

        return subset_comparisons

    @staticmethod
    def _rank_performance(result: SciMMIRBenchmarkResult, baselines: Dict) -> Dict[str, Any]:
        """Rank performance against baselines."""
        your_mrr = (result.text2img_mrr + result.img2text_mrr) / 2 * 100

        baseline_scores = []
        for name, baseline_data in baselines.items():
            # Use overall MRR if available, otherwise skip
            if "text2img_mrr" in baseline_data and "img2text_mrr" in baseline_data:
                avg_mrr = (baseline_data['text2img_mrr'] + baseline_data['img2text_mrr']) / 2 * 100
                baseline_scores.append((name, avg_mrr))

        baseline_scores.append((result.model_name, your_mrr))
        baseline_scores.sort(key=lambda x: x[1], reverse=True)

        your_rank = next(i for i, (name, _) in enumerate(baseline_scores, 1) if name == result.model_name)

        return {
            "ranking": baseline_scores,
            "your_rank": your_rank,
            "total_models": len(baseline_scores),
            "percentile": (len(baseline_scores) - your_rank + 1) / len(baseline_scores) * 100
        }

    @staticmethod
    def _analyze_improvements(result: SciMMIRBenchmarkResult, baselines: Dict) -> List[str]:
        """Analyze what improvements your model shows."""
        improvements = []

        your_text2img = result.text2img_mrr * 100
        your_img2text = result.img2text_mrr * 100

        # Find best baseline performance across all models
        best_baseline_text2img = 0
        best_baseline_img2text = 0
        best_text2img_model = ""
        best_img2text_model = ""

        for model_name, baseline_data in baselines.items():
            if "text2img_mrr" in baseline_data and "img2text_mrr" in baseline_data:
                t2i_mrr = baseline_data['text2img_mrr'] * 100 if baseline_data['text2img_mrr'] < 1 else baseline_data[
                    'text2img_mrr']
                i2t_mrr = baseline_data['img2text_mrr'] * 100 if baseline_data['img2text_mrr'] < 1 else baseline_data[
                    'img2text_mrr']

                if t2i_mrr > best_baseline_text2img:
                    best_baseline_text2img = t2i_mrr
                    best_text2img_model = model_name

                if i2t_mrr > best_baseline_img2text:
                    best_baseline_img2text = i2t_mrr
                    best_img2text_model = model_name

        if your_text2img > best_baseline_text2img:
            improvements.append(
                f"Text-to-Image retrievals improved by {your_text2img - best_baseline_text2img:.3f}% over {best_text2img_model}")

        if your_img2text > best_baseline_img2text:
            improvements.append(
                f"Image-to-Text retrievals improved by {your_img2text - best_baseline_img2text:.3f}% over {best_img2text_model}")

        if result.text2img_recall_at_1 > 0.01:  # 1% threshold
            improvements.append(f"Strong Recall@1 performance: {result.text2img_recall_at_1 * 100:.2f}%")

        # Check subset-specific improvements
        if result.subset_results:
            best_subset_performance = {}
            for subset_name in result.subset_results:
                best_t2i = 0
                best_i2t = 0
                for model_name, baseline_data in baselines.items():
                    if "subsets" in baseline_data and subset_name in baseline_data["subsets"]:
                        subset_data = baseline_data["subsets"][subset_name]
                        if subset_data["text2img_mrr"] > best_t2i:
                            best_t2i = subset_data["text2img_mrr"]
                        if subset_data["img2text_mrr"] > best_i2t:
                            best_i2t = subset_data["img2text_mrr"]

                your_subset = result.subset_results[subset_name]
                if your_subset['text2img_mrr'] * 100 > best_t2i:
                    improvements.append(f"Superior {subset_name.replace('_', ' ')} text-to-image performance")
                if your_subset['img2text_mrr'] * 100 > best_i2t:
                    improvements.append(f"Superior {subset_name.replace('_', ' ')} image-to-text performance")

        return improvements

    def generate_report(self, result: SciMMIRBenchmarkResult, save_path: Optional[str] = None) -> str:
        """Generate comprehensive benchmark report."""

        comparison = self.compare_with_baselines(result)

        report = f"""
# SciMMIR Benchmark Evaluation Report

**Model**: {result.model_name}  
**Evaluation Date**: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**Total Samples**: {result.total_samples:,}

## 📊 Overall Performance

### Text-to-Image Retrieval
- **MRR**: {result.text2img_mrr:.4f} ({result.text2img_mrr * 100:.2f}%)
- **Recall@1**: {result.text2img_recall_at_1:.4f} ({result.text2img_recall_at_1 * 100:.2f}%)
- **Recall@5**: {result.text2img_recall_at_5:.4f} ({result.text2img_recall_at_5 * 100:.2f}%)
- **Recall@10**: {result.text2img_recall_at_10:.4f} ({result.text2img_recall_at_10 * 100:.2f}%)

### Image-to-Text Retrieval
- **MRR**: {result.img2text_mrr:.4f} ({result.img2text_mrr * 100:.2f}%)
- **Recall@1**: {result.img2text_recall_at_1:.4f} ({result.img2text_recall_at_1 * 100:.2f}%)
- **Recall@5**: {result.img2text_recall_at_5:.4f} ({result.img2text_recall_at_5 * 100:.2f}%)
- **Recall@10**: {result.img2text_recall_at_10:.4f} ({result.img2text_recall_at_10 * 100:.2f}%)

## 🏆 Comparison with Baselines

**Your Rank**: #{comparison['performance_ranking']['your_rank']} out of {comparison['performance_ranking']['total_models']} models  
**Percentile**: {comparison['performance_ranking']['percentile']:.1f}th percentile

### Performance vs. Published Baselines:
"""

        for name, scores in comparison['baselines'].items():
            report += f"- **{name}**: T2I MRR: {scores['text2img_mrr']:.2f}%, I2T MRR: {scores['img2text_mrr']:.2f}%\n"

        report += f"- **{result.model_name} (Your Model)**: T2I MRR: {result.text2img_mrr * 100:.2f}%, I2T MRR: {result.img2text_mrr * 100:.2f}%\n\n"

        if comparison['improvement_analysis']:
            report += "## ✨ Key Improvements\n\n"
            for improvement in comparison['improvement_analysis']:
                report += f"- {improvement}\n"
            report += "\n"

        # Add subset evaluations results (like CLIP-BERT methodology)
        if result.subset_results:
            report += "## 📋 Subset Performance (CLIP-BERT Methodology)\n\n"

            # Add detailed baseline comparison table
            if "subset_baseline_comparison" in comparison:
                report += "### 🏆 Subset Performance vs. Baselines\n\n"

                subset_comparison = comparison["subset_baseline_comparison"]

                # Create comparison table for each subset
                for subset_name, subset_data in subset_comparison.items():
                    display_name = subset_name.replace('_', ' ').title()
                    report += f"#### {display_name}\n\n"
                    report += f"**Your Model Rank**: #{subset_data['ranking']['your_rank']} out of {subset_data['ranking']['total_models']} models\n\n"

                    # Performance table
                    report += "| Model | Text→Image MRR | Image→Text MRR | Text→Image Hits@10 | Image→Text Hits@10 |\n"
                    report += "|-------|----------------|----------------|--------------------|--------------------|\n"

                    # Sort by average performance for display
                    rankings = sorted(subset_data['ranking']['all_rankings'], key=lambda x: x[1], reverse=True)

                    for model_name, avg_score in rankings:
                        if model_name == result.model_name:
                            your_perf = subset_data['your_performance']
                            report += f"| **{model_name}** | **{your_perf['text2img_mrr']:.3f}%** | **{your_perf['img2text_mrr']:.3f}%** | **{your_perf['text2img_recall_at_10']:.3f}%** | **{your_perf['img2text_recall_at_10']:.3f}%** |\n"
                        elif model_name in subset_data['baseline_comparison']:
                            baseline_perf = subset_data['baseline_comparison'][model_name]
                            report += f"| {model_name} | {baseline_perf['text2img_mrr']:.3f}% | {baseline_perf['img2text_mrr']:.3f}% | {baseline_perf['text2img_hits_at_10']:.3f}% | {baseline_perf['img2text_hits_at_10']:.3f}% |\n"

                    report += "\n"

            # Figure subset breakdown
            figure_subsets = {k: v for k, v in result.subset_results.items() if k.startswith('figure_')}
            if figure_subsets:
                total_figure_samples = sum(metrics['sample_count'] for metrics in figure_subsets.values())
                report += f"### Figure Subset ({total_figure_samples:,} total samples):\n\n"

                for subset_name, metrics in figure_subsets.items():
                    display_name = subset_name.replace('_', ' ').title()
                    report += f"**{display_name}**: {metrics['sample_count']:,} samples\n"
                    report += f"- T2I MRR: {metrics['text2img_mrr']:.4f} ({metrics['text2img_mrr'] * 100:.2f}%)\n"
                    report += f"- I2T MRR: {metrics['img2text_mrr']:.4f} ({metrics['img2text_mrr'] * 100:.2f}%)\n"
                    report += f"- T2I Recall@1: {metrics['text2img_recall_at_1']:.4f} ({metrics['text2img_recall_at_1'] * 100:.2f}%)\n"
                    report += f"- I2T Recall@1: {metrics['img2text_recall_at_1']:.4f} ({metrics['img2text_recall_at_1'] * 100:.2f}%)\n\n"

            # Table subset breakdown
            table_subsets = {k: v for k, v in result.subset_results.items() if k.startswith('table_')}
            if table_subsets:
                total_table_samples = sum(metrics['sample_count'] for metrics in table_subsets.values())
                report += f"### Table Subset ({total_table_samples:,} total samples):\n\n"

                for subset_name, metrics in table_subsets.items():
                    display_name = subset_name.replace('_', ' ').title()
                    report += f"**{display_name}**: {metrics['sample_count']:,} samples\n"
                    report += f"- T2I MRR: {metrics['text2img_mrr']:.4f} ({metrics['text2img_mrr'] * 100:.2f}%)\n"
                    report += f"- I2T MRR: {metrics['img2text_mrr']:.4f} ({metrics['img2text_mrr'] * 100:.2f}%)\n"
                    report += f"- T2I Recall@1: {metrics['text2img_recall_at_1']:.4f} ({metrics['text2img_recall_at_1'] * 100:.2f}%)\n"
                    report += f"- I2T Recall@1: {metrics['img2text_recall_at_1']:.4f} ({metrics['img2text_recall_at_1'] * 100:.2f}%)\n\n"

            # Expected subset distribution (based on your requirements)
            report += "### Expected SciMMIR Subset Distribution:\n"
            report += "- **Figure Result**: ~9,488 samples (Expected from CLIP-BERT evaluations)\n"
            report += "- **Figure Illustration**: ~1,536 samples\n"
            report += "- **Figure Architecture**: ~467 samples\n"
            report += "- **Table Result**: ~4,229 samples\n"
            report += "- **Table Parameter**: ~543 samples\n"
            report += "- **Total**: 16,263 samples\n\n"

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {save_path}")

        return report
