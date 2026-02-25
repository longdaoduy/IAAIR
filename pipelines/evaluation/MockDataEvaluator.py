"""
Mock Data Evaluation Pipeline

This module evaluates the system's performance on the mock evaluation dataset
containing 50 questions (25 graph-based, 25 semantic-based) derived from
enriched OpenAlex papers data.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import os

logger = logging.getLogger(__name__)


@dataclass
class MockEvaluationResult:
    """Results from mock data evaluation."""
    question_id: str
    question: str
    question_type: str  # 'graph' or 'semantic'
    category: str
    success: bool
    response_time: float
    retrieved_papers: List[str]
    expected_papers: List[str]
    precision: float
    recall: float
    f1_score: float
    ai_response: Optional[str] = None
    expected_ai_response: Optional[str] = None
    ai_response_similarity: Optional[float] = None
    ai_generation_time: Optional[float] = None
    error_message: Optional[str] = None
    similarity_scores: Optional[List[float]] = None


@dataclass
class MockEvaluationSummary:
    """Summary of mock evaluation results."""
    total_questions: int
    successful_questions: int
    failed_questions: int
    avg_response_time: float
    overall_precision: float
    overall_recall: float
    overall_f1: float
    avg_ai_response_similarity: float
    avg_ai_generation_time: float
    ai_response_success_rate: float
    graph_performance: Dict[str, float]
    semantic_performance: Dict[str, float]
    category_breakdown: Dict[str, Dict[str, float]]


class MockDataEvaluator:
    """Evaluator for mock data questions testing both graph and semantic search."""

    def __init__(self, service_factory):
        self.service_factory = service_factory
        self.results = []

    def load_mock_data(self, mock_data_path: str = None) -> List[Dict]:
        """Load mock evaluation questions from JSON file."""
        if not mock_data_path:
            # Default path relative to current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            mock_data_path = os.path.join(project_root, 'data', 'evaluation_mock_data.json')

        try:
            with open(mock_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                questions = data.get('evaluation_questions', [])
                
            logger.info(f"Loaded {len(questions)} mock evaluation questions from {mock_data_path}")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to load mock data from {mock_data_path}: {e}")
            return []

    def evaluate_graph_question(self, question_data: Dict) -> MockEvaluationResult:
        """Evaluate a graph-based question using Cypher queries."""
        start_time = time.time()
        question_id = question_data['id']
        question = question_data['question']
        
        try:
            expected_evidence = question_data['expected_evidence']
            expected_papers = expected_evidence.get('paper_ids', [])
            
            # Execute the Cypher query if available
            if 'cypher_query' in expected_evidence:
                cypher_query = expected_evidence['cypher_query']
                
                # Use graph query handler to execute the query
                result = self.service_factory.query_handler.execute_cypher_query(cypher_query)
                
                # Extract paper IDs from result if available
                retrieved_papers = []
                if result and 'data' in result:
                    for record in result['data']:
                        # Try to find paper IDs in the record
                        for key, value in record.items():
                            if key in ['paper_id', 'id'] and isinstance(value, str) and value.startswith('W'):
                                retrieved_papers.append(value)
                            elif isinstance(value, str) and value.startswith('W'):
                                retrieved_papers.append(value)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_papers = []
                for paper_id in retrieved_papers:
                    if paper_id not in seen:
                        seen.add(paper_id)
                        unique_papers.append(paper_id)
                retrieved_papers = unique_papers
                
                # Calculate metrics
                precision, recall, f1 = self._calculate_metrics(retrieved_papers, expected_papers)
                
                # Generate AI response if deepseek client is available
                ai_response = None
                ai_generation_time = 0.0
                ai_response_similarity = 0.0
                expected_ai_response = expected_evidence.get('expected_ai_response', '')
                
                if self.service_factory and hasattr(self.service_factory, 'deepseek_client') and self.service_factory.deepseek_client:
                    ai_start_time = time.time()
                    try:
                        # Create context from retrieved results
                        context = f"Query: {question}\nResults: {len(retrieved_papers)} papers found"
                        if result and 'data' in result:
                            context += f"\nQuery result: {result['data'][:3]}"  # Sample of results
                        
                        ai_response = self.service_factory.deepseek_client.generate_content(
                            prompt=f"Based on the graph database query results, provide a comprehensive answer to: {question}",
                            system_prompt=f"You are an AI assistant analyzing academic papers. Use the provided query results to answer the question accurately. Context: {context}"
                        )
                        ai_generation_time = time.time() - ai_start_time
                        
                        # Calculate AI response similarity if expected response exists
                        if expected_ai_response and ai_response:
                            ai_response_similarity = self._calculate_text_similarity(ai_response, expected_ai_response)
                            
                    except Exception as e:
                        logger.warning(f"AI response generation failed for {question_id}: {e}")
                        ai_generation_time = time.time() - ai_start_time
                
                response_time = time.time() - start_time
                
                return MockEvaluationResult(
                    question_id=question_id,
                    question=question,
                    question_type='graph',
                    category=question_data['category'],
                    success=True,
                    response_time=response_time,
                    retrieved_papers=retrieved_papers,
                    expected_papers=expected_papers,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    ai_response=ai_response,
                    expected_ai_response=expected_ai_response,
                    ai_response_similarity=ai_response_similarity,
                    ai_generation_time=ai_generation_time
                )
            else:
                # Fallback: try to answer using hybrid search
                return self._evaluate_as_semantic_fallback(question_data, start_time)
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Error evaluating graph question {question_id}: {e}")
            
            return MockEvaluationResult(
                question_id=question_id,
                question=question,
                question_type='graph',
                category=question_data['category'],
                success=False,
                response_time=response_time,
                retrieved_papers=[],
                expected_papers=expected_evidence.get('paper_ids', []),
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                error_message=str(e)
            )

    def evaluate_semantic_question(self, question_data: Dict) -> MockEvaluationResult:
        """Evaluate a semantic question using vector similarity search."""
        start_time = time.time()
        question_id = question_data['id']
        question = question_data['question']
        
        try:
            expected_evidence = question_data['expected_evidence']
            expected_papers = expected_evidence.get('paper_ids', [])
            
            # Use semantic search to find similar papers
            search_results = self.service_factory.retrieval_handler.search_similar_papers(
                query_text=question,
                top_k=10,
                use_hybrid=True
            )
            
            # Extract paper IDs from search results
            retrieved_papers = []
            similarity_scores = []
            
            if search_results:
                for result in search_results:
                    if hasattr(result, 'paper_id'):
                        retrieved_papers.append(result.paper_id)
                        if hasattr(result, 'similarity_score'):
                            similarity_scores.append(result.similarity_score)
                    elif isinstance(result, dict):
                        if 'id' in result:
                            retrieved_papers.append(result['id'])
                        if 'similarity_score' in result:
                            similarity_scores.append(result['similarity_score'])
            
            # Calculate metrics
            precision, recall, f1 = self._calculate_metrics(retrieved_papers, expected_papers)
            
            # Generate AI response if deepseek client is available
            ai_response = None
            ai_generation_time = 0.0
            ai_response_similarity = 0.0
            expected_ai_response = expected_evidence.get('expected_ai_response', '')
            
            if self.service_factory and hasattr(self.service_factory, 'deepseek_client') and self.service_factory.deepseek_client:
                ai_start_time = time.time()
                try:
                    # Create context from search results
                    context = f"Query: {question}\nRetrieved {len(retrieved_papers)} relevant papers"
                    if search_results:
                        # Add paper titles/abstracts if available
                        context += "\nTop results:"
                        for i, result in enumerate(search_results[:3]):
                            if hasattr(result, 'title'):
                                context += f"\n{i+1}. {result.title}"
                            elif isinstance(result, dict) and 'title' in result:
                                context += f"\n{i+1}. {result['title']}"
                    
                    ai_response = self.service_factory.deepseek_client.generate_content(
                        prompt=f"Based on the semantic search results, provide a comprehensive answer to: {question}",
                        system_prompt=f"You are an AI assistant analyzing academic papers. Use the provided search results to answer the question comprehensively. Context: {context}"
                    )
                    ai_generation_time = time.time() - ai_start_time
                    
                    # Calculate AI response similarity if expected response exists
                    if expected_ai_response and ai_response:
                        ai_response_similarity = self._calculate_text_similarity(ai_response, expected_ai_response)
                        
                except Exception as e:
                    logger.warning(f"AI response generation failed for {question_id}: {e}")
                    ai_generation_time = time.time() - ai_start_time
            
            response_time = time.time() - start_time
            
            return MockEvaluationResult(
                question_id=question_id,
                question=question,
                question_type='semantic',
                category=question_data['category'],
                success=True,
                response_time=response_time,
                retrieved_papers=retrieved_papers,
                expected_papers=expected_papers,
                precision=precision,
                recall=recall,
                f1_score=f1,
                ai_response=ai_response,
                expected_ai_response=expected_ai_response,
                ai_response_similarity=ai_response_similarity,
                ai_generation_time=ai_generation_time,
                similarity_scores=similarity_scores
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Error evaluating semantic question {question_id}: {e}")
            
            return MockEvaluationResult(
                question_id=question_id,
                question=question,
                question_type='semantic',
                category=question_data['category'],
                success=False,
                response_time=response_time,
                retrieved_papers=[],
                expected_papers=expected_evidence.get('paper_ids', []),
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                error_message=str(e)
            )

    def _evaluate_as_semantic_fallback(self, question_data: Dict, start_time: float) -> MockEvaluationResult:
        """Fallback evaluation using semantic search for graph questions."""
        question_id = question_data['id']
        question = question_data['question']
        expected_evidence = question_data['expected_evidence']
        
        try:
            # Use hybrid search as fallback
            search_results = self.service_factory.retrieval_handler.search_similar_papers(
                query_text=question,
                top_k=5,
                use_hybrid=True
            )
            
            retrieved_papers = []
            if search_results:
                for result in search_results:
                    if hasattr(result, 'paper_id'):
                        retrieved_papers.append(result.paper_id)
                    elif isinstance(result, dict) and 'id' in result:
                        retrieved_papers.append(result['id'])
            
            expected_papers = expected_evidence.get('paper_ids', [])
            precision, recall, f1 = self._calculate_metrics(retrieved_papers, expected_papers)
            
            response_time = time.time() - start_time
            
            return MockEvaluationResult(
                question_id=question_id,
                question=question,
                question_type='graph',
                category=question_data['category'],
                success=True,
                response_time=response_time,
                retrieved_papers=retrieved_papers,
                expected_papers=expected_papers,
                precision=precision,
                recall=recall,
                f1_score=f1
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return MockEvaluationResult(
                question_id=question_id,
                question=question,
                question_type='graph',
                category=question_data['category'],
                success=False,
                response_time=response_time,
                retrieved_papers=[],
                expected_papers=expected_evidence.get('paper_ids', []),
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                error_message=str(e)
            )

    def _calculate_metrics(self, retrieved: List[str], expected: List[str]) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        if not expected:
            return 1.0 if not retrieved else 0.0, 1.0, 1.0 if not retrieved else 0.0
        
        if not retrieved:
            return 0.0, 0.0, 0.0
        
        # Convert to sets for intersection
        retrieved_set = set(retrieved)
        expected_set = set(expected)
        
        # Calculate intersection
        intersection = retrieved_set.intersection(expected_set)
        
        # Calculate metrics
        precision = len(intersection) / len(retrieved_set) if retrieved_set else 0.0
        recall = len(intersection) / len(expected_set) if expected_set else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using simple word overlap."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity (can be enhanced with embeddings)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def run_evaluation(self, limit: Optional[int] = None) -> List[MockEvaluationResult]:
        """Run evaluation on all mock questions."""
        questions = self.load_mock_data()
        
        if not questions:
            logger.error("No mock questions loaded")
            return []
        
        if limit:
            questions = questions[:limit]
            
        logger.info(f"Starting evaluation on {len(questions)} questions")
        
        results = []
        
        for i, question_data in enumerate(questions):
            logger.info(f"Evaluating question {i+1}/{len(questions)}: {question_data['id']}")
            
            if question_data['type'] == 'graph':
                result = self.evaluate_graph_question(question_data)
            else:  # semantic
                result = self.evaluate_semantic_question(question_data)
            
            results.append(result)
            
            # Log progress
            if result.success:
                logger.info(f"✅ {result.question_id}: P={result.precision:.3f}, R={result.recall:.3f}, F1={result.f1_score:.3f}")
            else:
                logger.warning(f"❌ {result.question_id}: Failed - {result.error_message}")
        
        self.results = results
        return results

    def generate_summary(self, results: List[MockEvaluationResult]) -> MockEvaluationSummary:
        """Generate evaluation summary from results."""
        if not results:
            return MockEvaluationSummary(0, 0, 0, 0.0, 0.0, 0.0, 0.0, {}, {}, {})
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        # Overall metrics
        total_questions = len(results)
        successful_questions = len(successful_results)
        failed_questions = len(failed_results)
        avg_response_time = sum(r.response_time for r in results) / len(results)
        
        # Performance metrics (only for successful questions)
        if successful_results:
            overall_precision = sum(r.precision for r in successful_results) / len(successful_results)
            overall_recall = sum(r.recall for r in successful_results) / len(successful_results)
            overall_f1 = sum(r.f1_score for r in successful_results) / len(successful_results)
            
            # AI response metrics
            ai_responses_with_similarity = [r for r in successful_results if r.ai_response_similarity is not None]
            avg_ai_response_similarity = sum(r.ai_response_similarity for r in ai_responses_with_similarity) / len(ai_responses_with_similarity) if ai_responses_with_similarity else 0.0
            
            ai_responses_with_time = [r for r in successful_results if r.ai_generation_time is not None]
            avg_ai_generation_time = sum(r.ai_generation_time for r in ai_responses_with_time) / len(ai_responses_with_time) if ai_responses_with_time else 0.0
            
            ai_successful_responses = [r for r in successful_results if r.ai_response is not None]
            ai_response_success_rate = len(ai_successful_responses) / len(successful_results) if successful_results else 0.0
        else:
            overall_precision = overall_recall = overall_f1 = 0.0
            avg_ai_response_similarity = avg_ai_generation_time = ai_response_success_rate = 0.0
        
        # Performance by type
        graph_results = [r for r in successful_results if r.question_type == 'graph']
        semantic_results = [r for r in successful_results if r.question_type == 'semantic']
        
        graph_performance = {}
        if graph_results:
            graph_performance = {
                'precision': sum(r.precision for r in graph_results) / len(graph_results),
                'recall': sum(r.recall for r in graph_results) / len(graph_results),
                'f1': sum(r.f1_score for r in graph_results) / len(graph_results),
                'avg_response_time': sum(r.response_time for r in graph_results) / len(graph_results),
                'success_rate': len(graph_results) / len([r for r in results if r.question_type == 'graph'])
            }
        
        semantic_performance = {}
        if semantic_results:
            semantic_performance = {
                'precision': sum(r.precision for r in semantic_results) / len(semantic_results),
                'recall': sum(r.recall for r in semantic_results) / len(semantic_results),
                'f1': sum(r.f1_score for r in semantic_results) / len(semantic_results),
                'avg_response_time': sum(r.response_time for r in semantic_results) / len(semantic_results),
                'success_rate': len(semantic_results) / len([r for r in results if r.question_type == 'semantic'])
            }
        
        # Performance by category
        category_breakdown = {}
        categories = set(r.category for r in results)
        for category in categories:
            category_results = [r for r in successful_results if r.category == category]
            if category_results:
                category_breakdown[category] = {
                    'precision': sum(r.precision for r in category_results) / len(category_results),
                    'recall': sum(r.recall for r in category_results) / len(category_results),
                    'f1': sum(r.f1_score for r in category_results) / len(category_results),
                    'count': len(category_results),
                    'success_rate': len(category_results) / len([r for r in results if r.category == category])
                }
        
        return MockEvaluationSummary(
            total_questions=total_questions,
            successful_questions=successful_questions,
            failed_questions=failed_questions,
            avg_response_time=avg_response_time,
            overall_precision=overall_precision,
            overall_recall=overall_recall,
            overall_f1=overall_f1,
            avg_ai_response_similarity=avg_ai_response_similarity,
            avg_ai_generation_time=avg_ai_generation_time,
            ai_response_success_rate=ai_response_success_rate,
            graph_performance=graph_performance,
            semantic_performance=semantic_performance,
            category_breakdown=category_breakdown
        )

    def generate_detailed_report(self, results: List[MockEvaluationResult], 
                                summary: MockEvaluationSummary) -> str:
        """Generate detailed evaluation report."""
        report = []
        report.append("# Mock Data Evaluation Report")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Questions:** {summary.total_questions}")
        report.append("")
        
        # Overall Performance
        report.append("## Overall Performance")
        report.append(f"- **Success Rate:** {summary.successful_questions}/{summary.total_questions} ({summary.successful_questions/summary.total_questions*100:.1f}%)")
        report.append(f"- **Average Response Time:** {summary.avg_response_time:.3f}s")
        report.append(f"- **Overall Precision:** {summary.overall_precision:.3f}")
        report.append(f"- **Overall Recall:** {summary.overall_recall:.3f}")
        report.append(f"- **Overall F1 Score:** {summary.overall_f1:.3f}")
        report.append("")
        
        # AI Response Performance
        report.append("## AI Response Performance")
        report.append(f"- **AI Response Success Rate:** {summary.ai_response_success_rate*100:.1f}%")
        report.append(f"- **Average AI Generation Time:** {summary.avg_ai_generation_time:.3f}s")
        report.append(f"- **Average AI Response Similarity:** {summary.avg_ai_response_similarity:.3f}")
        report.append("")
        
        # Performance by Type
        report.append("## Performance by Question Type")
        
        if summary.graph_performance:
            report.append("### Graph Questions")
            gp = summary.graph_performance
            report.append(f"- **Success Rate:** {gp.get('success_rate', 0)*100:.1f}%")
            report.append(f"- **Precision:** {gp.get('precision', 0):.3f}")
            report.append(f"- **Recall:** {gp.get('recall', 0):.3f}")
            report.append(f"- **F1 Score:** {gp.get('f1', 0):.3f}")
            report.append(f"- **Avg Response Time:** {gp.get('avg_response_time', 0):.3f}s")
            report.append("")
        
        if summary.semantic_performance:
            report.append("### Semantic Questions")
            sp = summary.semantic_performance
            report.append(f"- **Success Rate:** {sp.get('success_rate', 0)*100:.1f}%")
            report.append(f"- **Precision:** {sp.get('precision', 0):.3f}")
            report.append(f"- **Recall:** {sp.get('recall', 0):.3f}")
            report.append(f"- **F1 Score:** {sp.get('f1', 0):.3f}")
            report.append(f"- **Avg Response Time:** {sp.get('avg_response_time', 0):.3f}s")
            report.append("")
        
        # Performance by Category
        if summary.category_breakdown:
            report.append("## Performance by Category")
            for category, metrics in summary.category_breakdown.items():
                report.append(f"### {category.replace('_', ' ').title()}")
                report.append(f"- **Questions:** {metrics['count']}")
                report.append(f"- **Success Rate:** {metrics['success_rate']*100:.1f}%")
                report.append(f"- **Precision:** {metrics['precision']:.3f}")
                report.append(f"- **Recall:** {metrics['recall']:.3f}")
                report.append(f"- **F1 Score:** {metrics['f1']:.3f}")
                report.append("")
        
        # Failed Questions
        failed_results = [r for r in results if not r.success]
        if failed_results:
            report.append("## Failed Questions")
            for result in failed_results:
                report.append(f"- **{result.question_id}** ({result.question_type}/{result.category}): {result.error_message}")
            report.append("")
        
        # Top Performing Questions
        successful_results = [r for r in results if r.success]
        if successful_results:
            top_f1_results = sorted(successful_results, key=lambda x: x.f1_score, reverse=True)[:5]
            report.append("## Top Performing Questions (by F1 Score)")
            for result in top_f1_results:
                report.append(f"- **{result.question_id}** (F1: {result.f1_score:.3f}): {result.question[:60]}...")
            report.append("")
        
        return "\n".join(report)

    def save_results(self, results: List[MockEvaluationResult], 
                    summary: MockEvaluationSummary, 
                    output_dir: str = "./data") -> Dict[str, str]:
        """Save evaluation results and report to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results as JSON
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_questions': summary.total_questions,
                'successful_questions': summary.successful_questions,
                'failed_questions': summary.failed_questions,
                'avg_response_time': summary.avg_response_time,
                'overall_precision': summary.overall_precision,
                'overall_recall': summary.overall_recall,
                'overall_f1': summary.overall_f1,
                'graph_performance': summary.graph_performance,
                'semantic_performance': summary.semantic_performance,
                'category_breakdown': summary.category_breakdown
            },
            'detailed_results': [
                {
                    'question_id': r.question_id,
                    'question': r.question,
                    'question_type': r.question_type,
                    'category': r.category,
                    'success': r.success,
                    'response_time': r.response_time,
                    'retrieved_papers': r.retrieved_papers,
                    'expected_papers': r.expected_papers,
                    'precision': r.precision,
                    'recall': r.recall,
                    'f1_score': r.f1_score,
                    'error_message': r.error_message,
                    'similarity_scores': r.similarity_scores
                }
                for r in results
            ]
        }
        
        results_file = os.path.join(output_dir, f"mock_evaluation_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # Save report as markdown
        report = self.generate_detailed_report(results, summary)
        report_file = os.path.join(output_dir, f"mock_evaluation_report_{timestamp}.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return {
            'results_file': results_file,
            'report_file': report_file
        }