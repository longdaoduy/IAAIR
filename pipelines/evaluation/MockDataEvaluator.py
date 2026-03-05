"""
Mock Data Evaluation Pipeline

This module evaluates the system's performance on the mock evaluation dataset
containing 50 questions (25 graph-based, 25 semantic-based) derived from
enriched OpenAlex papers data.
"""

import json
import logging
import math
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
    dcg_at_5: Optional[float] = None
    dcg_at_10: Optional[float] = None
    ndcg_at_5: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    verification_labels: Optional[List[str]] = None


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
    avg_dcg_at_5: float
    avg_dcg_at_10: float
    avg_ndcg_at_5: float
    avg_ndcg_at_10: float
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

    # async def evaluate_graph_question(self, question_data: Dict) -> MockEvaluationResult:
    #     """Evaluate a graph-based question using Cypher queries."""
    #     start_time = time.time()
    #     question_id = question_data['id']
    #     question = question_data['question']
    #
    #     try:
    #         expected_evidence = question_data['expected_evidence']
    #         expected_papers = expected_evidence.get('paper_ids', [])
    #
    #         # Use graph query handler to execute the query
    #         result = await self.service_factory.retrieval_handler.execute_graph_search(question, 10)
    #
    #         # Extract paper IDs from result if available
    #         retrieved_papers = []
    #         if result:
    #             retrieved_papers.extend([r.get('paper_id') for r in result[:]])
    #
    #         # Calculate metrics
    #         precision, recall, f1 = self._calculate_metrics(retrieved_papers, expected_papers)
    #
    #         # Calculate DCG metrics
    #         dcg_at_5 = self._calculate_dcg_at_k(retrieved_papers, expected_papers, 5)
    #         dcg_at_10 = self._calculate_dcg_at_k(retrieved_papers, expected_papers, 10)
    #         ndcg_at_5 = self._calculate_ndcg_at_k(retrieved_papers, expected_papers, 5)
    #         ndcg_at_10 = self._calculate_ndcg_at_k(retrieved_papers, expected_papers, 10)
    #
    #         # Generate AI response using the retrieval handler's generate_ai_response function
    #         ai_response = None
    #         ai_generation_time = 0.0
    #         ai_response_similarity = 0.0
    #         verification_labels = None
    #         expected_ai_response = expected_evidence.get('expected_ai_response', '')
    #
    #         if self.service_factory and self.service_factory.retrieval_handler:
    #             ai_start_time = time.time()
    #             try:
    #                 # Convert graph results to search results format
    #                 search_results = []
    #                 if result and 'data' in result:
    #                     for record in result['data']:
    #                         # Convert graph record to search result format
    #                         paper_info = {
    #                             'paper_id': record.get('paper_id', ''),
    #                             'title': record.get('title', ''),
    #                             'abstract': record.get('abstract', ''),
    #                             'authors': record.get('authors', []),
    #                             'venue': record.get('venue', ''),
    #                             'publication_date': record.get('publication_date', ''),
    #                             'relevance_score': 0.8  # High relevance for graph results
    #                         }
    #                         search_results.append(paper_info)
    #
    #                 # Use generate_ai_response function with STRUCTURAL query type for graph questions
    #                 from models.entities.retrieval.QueryType import QueryType
    #                 ai_response = await self.service_factory.retrieval_handler.generate_ai_response(
    #                     query=question,
    #                     search_results=search_results,
    #                     query_type=QueryType.STRUCTURAL
    #                 )
    #
    #                 # Perform SciFact verification and store results
    #                 if ai_response and search_results:
    #                     try:
    #                         verification_results = await self.service_factory.retrieval_handler.verify_claims_scifact(
    #                             ai_response, search_results
    #                         )
    #                         if verification_results:
    #                             verification_labels = [v.get('label', 'UNKNOWN') for v in verification_results]
    #                     except Exception as e:
    #                         logger.warning(f"SciFact verification failed for {question_id}: {e}")
    #
    #                 ai_generation_time = time.time() - ai_start_time
    #
    #                 # Calculate AI response similarity if expected response exists
    #                 if expected_ai_response and ai_response:
    #                     ai_response_similarity = self._calculate_text_similarity(ai_response, expected_ai_response)
    #
    #             except Exception as e:
    #                 logger.warning(f"AI response generation failed for {question_id}: {e}")
    #                 ai_generation_time = time.time() - ai_start_time
    #
    #         response_time = time.time() - start_time
    #
    #         return MockEvaluationResult(
    #             question_id=question_id,
    #             question=question,
    #             question_type='graph',
    #             category=question_data['category'],
    #             success=True,
    #             response_time=response_time,
    #             retrieved_papers=retrieved_papers,
    #             expected_papers=expected_papers,
    #             precision=precision,
    #             recall=recall,
    #             f1_score=f1,
    #             ai_response=ai_response,
    #             expected_ai_response=expected_ai_response,
    #             ai_response_similarity=ai_response_similarity,
    #             ai_generation_time=ai_generation_time,
    #             dcg_at_5=dcg_at_5,
    #             dcg_at_10=dcg_at_10,
    #             ndcg_at_5=ndcg_at_5,
    #             ndcg_at_10=ndcg_at_10,
    #             verification_labels=verification_labels
    #         )
    #
    #     except Exception as e:
    #         response_time = time.time() - start_time
    #         logger.error(f"Error evaluating graph question {question_id}: {e}")
    #
    #         return MockEvaluationResult(
    #             question_id=question_id,
    #             question=question,
    #             question_type='graph',
    #             category=question_data['category'],
    #             success=False,
    #             response_time=response_time,
    #             retrieved_papers=[],
    #             expected_papers=expected_evidence.get('paper_ids', []),
    #             precision=0.0,
    #             recall=0.0,
    #             f1_score=0.0,
    #             error_message=str(e)
    #         )

    async def evaluate_semantic_question(self, question_data: Dict) -> MockEvaluationResult:
        """Evaluate a semantic question using vector similarity search."""
        start_time = time.time()
        question_id = question_data['id']
        question = question_data['question']

        try:
            expected_evidence = question_data['expected_evidence']
            expected_papers = expected_evidence.get('paper_ids', [])

            # Use semantic search to find similar papers
            if question_data['type'] == 'graph':
                search_results = await self.service_factory.retrieval_handler.execute_graph_search(
                    query=question,
                    top_k=10,
                )
            else:
                search_results = await self.service_factory.retrieval_handler.execute_vector_search(
                    query=question,
                    top_k=10,
                )

            # Extract paper IDs from search results
            retrieved_papers = []
            similarity_scores = []

            if search_results:
                retrieved_papers.extend([r.get('paper_id') for r in search_results[:]])
                similarity_scores.extend([r.get('similarity_score') for r in search_results[:]])

            # Calculate metrics
            precision, recall, f1 = self._calculate_metrics(retrieved_papers, expected_papers)

            # Calculate DCG metrics
            dcg_at_5 = self._calculate_dcg_at_k(retrieved_papers, expected_papers, 5)
            dcg_at_10 = self._calculate_dcg_at_k(retrieved_papers, expected_papers, 10)
            ndcg_at_5 = self._calculate_ndcg_at_k(retrieved_papers, expected_papers, 5)
            ndcg_at_10 = self._calculate_ndcg_at_k(retrieved_papers, expected_papers, 10)

            # Generate AI response using the retrieval handler's generate_ai_response function
            ai_response = None
            ai_generation_time = 0.0
            ai_response_similarity = 0.0
            verification_labels = None
            expected_ai_response = expected_evidence.get('expected_ai_response', '')

            if self.service_factory and self.service_factory.retrieval_handler:
                ai_start_time = time.time()
                try:
                    # Use generate_ai_response function with SEMANTIC query type for semantic questions
                    from models.entities.retrieval.QueryType import QueryType
                    ai_response = await self.service_factory.retrieval_handler.generate_ai_response(
                        query=question,
                        search_results=search_results,
                        query_type=QueryType.SEMANTIC
                    )
                    
                    # Perform SciFact verification and store results
                    if ai_response and search_results:
                        try:
                            # Convert search results to proper format if needed
                            papers_for_verification = []
                            for result in search_results:
                                if isinstance(result, dict):
                                    papers_for_verification.append(result)
                                elif hasattr(result, '__dict__'):
                                    papers_for_verification.append(vars(result))
                                    
                            if papers_for_verification:
                                verification_results = await self.service_factory.retrieval_handler.verify_claims_scifact(
                                    ai_response, papers_for_verification
                                )
                                if verification_results:
                                    verification_labels = [v.get('label', 'UNKNOWN') for v in verification_results]
                        except Exception as e:
                            logger.warning(f"SciFact verification failed for {question_id}: {e}")
                    
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
                question_type=question_data['type'],
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
                similarity_scores=similarity_scores,
                dcg_at_5=dcg_at_5,
                dcg_at_10=dcg_at_10,
                ndcg_at_5=ndcg_at_5,
                ndcg_at_10=ndcg_at_10,
                verification_labels=verification_labels
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

    async def _evaluate_as_semantic_fallback(self, question_data: Dict, start_time: float) -> MockEvaluationResult:
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

    @staticmethod
    def _calculate_metrics(retrieved: List[str], expected: List[str]) -> Tuple[float, float, float]:
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

    def _calculate_dcg_at_k(self, retrieved: List[str], expected: List[str], k: int) -> float:
        """Calculate Discounted Cumulative Gain at k.
        
        Args:
            retrieved: List of retrieved paper IDs in ranked order
            expected: List of expected/relevant paper IDs
            k: Cut-off rank
            
        Returns:
            DCG@k score
        """
        if not retrieved or not expected:
            return 0.0

        expected_set = set(expected)
        dcg = 0.0

        for i, paper_id in enumerate(retrieved[:k]):
            if paper_id in expected_set:
                # Binary relevance: 1 if relevant, 0 otherwise
                relevance = 1.0
                # DCG formula: rel_i / log2(i + 2) where i is 0-indexed
                dcg += relevance / math.log2(i + 2)

        return dcg

    def _calculate_ideal_dcg_at_k(self, expected: List[str], k: int) -> float:
        """Calculate Ideal DCG at k (maximum possible DCG).
        
        Args:
            expected: List of expected/relevant paper IDs
            k: Cut-off rank
            
        Returns:
            IDCG@k score
        """
        if not expected:
            return 0.0

        # For binary relevance, ideal ranking has all relevant items first
        num_relevant = min(len(expected), k)
        idcg = 0.0

        for i in range(num_relevant):
            idcg += 1.0 / math.log2(i + 2)

        return idcg

    def _calculate_ndcg_at_k(self, retrieved: List[str], expected: List[str], k: int) -> float:
        """Calculate Normalized DCG at k.
        
        Args:
            retrieved: List of retrieved paper IDs in ranked order
            expected: List of expected/relevant paper IDs
            k: Cut-off rank
            
        Returns:
            NDCG@k score (0-1, where 1 is perfect)
        """
        dcg = self._calculate_dcg_at_k(retrieved, expected, k)
        idcg = self._calculate_ideal_dcg_at_k(expected, k)

        return dcg / idcg if idcg > 0 else 0.0

    async def run_evaluation(self, limit: Optional[int] = None) -> List[MockEvaluationResult]:
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
            logger.info(f"Evaluating question {i + 1}/{len(questions)}: {question_data['id']}")

            if question_data['type'] == 'parallel':
                continue
                # result = await self.evaluate_graph_question(question_data)
            else:  # semantic
                result = await self.evaluate_semantic_question(question_data)

            results.append(result)

            # Log progress
            if result.success:
                logger.info(
                    f"✅ {result.question_id}: P={result.precision:.3f}, R={result.recall:.3f}, F1={result.f1_score:.3f}")
            else:
                logger.warning(f"❌ {result.question_id}: Failed - {result.error_message}")

        self.results = results

        # Save AI responses table
        self._save_ai_responses_table(results)

        return results

    def _save_ai_responses_table(self, results: List[MockEvaluationResult]) -> str:
        """Save a table with query, AI response, and expected AI response."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Prepare data for CSV table
        table_data = []
        headers = ['Question_ID', 'Query', 'AI_Response', 'Expected_AI_Response', 'Response_Similarity',
                   'Question_Type', 'Success', 'SciFact_Labels']

        for result in results:
            ai_response = result.ai_response if result.ai_response else "No AI response generated"
            expected_response = result.expected_ai_response if result.expected_ai_response else "No expected response available"
            similarity = f"{result.ai_response_similarity:.3f}" if result.ai_response_similarity is not None else "N/A"
            
            # Format verification labels
            scifact_labels = "No verification"
            if result.verification_labels:
                # Count different label types
                label_counts = {}
                for label in result.verification_labels:
                    clean_label = str(label).strip().upper()
                    if 'SUPPORTED' in clean_label:
                        label_counts['SUPPORTED'] = label_counts.get('SUPPORTED', 0) + 1
                    elif 'CONTRADICTED' in clean_label:
                        label_counts['CONTRADICTED'] = label_counts.get('CONTRADICTED', 0) + 1
                    elif 'NO_EVIDENCE' in clean_label:
                        label_counts['NO_EVIDENCE'] = label_counts.get('NO_EVIDENCE', 0) + 1
                    else:
                        label_counts['UNKNOWN'] = label_counts.get('UNKNOWN', 0) + 1
                
                # Format as readable summary
                label_parts = []
                for label_type, count in label_counts.items():
                    label_parts.append(f"{label_type}:{count}")
                scifact_labels = ", ".join(label_parts) if label_parts else "No labels"

            table_data.append([
                result.question_id,
                result.question,
                ai_response,
                expected_response,
                similarity,
                result.question_type,
                "Yes" if result.success else "No",
                scifact_labels
            ])

        # Create output directory if it doesn't exist
        output_dir = "./data"
        os.makedirs(output_dir, exist_ok=True)

        # Save as CSV
        csv_file = os.path.join(output_dir, f"ai_responses_comparison_{timestamp}.csv")
        import csv
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(table_data)

        # Save as formatted markdown table
        md_file = os.path.join(output_dir, f"ai_responses_comparison_{timestamp}.md")
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# AI Responses Comparison Table\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Markdown table headers
            f.write("| Question ID | Query | AI Response | Expected AI Response | Similarity | Type | Success | SciFact Labels |\n")
            f.write("|-------------|-------|-------------|---------------------|------------|------|--------|----------------|\n")

            # Markdown table rows
            for row in table_data:
                # Truncate long text for readability
                query = (row[1][:100] + "...") if len(row[1]) > 100 else row[1]
                ai_resp = (row[2][:150] + "...") if len(row[2]) > 150 else row[2]
                expected_resp = (row[3][:150] + "...") if len(row[3]) > 150 else row[3]
                scifact_labels = (row[7][:50] + "...") if len(row[7]) > 50 else row[7]

                # Escape pipe characters in text
                query = query.replace("|", "\\|")
                ai_resp = ai_resp.replace("|", "\\|")
                expected_resp = expected_resp.replace("|", "\\|")
                scifact_labels = scifact_labels.replace("|", "\\|")

                f.write(f"| {row[0]} | {query} | {ai_resp} | {expected_resp} | {row[4]} | {row[5]} | {row[6]} | {scifact_labels} |\n")

        logger.info(f"AI responses comparison saved to: {csv_file} and {md_file}")
        return csv_file

    def generate_summary(self, results: List[MockEvaluationResult]) -> MockEvaluationSummary:
        """Generate evaluation summary from results."""
        if not results:
            return MockEvaluationSummary(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, {}, {})

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
            avg_ai_response_similarity = sum(r.ai_response_similarity for r in ai_responses_with_similarity) / len(
                ai_responses_with_similarity) if ai_responses_with_similarity else 0.0

            ai_responses_with_time = [r for r in successful_results if r.ai_generation_time is not None]
            avg_ai_generation_time = sum(r.ai_generation_time for r in ai_responses_with_time) / len(
                ai_responses_with_time) if ai_responses_with_time else 0.0

            ai_successful_responses = [r for r in successful_results if r.ai_response is not None]
            ai_response_success_rate = len(ai_successful_responses) / len(
                successful_results) if successful_results else 0.0

            # Calculate average DCG metrics
            dcg_results = [r for r in successful_results if r.dcg_at_5 is not None]
            avg_dcg_at_5 = sum(r.dcg_at_5 for r in dcg_results) / len(dcg_results) if dcg_results else 0.0
            avg_dcg_at_10 = sum(r.dcg_at_10 for r in dcg_results) / len(dcg_results) if dcg_results else 0.0
            avg_ndcg_at_5 = sum(r.ndcg_at_5 for r in dcg_results) / len(dcg_results) if dcg_results else 0.0
            avg_ndcg_at_10 = sum(r.ndcg_at_10 for r in dcg_results) / len(dcg_results) if dcg_results else 0.0
        else:
            overall_precision = overall_recall = overall_f1 = 0.0
            avg_ai_response_similarity = avg_ai_generation_time = ai_response_success_rate = 0.0
            avg_dcg_at_5 = avg_dcg_at_10 = avg_ndcg_at_5 = avg_ndcg_at_10 = 0.0

        # Performance by type
        graph_results = [r for r in successful_results if r.question_type == 'graph']
        semantic_results = [r for r in successful_results if r.question_type == 'semantic']

        graph_performance = {}
        if graph_results:
            graph_dcg_results = [r for r in graph_results if r.dcg_at_5 is not None]
            graph_performance = {
                'precision': sum(r.precision for r in graph_results) / len(graph_results),
                'recall': sum(r.recall for r in graph_results) / len(graph_results),
                'f1': sum(r.f1_score for r in graph_results) / len(graph_results),
                'avg_response_time': sum(r.response_time for r in graph_results) / len(graph_results),
                'success_rate': len(graph_results) / len([r for r in results if r.question_type == 'graph']),
                'dcg_at_5': sum(r.dcg_at_5 for r in graph_dcg_results) / len(
                    graph_dcg_results) if graph_dcg_results else 0.0,
                'dcg_at_10': sum(r.dcg_at_10 for r in graph_dcg_results) / len(
                    graph_dcg_results) if graph_dcg_results else 0.0,
                'ndcg_at_5': sum(r.ndcg_at_5 for r in graph_dcg_results) / len(
                    graph_dcg_results) if graph_dcg_results else 0.0,
                'ndcg_at_10': sum(r.ndcg_at_10 for r in graph_dcg_results) / len(
                    graph_dcg_results) if graph_dcg_results else 0.0
            }

        semantic_performance = {}
        if semantic_results:
            semantic_dcg_results = [r for r in semantic_results if r.dcg_at_5 is not None]
            semantic_performance = {
                'precision': sum(r.precision for r in semantic_results) / len(semantic_results),
                'recall': sum(r.recall for r in semantic_results) / len(semantic_results),
                'f1': sum(r.f1_score for r in semantic_results) / len(semantic_results),
                'avg_response_time': sum(r.response_time for r in semantic_results) / len(semantic_results),
                'success_rate': len(semantic_results) / len([r for r in results if r.question_type == 'semantic']),
                'dcg_at_5': sum(r.dcg_at_5 for r in semantic_dcg_results) / len(
                    semantic_dcg_results) if semantic_dcg_results else 0.0,
                'dcg_at_10': sum(r.dcg_at_10 for r in semantic_dcg_results) / len(
                    semantic_dcg_results) if semantic_dcg_results else 0.0,
                'ndcg_at_5': sum(r.ndcg_at_5 for r in semantic_dcg_results) / len(
                    semantic_dcg_results) if semantic_dcg_results else 0.0,
                'ndcg_at_10': sum(r.ndcg_at_10 for r in semantic_dcg_results) / len(
                    semantic_dcg_results) if semantic_dcg_results else 0.0
            }

        # Performance by category
        category_breakdown = {}
        categories = set(r.category for r in results)
        for category in categories:
            category_results = [r for r in successful_results if r.category == category]
            if category_results:
                category_dcg_results = [r for r in category_results if r.dcg_at_5 is not None]
                category_breakdown[category] = {
                    'precision': sum(r.precision for r in category_results) / len(category_results),
                    'recall': sum(r.recall for r in category_results) / len(category_results),
                    'f1': sum(r.f1_score for r in category_results) / len(category_results),
                    'count': len(category_results),
                    'success_rate': len(category_results) / len([r for r in results if r.category == category]),
                    'dcg_at_5': sum(r.dcg_at_5 for r in category_dcg_results) / len(
                        category_dcg_results) if category_dcg_results else 0.0,
                    'ndcg_at_5': sum(r.ndcg_at_5 for r in category_dcg_results) / len(
                        category_dcg_results) if category_dcg_results else 0.0
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
            avg_dcg_at_5=avg_dcg_at_5,
            avg_dcg_at_10=avg_dcg_at_10,
            avg_ndcg_at_5=avg_ndcg_at_5,
            avg_ndcg_at_10=avg_ndcg_at_10,
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
        report.append(
            f"- **Success Rate:** {summary.successful_questions}/{summary.total_questions} ({summary.successful_questions / summary.total_questions * 100:.1f}%)")
        report.append(f"- **Average Response Time:** {summary.avg_response_time:.3f}s")
        report.append(f"- **Overall Precision:** {summary.overall_precision:.3f}")
        report.append(f"- **Overall Recall:** {summary.overall_recall:.3f}")
        report.append(f"- **Overall F1 Score:** {summary.overall_f1:.3f}")
        report.append(f"- **Average DCG@5:** {summary.avg_dcg_at_5:.3f}")
        report.append(f"- **Average DCG@10:** {summary.avg_dcg_at_10:.3f}")
        report.append(f"- **Average NDCG@5:** {summary.avg_ndcg_at_5:.3f}")
        report.append(f"- **Average NDCG@10:** {summary.avg_ndcg_at_10:.3f}")
        report.append("")

        # AI Response Performance
        report.append("## AI Response Performance")
        report.append(f"- **AI Response Success Rate:** {summary.ai_response_success_rate * 100:.1f}%")
        report.append(f"- **Average AI Generation Time:** {summary.avg_ai_generation_time:.3f}s")
        report.append(f"- **Average AI Response Similarity:** {summary.avg_ai_response_similarity:.3f}")
        report.append("")

        # Performance by Type
        report.append("## Performance by Question Type")

        if summary.graph_performance:
            report.append("### Graph Questions")
            gp = summary.graph_performance
            report.append(f"- **Success Rate:** {gp.get('success_rate', 0) * 100:.1f}%")
            report.append(f"- **Precision:** {gp.get('precision', 0):.3f}")
            report.append(f"- **Recall:** {gp.get('recall', 0):.3f}")
            report.append(f"- **F1 Score:** {gp.get('f1', 0):.3f}")
            report.append(f"- **DCG@5:** {gp.get('dcg_at_5', 0):.3f}")
            report.append(f"- **NDCG@5:** {gp.get('ndcg_at_5', 0):.3f}")
            report.append(f"- **Avg Response Time:** {gp.get('avg_response_time', 0):.3f}s")
            report.append("")

        if summary.semantic_performance:
            report.append("### Semantic Questions")
            sp = summary.semantic_performance
            report.append(f"- **Success Rate:** {sp.get('success_rate', 0) * 100:.1f}%")
            report.append(f"- **Precision:** {sp.get('precision', 0):.3f}")
            report.append(f"- **Recall:** {sp.get('recall', 0):.3f}")
            report.append(f"- **F1 Score:** {sp.get('f1', 0):.3f}")
            report.append(f"- **DCG@5:** {sp.get('dcg_at_5', 0):.3f}")
            report.append(f"- **NDCG@5:** {sp.get('ndcg_at_5', 0):.3f}")
            report.append(f"- **Avg Response Time:** {sp.get('avg_response_time', 0):.3f}s")
            report.append("")

        # Performance by Category
        if summary.category_breakdown:
            report.append("## Performance by Category")
            for category, metrics in summary.category_breakdown.items():
                report.append(f"### {category.replace('_', ' ').title()}")
                report.append(f"- **Questions:** {metrics['count']}")
                report.append(f"- **Success Rate:** {metrics['success_rate'] * 100:.1f}%")
                report.append(f"- **Precision:** {metrics['precision']:.3f}")
                report.append(f"- **Recall:** {metrics['recall']:.3f}")
                report.append(f"- **F1 Score:** {metrics['f1']:.3f}")
                report.append("")

        # Failed Questions
        failed_results = [r for r in results if not r.success]
        if failed_results:
            report.append("## Failed Questions")
            for result in failed_results:
                report.append(
                    f"- **{result.question_id}** ({result.question_type}/{result.category}): {result.error_message}")
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
                    'dcg_at_5': r.dcg_at_5,
                    'dcg_at_10': r.dcg_at_10,
                    'ndcg_at_5': r.ndcg_at_5,
                    'ndcg_at_10': r.ndcg_at_10,
                    'error_message': r.error_message,
                    'similarity_scores': r.similarity_scores,
                    'verification_labels': r.verification_labels
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
