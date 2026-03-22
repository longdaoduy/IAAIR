import math
import re
from models.entities.retrievals.SearchResult import SearchResult
from models.entities.retrievals.AttributionSpan import AttributionSpan
from typing import Dict, List, Optional


class ResultFusion:
    """Convert hybrid search results into SearchResult objects with confidence scores.

    The hybrid results combine graph and vector search (multi-modal vector search
    with cross-modal re-ranking finds paper IDs → graph query enriches with metadata).

    Confidence scoring:
        - multimodal_confidence: from the vector+visual re-ranking score (SciBERT + CLIP)
        - graph_confidence: computed from Neo4j metadata completeness and citation impact
        - hybrid_confidence: weighted combination of multimodal + graph confidence
    """

    # Weights for combining confidence scores into hybrid_confidence
    MULTIMODAL_WEIGHT = 0.3
    GRAPH_WEIGHT = 0.7

    @staticmethod
    def build_attributions(ai_response: str, fused_results: List[SearchResult]) -> List[SearchResult]:
        """Build attribution spans by matching AI response text to source papers.

        Scans the AI response for references to each paper's title, authors,
        or bracket citations (e.g. [1], [2]) and populates the `attributions`
        field on each matching SearchResult.

        Args:
            ai_response: The generated AI answer text
            fused_results: SearchResult objects from fusion step

        Returns:
            The same fused_results list with `attributions` populated
        """
        if not ai_response or not fused_results:
            return fused_results

        response_lower = ai_response.lower()

        for idx, result in enumerate(fused_results):
            spans = []

            # 1. Title match — find if the paper's title appears in the AI response
            title = (result.title or '').strip()
            if title and len(title) > 10:
                # Try exact substring match (case-insensitive)
                title_lower = title.lower()
                pos = response_lower.find(title_lower)
                if pos != -1:
                    spans.append(AttributionSpan(
                        text=ai_response[pos:pos + len(title)],
                        source_id=result.paper_id,
                        source_type='paper',
                        confidence=0.95,
                        char_start=pos,
                        char_end=pos + len(title),
                        supporting_passages=[title],
                    ))
                else:
                    # Try partial title match (first 5+ significant words)
                    title_words = [w for w in title_lower.split() if len(w) > 3]
                    if len(title_words) >= 3:
                        # Check if 60%+ of significant title words appear in response
                        matched_words = [w for w in title_words if w in response_lower]
                        if len(matched_words) >= len(title_words) * 0.6:
                            spans.append(AttributionSpan(
                                text=f"Partial title match: {', '.join(matched_words[:5])}",
                                source_id=result.paper_id,
                                source_type='paper',
                                confidence=0.6,
                                char_start=0,
                                char_end=0,
                                supporting_passages=[title],
                            ))

            # 2. Author match — check if any author name appears in the response
            for author in (result.authors or []):
                if not author or len(author) < 3:
                    continue
                author_lower = author.lower()
                pos = response_lower.find(author_lower)
                if pos != -1:
                    spans.append(AttributionSpan(
                        text=ai_response[pos:pos + len(author)],
                        source_id=result.paper_id,
                        source_type='paper',
                        confidence=0.85,
                        char_start=pos,
                        char_end=pos + len(author),
                        supporting_passages=[f"Author: {author}"],
                    ))
                    break  # one author match per paper is enough

            # 3. Bracket citation match — e.g. [1], [2] in the AI response
            bracket_pattern = re.compile(r'\[' + str(idx + 1) + r'\]')
            for m in bracket_pattern.finditer(ai_response):
                spans.append(AttributionSpan(
                    text=m.group(),
                    source_id=result.paper_id,
                    source_type='paper',
                    confidence=0.90,
                    char_start=m.start(),
                    char_end=m.end(),
                    supporting_passages=[f"Citation reference [{idx + 1}]"],
                ))

            # 4. Abstract content match — check if key phrases from the abstract
            # appear in the response (indicates the LLM used this paper's content)
            abstract = (result.abstract or '').strip()
            if abstract and len(abstract) > 50:
                # Extract key phrases (3+ word sequences)
                abstract_lower = abstract.lower()
                # Find 4-word windows that appear in both abstract and response
                abstract_words = abstract_lower.split()
                for i in range(len(abstract_words) - 3):
                    phrase = ' '.join(abstract_words[i:i + 4])
                    if len(phrase) > 15 and phrase in response_lower:
                        pos = response_lower.find(phrase)
                        spans.append(AttributionSpan(
                            text=ai_response[pos:pos + len(phrase)],
                            source_id=result.paper_id,
                            source_type='abstract',
                            confidence=0.75,
                            char_start=pos,
                            char_end=pos + len(phrase),
                            supporting_passages=[phrase],
                        ))
                        break  # one content match per paper is enough

            # Deduplicate spans by char_start
            seen_starts = set()
            unique_spans = []
            for s in spans:
                if s.char_start not in seen_starts:
                    seen_starts.add(s.char_start)
                    unique_spans.append(s)

            result.attributions = unique_spans

            # Update source_path if attributions found
            if unique_spans and 'attributed' not in result.source_path:
                result.source_path.append('attributed')

        return fused_results

    @staticmethod
    def _compute_graph_confidence(result: Dict) -> float:
        """Compute graph confidence from Neo4j metadata quality.

        Measures how rich and complete the graph data is for this paper.
        A paper with authors, venue, DOI, abstract, and high citations
        gets a higher graph confidence than one with sparse metadata.

        Returns:
            float in [0, 1]
        """
        score = 0.0

        # Has title (basic completeness)
        if result.get('title'):
            score += 0.15

        # Has abstract
        if result.get('abstract'):
            score += 0.15

        # Has authors (more authors = richer graph connections)
        authors = result.get('authors', []) or []
        if authors:
            score += min(0.20, 0.05 * len(authors))  # up to 0.20 for 4+ authors

        # Has venue (published in a known venue)
        if result.get('venue'):
            score += 0.15

        # Has DOI (verified publication)
        if result.get('doi'):
            score += 0.10

        # Has publication date
        if result.get('publication_date'):
            score += 0.05

        # Citation impact (log-scaled, up to 0.20)
        cited_by = result.get('cited_by_count', 0) or 0
        if cited_by > 0:
            # log10(1) = 0, log10(10) = 1, log10(100) = 2, log10(1000) = 3
            citation_score = min(0.20, 0.05 * math.log10(1 + cited_by))
            score += citation_score

        return min(score, 1.0)

    def fuse_results(self, hybrid_results: List[Dict],
                     visual_data: Optional[Dict] = None) -> List[SearchResult]:
        """Convert hybrid search results to SearchResult objects with confidence scoring.

        Computes three confidence dimensions:
        - multimodal_confidence: re-ranking score from vector + visual search
        - graph_confidence: metadata completeness from Neo4j
        - hybrid_confidence: weighted sum of multimodal + graph

        Args:
            hybrid_results: Results from the hybrid search pipeline (already
                            ranked by multi-modal vector search + graph enrichment)
            visual_data:    Optional cross-modal visual search data containing:
                            - multimodal_scores: {paper_id: float} combined vector+visual scores
                            - paper_visual_scores: {paper_id: float} visual-only scores
                            - figure_results: list of matched figures
                            - table_results:  list of matched tables
        """
        # Extract visual/multimodal data
        multimodal_scores = {}
        paper_visual_scores = {}
        figure_results = []
        table_results = []
        if visual_data:
            multimodal_scores = visual_data.get('multimodal_scores', {})
            paper_visual_scores = visual_data.get('paper_visual_scores', {})
            figure_results = visual_data.get('figure_results', [])
            table_results = visual_data.get('table_results', [])

        # Build per-paper visual evidence lookup
        paper_visual_evidence: Dict[str, list] = {}
        paper_fig_count: Dict[str, int] = {}
        paper_tab_count: Dict[str, int] = {}
        for r in figure_results:
            pid = r.get('paper_id')
            if pid:
                paper_visual_evidence.setdefault(pid, []).append(r)
                paper_fig_count[pid] = paper_fig_count.get(pid, 0) + 1
        for r in table_results:
            pid = r.get('paper_id')
            if pid:
                paper_visual_evidence.setdefault(pid, []).append(r)
                paper_tab_count[pid] = paper_tab_count.get(pid, 0) + 1

        # Convert hybrid results to SearchResult objects (preserving existing order)
        fused_results = []
        for result in hybrid_results:
            paper_id = result.get('paper_id') or result.get('id')
            if not paper_id:
                continue

            # ── Confidence scores ──
            # Multimodal confidence: pre-computed similarity score from
            # HybridRetrievalHandler (already higher=better).
            # Falls back to the per-result similarity_score injected by the handler,
            # or to the multimodal_scores map from visual_data.
            multimodal_conf = multimodal_scores.get(paper_id, 0.0)
            # Clamp to [0, 1] — boosted scores for graph/requested papers may exceed 1.0
            multimodal_conf = min(max(multimodal_conf, 0.0), 1.0)

            # Graph confidence: from Neo4j metadata quality (0-1 range)
            graph_conf = self._compute_graph_confidence(result)

            # Visual-only score (for display)
            v_score = paper_visual_scores.get(paper_id, 0.0)

            # Hybrid confidence: weighted combination
            hybrid_conf = result.get('similarity_score', 0.0)

            # Use hybrid_confidence as the relevance_score for sorting/display
            relevance_score = min(hybrid_conf, 1.0)

            # Determine source path
            source_path = ['hybrid_search']
            if multimodal_conf > 0:
                source_path.append('multimodal_search')
            if v_score > 0:
                source_path.append('visual_search')

            # Trim visual_evidence to essential fields for response size
            trimmed_evidence = []
            for ve in paper_visual_evidence.get(paper_id, [])[:6]:
                trimmed_evidence.append({
                    'id': ve.get('id', ''),
                    'description': (ve.get('description', '') or '')[:200],
                    'similarity_score': ve.get('similarity_score', 0.0),
                    'collection': ve.get('collection', ''),
                    'search_type': ve.get('search_type', ''),
                })

            search_result = SearchResult(
                paper_id=paper_id,
                title=result.get('title') or 'Unknown Title',
                abstract=result.get('abstract'),
                authors=result.get('authors', []) or [],
                venue=result.get('venue'),
                publication_date=result.get('publication_date'),
                doi=result.get('doi'),
                cited_by_count=result.get('cited_by_count', 0) or 0,
                relevance_score=relevance_score,
                vector_score=multimodal_conf,  # multimodal (vector+visual) score
                graph_score=graph_conf,         # graph metadata quality score
                visual_score=v_score,
                matched_figures=paper_fig_count.get(paper_id, 0),
                matched_tables=paper_tab_count.get(paper_id, 0),
                visual_evidence=trimmed_evidence,
                source_path=source_path,
                attributions=[],
                confidence_scores={
                    'hybrid_confidence': round(relevance_score, 4),
                    'multimodal_confidence': round(multimodal_conf, 4),
                    'graph_confidence': round(graph_conf, 4),
                    'visual_confidence': round(v_score, 4),
                }
            )
            fused_results.append(search_result)

        # Boost relevance_score for explicitly requested paper IDs so they
        # always appear at the top of results regardless of multimodal score.
        requested_pids = set()
        if visual_data:
            requested_pids = set(visual_data.get('requested_paper_ids', []))
        if requested_pids:
            best_score = max((r.relevance_score for r in fused_results), default=1.0)
            boost = best_score + 1.0  # guarantee above all other scores
            for r in fused_results:
                if r.paper_id in requested_pids:
                    r.relevance_score = r.relevance_score + boost

        # Re-sort by hybrid_confidence (descending) since graph_confidence
        # may change the relative ordering from the original pipeline
        fused_results.sort(key=lambda r: r.relevance_score, reverse=True)

        return fused_results