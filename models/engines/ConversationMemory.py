"""
Conversation Memory for multi-turn RAG interactions.

Maintains per-session conversation history so follow-up queries can
reference prior context (coreference resolution, topic continuity).

Each session stores the last N turns (query + AI response summary)
and provides a formatted context block for the LLM prompt.
"""

import time
import logging
from typing import Dict, List, Optional
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Maximum number of turns to keep per session
MAX_TURNS_PER_SESSION = 10

# Maximum sessions before evicting the oldest
MAX_SESSIONS = 500

# Session timeout in seconds (30 minutes)
SESSION_TIMEOUT_SEC = 30 * 60


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    query: str
    ai_response_summary: str  # first 300 chars of AI response
    template_key: Optional[str] = None
    paper_ids: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class ConversationMemory:
    """Session-based conversation memory for multi-turn RAG.

    Usage:
        memory = ConversationMemory()

        # Before generating AI response, get prior context
        history = memory.get_context(session_id)

        # After generating AI response, record the turn
        memory.add_turn(session_id, query, ai_response, template_key, paper_ids)
    """

    def __init__(self, max_turns: int = MAX_TURNS_PER_SESSION,
                 max_sessions: int = MAX_SESSIONS,
                 session_timeout: float = SESSION_TIMEOUT_SEC):
        self._sessions: OrderedDict[str, List[ConversationTurn]] = OrderedDict()
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._session_timeout = session_timeout

    def _evict_expired(self):
        """Remove sessions that have been idle too long."""
        now = time.time()
        expired = [
            sid for sid, turns in self._sessions.items()
            if turns and (now - turns[-1].timestamp) > self._session_timeout
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.debug(f"Evicted {len(expired)} expired conversation sessions")

    def _ensure_capacity(self):
        """Evict oldest sessions if capacity exceeded."""
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)  # remove oldest

    def add_turn(self, session_id: str, query: str, ai_response: Optional[str],
                 template_key: Optional[str] = None,
                 paper_ids: Optional[List[str]] = None):
        """Record a conversation turn.

        Args:
            session_id: Client session identifier
            query: User's search query
            ai_response: AI-generated answer (will be summarized to 300 chars)
            template_key: Graph template used for this query
            paper_ids: Paper IDs returned in this turn
        """
        if not session_id:
            return

        self._evict_expired()

        if session_id not in self._sessions:
            self._ensure_capacity()
            self._sessions[session_id] = []

        summary = (ai_response or "")[:300]
        turn = ConversationTurn(
            query=query,
            ai_response_summary=summary,
            template_key=template_key,
            paper_ids=(paper_ids or [])[:10],  # keep top 10 IDs
        )

        turns = self._sessions[session_id]
        turns.append(turn)

        # Trim to max turns
        if len(turns) > self._max_turns:
            self._sessions[session_id] = turns[-self._max_turns:]

        # Move to end (most recently used)
        self._sessions.move_to_end(session_id)

        logger.debug(
            f"Conversation turn recorded: session={session_id[:8]}... "
            f"turns={len(self._sessions[session_id])}"
        )

    def get_context(self, session_id: str, max_turns: int = 3) -> str:
        """Get formatted conversation history for LLM prompt injection.

        Returns a string block summarizing the last N turns, or empty string
        if no prior conversation exists.

        Args:
            session_id: Client session identifier
            max_turns: Maximum number of prior turns to include

        Returns:
            Formatted conversation context string (may be empty)
        """
        if not session_id or session_id not in self._sessions:
            return ""

        self._evict_expired()

        turns = self._sessions.get(session_id, [])
        if not turns:
            return ""

        # Take the most recent turns (exclude the current one being processed)
        recent = turns[-max_turns:]

        lines = ["Prior conversation context:"]
        for i, turn in enumerate(recent, 1):
            lines.append(f"  Turn {i}: User asked: \"{turn.query}\"")
            if turn.ai_response_summary:
                lines.append(f"    AI answered: {turn.ai_response_summary}")
            if turn.paper_ids:
                lines.append(f"    Papers discussed: {', '.join(turn.paper_ids[:5])}")

        return "\n".join(lines)

    def get_prior_paper_ids(self, session_id: str, max_turns: int = 3) -> List[str]:
        """Get paper IDs from recent conversation turns.

        Useful for resolving references like "that paper" or "those results".
        """
        if not session_id or session_id not in self._sessions:
            return []

        turns = self._sessions.get(session_id, [])
        pids = []
        seen = set()
        for turn in reversed(turns[-max_turns:]):
            for pid in turn.paper_ids:
                if pid not in seen:
                    seen.add(pid)
                    pids.append(pid)
        return pids

    def get_history(self, session_id: str) -> List[Dict]:
        """Get conversation history as a list of dicts for API response.

        Args:
            session_id: Client session identifier

        Returns:
            List of turn dicts with query, ai_response, template_key,
            paper_ids, and timestamp for each turn.
        """
        if not session_id or session_id not in self._sessions:
            return []

        self._evict_expired()

        turns = self._sessions.get(session_id, [])
        return [
            {
                "turn": i,
                "query": turn.query,
                "ai_response": turn.ai_response_summary,
                "template_key": turn.template_key,
                "paper_ids": turn.paper_ids,
                "timestamp": turn.timestamp,
            }
            for i, turn in enumerate(turns, 1)
        ]

    def get_session_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)

    def get_total_turns(self) -> int:
        """Total conversation turns across all sessions."""
        return sum(len(t) for t in self._sessions.values())

    def clear_session(self, session_id: str):
        """Clear a specific session's history."""
        self._sessions.pop(session_id, None)

    def clear_all(self):
        """Clear all conversation memory."""
        self._sessions.clear()
        logger.info("All conversation memory cleared")
