"""Context assembly for reasoning operations."""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

from ..config.logging import get_logger
from ..graph.repository import StoredEntity, StoredFact

if TYPE_CHECKING:
    from ..graph.repository import GraphRepository

logger = get_logger(__name__)


@dataclass
class EntityContext:
    """Context about an entity."""

    uuid: str
    name: str
    hex_code: str
    source: str


@dataclass
class AssembledContext:
    """Assembled context for reasoning."""

    query: str
    facts: list[StoredFact] = field(default_factory=list)
    entities: list[EntityContext] = field(default_factory=list)
    preferences: dict[str, str] = field(default_factory=dict)
    recent_traces: list[str] = field(default_factory=list)

    def as_string(self) -> str:
        """Format context as string for API calls."""
        parts = []

        if self.facts:
            parts.append("Known facts:")
            for f in self.facts[:5]:  # Limit to top 5
                parts.append(f"  - {f.subject} {f.predicate} {f.object}")

        if self.entities:
            parts.append("Relevant entities:")
            for e in self.entities[:3]:
                parts.append(f"  - {e.name} ({e.hex_code})")

        if self.preferences:
            parts.append("User preferences:")
            for k, v in list(self.preferences.items())[:3]:
                parts.append(f"  - {k}: {v}")

        return "\n".join(parts) if parts else ""

    def has_entity(self, name: str) -> bool:
        """Check if context has an entity by name."""
        name_lower = name.lower()
        return any(e.name.lower() == name_lower for e in self.entities)

    def get_entity(self, name: str) -> Optional[EntityContext]:
        """Get entity from context by name."""
        name_lower = name.lower()
        for e in self.entities:
            if e.name.lower() == name_lower:
                return e
        return None

    def has_sufficient_facts(self, min_facts: int = 3) -> bool:
        """Check if we have enough facts for local reasoning."""
        return len(self.facts) >= min_facts

    @property
    def entity_uuids(self) -> list[str]:
        """Get UUIDs of all entities in context."""
        return [e.uuid for e in self.entities]


def calculate_fact_relevance(
    fact: StoredFact,
    query: str,
    current_time: datetime,
    relevance_window_hours: int = 168,
) -> float:
    """
    Calculate relevance score for a fact.

    Args:
        fact: The fact to score
        query: The current query
        current_time: Current timestamp
        relevance_window_hours: Time window for decay

    Returns:
        Relevance score (0-1)
    """
    # Temporal decay
    age = current_time - fact.created_at
    age_hours = age.total_seconds() / 3600
    min_weight = 0.3
    temporal_weight = max(min_weight, 1.0 - (age_hours / relevance_window_hours))

    # Content relevance (simple keyword matching)
    query_lower = query.lower()
    content_score = 0.0

    if fact.subject.lower() in query_lower:
        content_score += 0.5
    if fact.object.lower() in query_lower:
        content_score += 0.3
    if fact.predicate.lower() in query_lower:
        content_score += 0.2

    # Ensure minimum content score
    content_score = max(0.1, content_score)

    return temporal_weight * content_score * fact.confidence


class ContextAssembler:
    """Assembles context from knowledge graph for reasoning."""

    def __init__(
        self,
        graph: "GraphRepository",
        relevance_window_hours: int = 168,
    ):
        """
        Initialize the context assembler.

        Args:
            graph: Graph repository for lookups
            relevance_window_hours: Time window for relevance decay
        """
        self._graph = graph
        self._relevance_window_hours = relevance_window_hours

    async def build(
        self,
        query: str,
        user_id: Optional[str] = None,
        additional_context: Optional[str] = None,
        max_facts: int = 10,
        max_entities: int = 5,
    ) -> AssembledContext:
        """
        Build context from various sources.

        Args:
            query: The user's query
            user_id: Optional user ID for personalized context
            additional_context: Additional context string
            max_facts: Maximum facts to include
            max_entities: Maximum entities to include

        Returns:
            Assembled context
        """
        context = AssembledContext(query=query)
        current_time = datetime.utcnow()

        # 1. Extract potential entity names from query
        potential_entities = self._extract_entities(query)
        if additional_context:
            potential_entities.extend(self._extract_entities(additional_context))

        # 2. Look up entities in local graph
        for name in potential_entities[:10]:  # Limit extraction
            entity = await self._graph.find_entity_by_name(name)
            if entity:
                context.entities.append(
                    EntityContext(
                        uuid=entity.uuid,
                        name=entity.name,
                        hex_code=entity.hex_code,
                        source=entity.source,
                    )
                )
                if len(context.entities) >= max_entities:
                    break

        # 3. Get user facts if user_id provided
        if user_id:
            user_facts = await self._graph.get_user_facts(user_id, limit=50)

            # Score and sort facts by relevance
            scored_facts = [
                (
                    f,
                    calculate_fact_relevance(
                        f, query, current_time, self._relevance_window_hours
                    ),
                )
                for f in user_facts
            ]
            scored_facts.sort(key=lambda x: x[1], reverse=True)

            context.facts = [f for f, _ in scored_facts[:max_facts]]

            # Get preferences
            context.preferences = await self._graph.get_user_preferences(user_id)

            # Get recent reasoning traces
            traces = await self._graph.get_recent_traces(hours=24, limit=3)
            context.recent_traces = [t.uuid for t in traces]

        logger.debug(
            "Built context",
            query_length=len(query),
            entity_count=len(context.entities),
            fact_count=len(context.facts),
        )

        return context

    def _extract_entities(self, text: str) -> list[str]:
        """
        Extract potential entity names from text.

        Args:
            text: Text to extract from

        Returns:
            List of potential entity names
        """
        entities: list[str] = []

        # Look for quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted)

        # Look for "what is X" patterns
        what_is = re.findall(
            r"what (?:is|are) (?:a |an |the )?([a-z][a-z\s]+?)(?:\?|$|,|\.|;)",
            text.lower(),
        )
        entities.extend([w.strip() for w in what_is])

        # Look for capitalized phrases (proper nouns)
        capitalized = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
        entities.extend(capitalized)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for e in entities:
            e_lower = e.lower().strip()
            if e_lower and e_lower not in seen:
                seen.add(e_lower)
                unique.append(e.strip())

        return unique
