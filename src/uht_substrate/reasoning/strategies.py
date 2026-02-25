"""Query routing strategies for the reasoning engine."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .context import AssembledContext


class QueryIntent(Enum):
    """Types of user query intents."""

    CLASSIFY = "classify"  # What is X?
    COMPARE = "compare"  # How do X and Y relate?
    INFER = "infer"  # What can we conclude about X?
    EXPLORE = "explore"  # What's connected to X?
    DISAMBIGUATE = "disambiguate"  # Which X do you mean?
    STORE = "store"  # Remember this fact
    GENERAL = "general"  # General question


@dataclass
class QueryStrategy:
    """A strategy for handling a query."""

    name: str
    requires_uht: bool
    requires_local_graph: bool = True
    requires_inference: bool = False
    description: str = ""


# Available strategies
STRATEGIES = {
    "use_local": QueryStrategy(
        name="use_local",
        requires_uht=False,
        requires_local_graph=True,
        description="Use locally cached classification",
    ),
    "query_uht": QueryStrategy(
        name="query_uht",
        requires_uht=True,
        requires_local_graph=True,
        description="Query UHT Factory for fresh classification",
    ),
    "classify_then_compare": QueryStrategy(
        name="classify_then_compare",
        requires_uht=True,
        requires_local_graph=True,
        requires_inference=True,
        description="Classify both entities then compare",
    ),
    "local_inference": QueryStrategy(
        name="local_inference",
        requires_uht=False,
        requires_local_graph=True,
        requires_inference=True,
        description="Use local facts and axioms for inference",
    ),
    "axiom_inference": QueryStrategy(
        name="axiom_inference",
        requires_uht=False,
        requires_local_graph=True,
        requires_inference=True,
        description="Apply trait axioms to derive properties",
    ),
    "local_graph_traversal": QueryStrategy(
        name="local_graph_traversal",
        requires_uht=False,
        requires_local_graph=True,
        description="Traverse local knowledge graph",
    ),
    "uht_neighborhood": QueryStrategy(
        name="uht_neighborhood",
        requires_uht=True,
        requires_local_graph=True,
        description="Explore neighborhood via UHT API",
    ),
    "uht_disambiguation": QueryStrategy(
        name="uht_disambiguation",
        requires_uht=True,
        requires_local_graph=False,
        description="Use UHT dictionary for disambiguation",
    ),
    "store_fact": QueryStrategy(
        name="store_fact",
        requires_uht=False,
        requires_local_graph=True,
        description="Store fact in local graph",
    ),
    "general": QueryStrategy(
        name="general",
        requires_uht=True,
        requires_local_graph=True,
        requires_inference=True,
        description="General reasoning with all resources",
    ),
}


def analyze_intent(query: str) -> QueryIntent:
    """
    Analyze a query to determine user intent.

    Args:
        query: The user's query

    Returns:
        Detected query intent
    """
    query_lower = query.lower()

    # Classification intent
    classify_patterns = [
        "what is",
        "what are",
        "define",
        "classify",
        "describe",
        "tell me about",
        "explain what",
    ]
    if any(p in query_lower for p in classify_patterns):
        return QueryIntent.CLASSIFY

    # Comparison intent
    compare_patterns = [
        "compare",
        "difference between",
        "different from",
        "similar to",
        "vs",
        "versus",
        "how do .* relate",
        "relationship between",
    ]
    if any(p in query_lower for p in compare_patterns):
        return QueryIntent.COMPARE

    # Inference intent
    infer_patterns = [
        "why",
        "how come",
        "explain why",
        "reason",
        "because",
        "conclude",
        "infer",
        "deduce",
    ]
    if any(p in query_lower for p in infer_patterns):
        return QueryIntent.INFER

    # Exploration intent
    explore_patterns = [
        "related to",
        "connected to",
        "associated with",
        "similar things",
        "like this",
        "neighbors",
        "explore",
    ]
    if any(p in query_lower for p in explore_patterns):
        return QueryIntent.EXPLORE

    # Disambiguation intent
    disambiguate_patterns = [
        "which",
        "do you mean",
        "clarify",
        "ambiguous",
        "sense of",
        "meaning of",
    ]
    if any(p in query_lower for p in disambiguate_patterns):
        return QueryIntent.DISAMBIGUATE

    # Storage intent
    store_patterns = [
        "remember",
        "store",
        "save",
        "note that",
        "keep in mind",
        "i prefer",
        "my favorite",
    ]
    if any(p in query_lower for p in store_patterns):
        return QueryIntent.STORE

    # Default to classify for entity-focused queries
    return QueryIntent.GENERAL


class StrategySelector:
    """Selects appropriate reasoning strategy based on intent and context."""

    def select(
        self,
        intent: QueryIntent,
        context: AssembledContext,
        force_refresh: bool = False,
    ) -> QueryStrategy:
        """
        Select the best strategy for handling a query.

        Args:
            intent: The detected query intent
            context: Assembled context
            force_refresh: Force fresh data from UHT

        Returns:
            Selected query strategy
        """
        if intent == QueryIntent.CLASSIFY:
            return self._select_classify_strategy(context, force_refresh)
        elif intent == QueryIntent.COMPARE:
            return STRATEGIES["classify_then_compare"]
        elif intent == QueryIntent.INFER:
            return self._select_infer_strategy(context)
        elif intent == QueryIntent.EXPLORE:
            return self._select_explore_strategy(context)
        elif intent == QueryIntent.DISAMBIGUATE:
            return STRATEGIES["uht_disambiguation"]
        elif intent == QueryIntent.STORE:
            return STRATEGIES["store_fact"]
        else:
            return STRATEGIES["general"]

    def _select_classify_strategy(
        self,
        context: AssembledContext,
        force_refresh: bool,
    ) -> QueryStrategy:
        """Select strategy for classification queries."""
        if force_refresh:
            return STRATEGIES["query_uht"]

        # Check if we have fresh cached entity
        if context.entities:
            # Could add freshness check here
            return STRATEGIES["use_local"]

        return STRATEGIES["query_uht"]

    def _select_infer_strategy(self, context: AssembledContext) -> QueryStrategy:
        """Select strategy for inference queries."""
        # If we have sufficient local facts, try local inference first
        if context.has_sufficient_facts(min_facts=3):
            return STRATEGIES["local_inference"]

        # Otherwise use axiom-based inference
        return STRATEGIES["axiom_inference"]

    def _select_explore_strategy(self, context: AssembledContext) -> QueryStrategy:
        """Select strategy for exploration queries."""
        # If we have local entities, try graph traversal first
        if context.entities:
            return STRATEGIES["local_graph_traversal"]

        return STRATEGIES["uht_neighborhood"]
