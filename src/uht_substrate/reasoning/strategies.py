"""Query routing strategies for the reasoning engine."""

import re
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

    # Storage intent (check first - explicit user request)
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

    # Comparison intent (check before "why" catches these as infer)
    # More specific patterns first
    compare_patterns = [
        "compare",
        "difference between",
        "different from",
        "similar to",
        "why is",  # "why is X similar to Y" is comparison, not inference
        "why are",
        "how is",  # "how is X different from Y"
        "how are",
        "how does",
        "how do",
        " vs ",
        " vs.",
        "versus",
        "relationship between",
        "more like",
        "closer to",
    ]
    # Check if query contains comparison patterns AND mentions two things
    if any(p in query_lower for p in compare_patterns):
        # Additional check: does it look like it's comparing two things?
        two_things = re.search(
            r"(?:compare|similar|different|like|vs|versus|between|and|to|from|with)\s+(?:a |an |the )?[a-z]+",
            query_lower,
        )
        if two_things or "between" in query_lower or " vs" in query_lower:
            return QueryIntent.COMPARE

    # Classification intent
    classify_patterns = [
        "what is a ",
        "what is an ",
        "what is the ",
        "what is ",
        "what are ",
        "define ",
        "classify ",
        "describe ",
        "tell me about ",
        "explain what ",
    ]
    if any(query_lower.startswith(p) or f" {p}" in query_lower for p in classify_patterns):
        return QueryIntent.CLASSIFY

    # Category membership questions are also classification-like
    # "Is X a Y?" or "Is X alive?"
    if query_lower.startswith("is a ") or query_lower.startswith("is an "):
        return QueryIntent.CLASSIFY

    # "Can X do Y?" questions about capability
    if query_lower.startswith("can a ") or query_lower.startswith("can an "):
        return QueryIntent.INFER

    # Inference intent (more specific patterns)
    infer_patterns = [
        "why does",
        "why do",
        "why would",
        "why can",
        "how come",
        "explain why",
        "what makes",
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
        "what else",
    ]
    if any(p in query_lower for p in explore_patterns):
        return QueryIntent.EXPLORE

    # Disambiguation intent
    disambiguate_patterns = [
        "which meaning",
        "which sense",
        "do you mean",
        "clarify",
        "ambiguous",
        "senses of",
        "meanings of",
    ]
    if any(p in query_lower for p in disambiguate_patterns):
        return QueryIntent.DISAMBIGUATE

    # Default to general (will fall back to classification if entity found)
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
