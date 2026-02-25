"""Main reasoning engine orchestrator."""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from ..config.logging import get_logger
from ..config.settings import get_settings
from ..graph.repository import GraphRepository, StoredEntity
from ..priors.inference import InferredProperty, PriorInferenceEngine
from ..uht_client.client import UHTClient
from ..uht_client.models import ClassificationResult, Entity
from .context import AssembledContext, ContextAssembler
from .strategies import QueryIntent, QueryStrategy, StrategySelector, analyze_intent
from .trace import ReasoningTrace, ReasoningTraceBuilder

logger = get_logger(__name__)


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""

    answer: str
    confidence: float
    trace_id: str
    sources: list[str] = field(default_factory=list)
    inferred_properties: list[dict[str, Any]] = field(default_factory=list)
    entities_used: list[str] = field(default_factory=list)
    hex_codes: dict[str, str] = field(default_factory=dict)


class ReasoningEngine:
    """
    Main orchestrator for reasoning operations.

    Coordinates between:
    - Local knowledge graph (Neo4j)
    - UHT Factory API (ground truth)
    - Priors system (axioms, ontology, heuristics)
    """

    def __init__(
        self,
        graph: GraphRepository,
        uht: UHTClient,
        inference: Optional[PriorInferenceEngine] = None,
    ):
        """
        Initialize the reasoning engine.

        Args:
            graph: Graph repository for local knowledge
            uht: UHT Factory API client
            inference: Prior inference engine
        """
        self._graph = graph
        self._uht = uht
        self._inference = inference or PriorInferenceEngine()
        self._strategy_selector = StrategySelector()
        self._context_assembler = ContextAssembler(graph)
        self._settings = get_settings()

    async def reason(
        self,
        query: str,
        user_id: Optional[str] = None,
        additional_context: Optional[str] = None,
        force_refresh: bool = False,
    ) -> ReasoningResult:
        """
        Main entry point for reasoning about a query.

        Args:
            query: The user's query
            user_id: Optional user ID for personalized context
            additional_context: Additional context string
            force_refresh: Force fresh data from UHT

        Returns:
            ReasoningResult with answer and metadata
        """
        trace = ReasoningTraceBuilder(query)

        # 1. Determine query intent
        intent = analyze_intent(query)
        trace.log(f"Query intent: {intent.value}", action="analyze")

        # 2. Build context from user history and graph
        context = await self._context_assembler.build(
            query=query,
            user_id=user_id,
            additional_context=additional_context,
        )
        trace.log(
            f"Context: {len(context.entities)} entities, {len(context.facts)} facts",
            action="context",
        )

        # 3. Select strategy
        strategy = self._strategy_selector.select(intent, context, force_refresh)
        trace.set_strategy(strategy.name)
        trace.log(f"Strategy: {strategy.name}", action="strategy")

        # 4. Execute strategy
        result = await self._execute_strategy(strategy, intent, query, context, trace, user_id)

        # 5. Store reasoning trace
        if user_id:
            completed_trace = trace.complete(result.answer, result.confidence)
            await self._store_trace(completed_trace, user_id)
        else:
            trace.complete(result.answer, result.confidence)

        result.trace_id = trace.id
        return result

    async def _execute_strategy(
        self,
        strategy: QueryStrategy,
        intent: QueryIntent,
        query: str,
        context: AssembledContext,
        trace: ReasoningTraceBuilder,
        user_id: Optional[str],
    ) -> ReasoningResult:
        """Execute the selected reasoning strategy."""

        if intent == QueryIntent.CLASSIFY:
            return await self._classify(query, context, trace)
        elif intent == QueryIntent.COMPARE:
            return await self._compare(query, context, trace)
        elif intent == QueryIntent.INFER:
            return await self._infer(query, context, trace)
        elif intent == QueryIntent.EXPLORE:
            return await self._explore(query, context, trace)
        elif intent == QueryIntent.DISAMBIGUATE:
            return await self._disambiguate(query, context, trace)
        elif intent == QueryIntent.STORE:
            return await self._store(query, context, trace, user_id)
        else:
            return await self._general(query, context, trace)

    async def _classify(
        self,
        query: str,
        context: AssembledContext,
        trace: ReasoningTraceBuilder,
    ) -> ReasoningResult:
        """Handle classification queries."""
        # Extract entity name from query
        entity_name = self._extract_entity_name(query)
        trace.log(f"Extracting entity: {entity_name}", action="extract")

        # Check local graph first
        local_entity = await self._graph.find_entity_by_name(entity_name)
        classification: Optional[ClassificationResult | Entity] = None

        if local_entity and self._is_fresh(local_entity):
            trace.log_entity_lookup(entity_name, True, local_entity.uuid)
            # Use cached classification
            classification = await self._uht.get_entity(local_entity.uuid)
        else:
            trace.log_entity_lookup(entity_name, False)
            # Query UHT Factory
            trace.log("Querying UHT Factory for classification", action="api")
            classification = await self._uht.classify(entity_name, context=context.as_string())

            # Store in local graph
            await self._graph.upsert_entity(classification, source="uht_factory")
            trace.log_classification(
                entity_name,
                classification.hex_code,
                classification.uuid,
            )

        # Apply trait axioms
        properties = self._inference.infer_properties(
            classification.hex_code,
            entity_name if isinstance(classification, ClassificationResult) else classification.name,
        )
        trace.log(f"Inferred {len(properties)} properties from traits", action="infer")

        for prop in properties[:5]:  # Log first 5
            trace.log_axiom_application(
                prop.source_axiom_name,
                prop.source_axiom_uuid,
                prop.property_name,
                prop.confidence,
            )

        # Format answer
        answer = self._format_classification_answer(classification, properties)

        return ReasoningResult(
            answer=answer,
            confidence=0.9,
            trace_id="",
            sources=["uht_factory" if not local_entity else "local_graph", "axiom_inference"],
            inferred_properties=[
                {
                    "property": p.property_name,
                    "confidence": p.confidence,
                    "source": p.source_axiom_name,
                }
                for p in properties
            ],
            entities_used=[classification.uuid],
            hex_codes={
                entity_name if isinstance(classification, ClassificationResult) else classification.name: classification.hex_code
            },
        )

    async def _compare(
        self,
        query: str,
        context: AssembledContext,
        trace: ReasoningTraceBuilder,
    ) -> ReasoningResult:
        """Handle comparison queries."""
        # Extract two entity names
        entities = self._extract_entity_pair(query)
        if len(entities) < 2:
            return ReasoningResult(
                answer="Please specify two entities to compare.",
                confidence=0.0,
                trace_id="",
            )

        entity_a, entity_b = entities[0], entities[1]
        trace.log(f"Comparing: {entity_a} vs {entity_b}", action="extract")

        # Ensure both are classified
        class_a = await self._ensure_classified(entity_a, context, trace)
        class_b = await self._ensure_classified(entity_b, context, trace)

        # Analyze similarity
        analysis = self._inference.analyze_similarity(
            class_a.hex_code,
            class_b.hex_code,
            entity_a,
            entity_b,
        )

        trace.log_similarity(
            entity_a,
            entity_b,
            analysis.similarity_score,
            analysis.hamming_distance,
        )

        # Check inheritance
        inheritance = self._inference.check_inheritance(
            class_a.hex_code,
            class_b.hex_code,
            entity_a,
            entity_b,
        )

        # Store relationship if similar enough
        if analysis.similarity_score > 0.5:
            await self._graph.create_similar_to_relationship(
                class_a.uuid,
                class_b.uuid,
                analysis.similarity_score,
                analysis.shared_traits,
            )
            trace.log("Created SIMILAR_TO relationship", action="store")

        # Format answer
        answer = self._format_comparison_answer(
            entity_a,
            entity_b,
            class_a.hex_code,
            class_b.hex_code,
            analysis,
            inheritance,
        )

        return ReasoningResult(
            answer=answer,
            confidence=0.85,
            trace_id="",
            sources=["uht_factory", "local_inference"],
            entities_used=[class_a.uuid, class_b.uuid],
            hex_codes={entity_a: class_a.hex_code, entity_b: class_b.hex_code},
        )

    async def _infer(
        self,
        query: str,
        context: AssembledContext,
        trace: ReasoningTraceBuilder,
    ) -> ReasoningResult:
        """Handle inference queries."""
        entity_name = self._extract_entity_name(query)
        trace.log(f"Inferring about: {entity_name}", action="extract")

        # Get or create classification
        classification = await self._ensure_classified(entity_name, context, trace)

        # Apply all axioms
        properties = self._inference.infer_properties(
            classification.hex_code,
            entity_name,
            min_confidence=0.5,  # Include more properties for inference
        )

        # Group by confidence
        certain = [p for p in properties if p.confidence >= 0.9]
        likely = [p for p in properties if 0.7 <= p.confidence < 0.9]
        possible = [p for p in properties if 0.5 <= p.confidence < 0.7]

        trace.log(
            f"Inferred: {len(certain)} certain, {len(likely)} likely, {len(possible)} possible",
            action="infer",
        )

        # Format answer
        answer = self._format_inference_answer(entity_name, certain, likely, possible)

        return ReasoningResult(
            answer=answer,
            confidence=0.8,
            trace_id="",
            sources=["axiom_inference"],
            inferred_properties=[
                {"property": p.property_name, "confidence": p.confidence}
                for p in properties
            ],
            entities_used=[classification.uuid],
            hex_codes={entity_name: classification.hex_code},
        )

    async def _explore(
        self,
        query: str,
        context: AssembledContext,
        trace: ReasoningTraceBuilder,
    ) -> ReasoningResult:
        """Handle exploration queries."""
        entity_name = self._extract_entity_name(query)
        trace.log(f"Exploring neighborhood of: {entity_name}", action="extract")

        # Get classification
        classification = await self._ensure_classified(entity_name, context, trace)

        # Get neighborhood from UHT
        trace.log("Fetching semantic neighborhood from UHT", action="api")
        neighborhood = await self._uht.get_neighborhood(classification.uuid)

        # Also find similar in local graph
        local_similar = await self._graph.find_similar_entities(
            classification.uuid,
            min_shared=20,
            limit=5,
        )
        trace.log(f"Found {len(local_similar)} similar entities locally", action="lookup")

        # Format answer
        answer = self._format_exploration_answer(
            entity_name,
            classification.hex_code,
            neighborhood,
            local_similar,
        )

        return ReasoningResult(
            answer=answer,
            confidence=0.85,
            trace_id="",
            sources=["uht_factory", "local_graph"],
            entities_used=[classification.uuid] + [n.uuid for n in neighborhood.nodes],
            hex_codes={entity_name: classification.hex_code},
        )

    async def _disambiguate(
        self,
        query: str,
        context: AssembledContext,
        trace: ReasoningTraceBuilder,
    ) -> ReasoningResult:
        """Handle disambiguation queries."""
        # Extract word to disambiguate
        word = self._extract_entity_name(query)
        trace.log(f"Disambiguating: {word}", action="extract")

        # Get senses from UHT dictionary
        trace.log("Fetching word senses from UHT", action="api")
        result = await self._uht.disambiguate(word)

        senses = result.senses
        trace.log(f"Found {len(senses)} senses", action="info")

        # Format answer
        answer = self._format_disambiguation_answer(word, senses)

        return ReasoningResult(
            answer=answer,
            confidence=0.9,
            trace_id="",
            sources=["uht_dictionary"],
        )

    async def _store(
        self,
        query: str,
        context: AssembledContext,
        trace: ReasoningTraceBuilder,
        user_id: Optional[str],
    ) -> ReasoningResult:
        """Handle fact storage requests."""
        # Parse fact from query
        fact = self._parse_fact_from_query(query)

        if not fact or not user_id:
            return ReasoningResult(
                answer="I couldn't understand the fact to store. Try: 'Remember that X is Y'",
                confidence=0.0,
                trace_id="",
            )

        subject, predicate, obj = fact
        trace.log(f"Storing fact: {subject} {predicate} {obj}", action="store")

        # Store the fact
        stored = await self._graph.store_fact(
            subject=subject,
            predicate=predicate,
            obj=obj,
            confidence=1.0,
            source="user",
            user_id=user_id,
        )

        trace.log_fact_stored(stored.uuid, subject, predicate, obj)

        return ReasoningResult(
            answer=f"I'll remember that {subject} {predicate} {obj}.",
            confidence=1.0,
            trace_id="",
            sources=["user_input"],
        )

    async def _general(
        self,
        query: str,
        context: AssembledContext,
        trace: ReasoningTraceBuilder,
    ) -> ReasoningResult:
        """Handle general queries."""
        # Try to extract and classify any entities mentioned
        entity_name = self._extract_entity_name(query)

        if entity_name:
            return await self._classify(query, context, trace)

        return ReasoningResult(
            answer="I can help you classify entities, compare them, explore relationships, "
            "or store facts. What would you like to know?",
            confidence=0.5,
            trace_id="",
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _ensure_classified(
        self,
        entity_name: str,
        context: AssembledContext,
        trace: ReasoningTraceBuilder,
    ) -> Entity | ClassificationResult:
        """Ensure an entity is classified, fetching from UHT if needed."""
        local = await self._graph.find_entity_by_name(entity_name)

        if local and self._is_fresh(local):
            trace.log_entity_lookup(entity_name, True, local.uuid)
            return await self._uht.get_entity(local.uuid)

        trace.log_entity_lookup(entity_name, False)
        trace.log(f"Classifying {entity_name} via UHT", action="api")

        classification = await self._uht.classify(entity_name, context=context.as_string())
        await self._graph.upsert_entity(classification, source="uht_factory")

        trace.log_classification(entity_name, classification.hex_code, classification.uuid)
        return classification

    def _is_fresh(self, entity: StoredEntity) -> bool:
        """Check if a cached entity is still fresh."""
        freshness_hours = self._settings.context_relevance_window_hours
        age = datetime.utcnow() - entity.updated_at
        return age < timedelta(hours=freshness_hours)

    def _extract_entity_name(self, query: str) -> str:
        """Extract entity name from a query."""
        # Try quoted strings first
        quoted = re.findall(r'"([^"]+)"', query)
        if quoted:
            return quoted[0]

        # Try "what is X" pattern
        what_is = re.search(
            r"what (?:is|are) (?:a |an |the )?(.+?)(?:\?|$)",
            query,
            re.IGNORECASE,
        )
        if what_is:
            return what_is.group(1).strip()

        # Try "about X" pattern
        about = re.search(r"about (?:a |an |the )?(.+?)(?:\?|$)", query, re.IGNORECASE)
        if about:
            return about.group(1).strip()

        # Fall back to last noun phrase (simple heuristic)
        words = query.split()
        # Remove question words
        filtered = [w for w in words if w.lower() not in {"what", "is", "are", "the", "a", "an", "?"}]
        return " ".join(filtered[-3:]) if filtered else query

    def _extract_entity_pair(self, query: str) -> list[str]:
        """Extract two entity names from a comparison query."""
        # Try "X vs Y" or "X and Y" patterns
        vs_match = re.search(r"(.+?)\s+(?:vs\.?|versus|and|compared to|with)\s+(.+?)(?:\?|$)", query, re.IGNORECASE)
        if vs_match:
            return [vs_match.group(1).strip(), vs_match.group(2).strip()]

        # Try "between X and Y"
        between = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$)", query, re.IGNORECASE)
        if between:
            return [between.group(1).strip(), between.group(2).strip()]

        return []

    def _parse_fact_from_query(self, query: str) -> Optional[tuple[str, str, str]]:
        """Parse a fact from a storage request."""
        # "Remember that X is Y"
        remember = re.search(r"remember (?:that )?(.+?)\s+(?:is|are|has|have)\s+(.+?)(?:\.|$)", query, re.IGNORECASE)
        if remember:
            return (remember.group(1).strip(), "is", remember.group(2).strip())

        # "Note: X = Y"
        note = re.search(r"(?:note|save|store)[:\s]+(.+?)\s*[=:]\s*(.+?)(?:\.|$)", query, re.IGNORECASE)
        if note:
            return (note.group(1).strip(), "is", note.group(2).strip())

        return None

    async def _store_trace(self, trace: ReasoningTrace, user_id: str) -> None:
        """Store a completed reasoning trace."""
        await self._graph.create_reasoning_trace(
            query=trace.query,
            conclusion=trace.conclusion or "",
            strategy=trace.strategy,
            confidence=trace.confidence,
            entity_uuids=trace.entity_uuids,
            axiom_uuids=trace.axiom_uuids,
        )

    # =========================================================================
    # Formatting Methods
    # =========================================================================

    def _format_classification_answer(
        self,
        classification: ClassificationResult | Entity,
        properties: list[InferredProperty],
    ) -> str:
        """Format a classification result as an answer."""
        name = classification.entity if isinstance(classification, ClassificationResult) else classification.name
        hex_code = classification.hex_code

        lines = [
            f"**{name}** is classified as `{hex_code}`",
            "",
        ]

        # Group properties by confidence
        certain = [p for p in properties if p.confidence >= 0.9]
        likely = [p for p in properties if 0.7 <= p.confidence < 0.9]

        if certain:
            lines.append("This entity:")
            for p in certain[:5]:
                lines.append(f"- {p.property_name.replace('_', ' ')}")

        if likely:
            lines.append("")
            lines.append("It likely also:")
            for p in likely[:3]:
                lines.append(f"- {p.property_name.replace('_', ' ')} ({p.confidence:.0%})")

        return "\n".join(lines)

    def _format_comparison_answer(
        self,
        name_a: str,
        name_b: str,
        hex_a: str,
        hex_b: str,
        analysis: Any,
        inheritance: Any,
    ) -> str:
        """Format a comparison result as an answer."""
        lines = [
            f"Comparing **{name_a}** (`{hex_a}`) with **{name_b}** (`{hex_b}`)",
            "",
            f"- Similarity: {analysis.similarity_score:.0%}",
            f"- Hamming distance: {analysis.hamming_distance}",
            f"- Shared traits: {len(analysis.shared_traits)}",
            "",
        ]

        if inheritance.is_valid:
            lines.append(f"{name_a} could be considered a type of {name_b} (inheritance score: {inheritance.inheritance_score:.0%})")
        elif analysis.can_transfer_properties:
            lines.append(f"These entities are similar enough to share many properties.")
        else:
            lines.append(f"These entities are quite different from each other.")

        return "\n".join(lines)

    def _format_inference_answer(
        self,
        entity_name: str,
        certain: list[InferredProperty],
        likely: list[InferredProperty],
        possible: list[InferredProperty],
    ) -> str:
        """Format inference results as an answer."""
        lines = [f"Based on the classification of **{entity_name}**:", ""]

        if certain:
            lines.append("**Certainly:**")
            for p in certain[:5]:
                lines.append(f"- {p.property_name.replace('_', ' ')}")
            lines.append("")

        if likely:
            lines.append("**Likely:**")
            for p in likely[:5]:
                lines.append(f"- {p.property_name.replace('_', ' ')} ({p.confidence:.0%})")
            lines.append("")

        if possible:
            lines.append("**Possibly:**")
            for p in possible[:3]:
                lines.append(f"- {p.property_name.replace('_', ' ')} ({p.confidence:.0%})")

        return "\n".join(lines)

    def _format_exploration_answer(
        self,
        entity_name: str,
        hex_code: str,
        neighborhood: Any,
        local_similar: list[tuple[StoredEntity, int, int]],
    ) -> str:
        """Format exploration results as an answer."""
        lines = [
            f"Semantic neighborhood of **{entity_name}** (`{hex_code}`):",
            "",
        ]

        if neighborhood.nodes:
            lines.append("**Related entities from UHT:**")
            for node in neighborhood.nodes[:5]:
                lines.append(f"- {node.name} (`{node.hex_code}`)")
            lines.append("")

        if local_similar:
            lines.append("**Similar entities in local knowledge:**")
            for entity, shared, distance in local_similar[:5]:
                lines.append(f"- {entity.name} ({shared} shared traits)")

        return "\n".join(lines)

    def _format_disambiguation_answer(self, word: str, senses: list[Any]) -> str:
        """Format disambiguation results as an answer."""
        lines = [f'The word "{word}" has {len(senses)} senses:', ""]

        for i, sense in enumerate(senses[:5], 1):
            lines.append(f"**{i}. {sense.definition}**")
            if sense.hex_code:
                lines.append(f"   UHT: `{sense.hex_code}`")
            if sense.examples:
                lines.append(f"   Example: {sense.examples[0]}")
            lines.append("")

        return "\n".join(lines)
