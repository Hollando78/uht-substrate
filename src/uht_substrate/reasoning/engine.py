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
        # Try to extract entity name
        entity_name = self._extract_entity_name(query)

        # Only fall back to classification if we have a valid-looking entity
        if entity_name and self._is_valid_entity_name(entity_name):
            trace.log(f"Extracted entity for classification: {entity_name}", action="extract")
            return await self._classify(query, context, trace)

        # Check if this looks like a comparison question we failed to parse
        q_lower = query.lower()
        if any(word in q_lower for word in ["similar", "compare", "like", "different", "vs"]):
            return ReasoningResult(
                answer="I detected a comparison question but couldn't identify both entities. "
                "Try: 'Compare X and Y' or 'How is X similar to Y?'",
                confidence=0.3,
                trace_id="",
            )

        # Check if this looks like a question about capability
        if any(word in q_lower for word in ["can", "could", "able"]):
            return ReasoningResult(
                answer="I detected a capability question but couldn't parse it. "
                "Try: 'What is X?' to learn about an entity's properties.",
                confidence=0.3,
                trace_id="",
            )

        return ReasoningResult(
            answer="I can help you:\n"
            "- **Classify**: 'What is a hammer?'\n"
            "- **Compare**: 'Compare cat and dog'\n"
            "- **Infer**: 'What properties does X have?'\n"
            "- **Explore**: 'What's related to X?'\n"
            "\nWhat would you like to know?",
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
            # Return a ClassificationResult from local data (don't fetch from UHT with our UUID)
            return ClassificationResult(
                uuid=local.uuid,
                name=local.name,
                hex_code=local.hex_code,
                binary=local.binary_code,
                traits=[],  # Traits not stored locally
                created_at=local.created_at if isinstance(local.created_at, datetime) else datetime.utcnow(),
            )

        trace.log_entity_lookup(entity_name, False)

        # Validate entity name before classifying to avoid persisting junk
        if not self._is_valid_entity_name(entity_name):
            trace.log(f"Skipping invalid entity name: {entity_name}", action="skip")
            # Return a minimal result without persisting
            return ClassificationResult(
                uuid="invalid",
                name=entity_name,
                hex_code="00000000",
                binary="0" * 32,
                traits=[],
                created_at=datetime.utcnow(),
            )

        # Check Factory corpus before running expensive classification
        trace.log(f"Checking Factory corpus for {entity_name}", action="search")
        try:
            existing = await self._uht.search_entities(query=entity_name, limit=5)
            # Look for exact match (case-insensitive)
            for entity in existing:
                if entity.name.lower() == entity_name.lower():
                    trace.log(f"Found in Factory corpus: {entity.name} ({entity.hex_code})", action="cache_hit")
                    # Cache locally for future lookups
                    await self._graph.upsert_entity(entity, source="uht_factory")
                    return entity
        except Exception as e:
            # If search fails, continue to classification
            trace.log(f"Factory search failed: {e}", action="error")

        # Not found in corpus - run new classification (32 GPT calls)
        trace.log(f"Classifying {entity_name} via UHT (new)", action="api")

        classification = await self._uht.classify(entity_name, context=context.as_string())
        await self._graph.upsert_entity(classification, source="uht_factory")

        trace.log_classification(entity_name, classification.hex_code, classification.uuid)
        return classification

    def _is_fresh(self, entity: StoredEntity) -> bool:
        """Check if a cached entity is still fresh."""
        freshness_hours = self._settings.context_relevance_window_hours
        # Handle Neo4j DateTime type by converting to Python datetime
        updated_at = entity.updated_at
        if hasattr(updated_at, 'to_native'):
            # Neo4j DateTime has to_native() method
            updated_at = updated_at.to_native()
        elif not isinstance(updated_at, datetime):
            # Try to extract from Neo4j DateTime
            try:
                updated_at = datetime(
                    updated_at.year, updated_at.month, updated_at.day,
                    updated_at.hour, updated_at.minute, updated_at.second
                )
            except (AttributeError, TypeError):
                # If conversion fails, assume fresh
                return True
        age = datetime.utcnow() - updated_at.replace(tzinfo=None)
        return age < timedelta(hours=freshness_hours)

    def _extract_entity_name(self, query: str) -> str:
        """Extract entity name from a query."""
        # Try quoted strings first
        quoted = re.findall(r'"([^"]+)"', query)
        if quoted:
            return self._clean_entity_name(quoted[0])

        # Normalize query
        q = query.strip().rstrip("?").lower()

        # Pattern 1: "what is/are [a/an/the] X"
        what_is = re.search(r"what (?:is|are) (?:a |an |the )?(.+)", q)
        if what_is:
            return self._clean_entity_name(what_is.group(1).strip())

        # Pattern 2: "is [a/an/the] X alive/Y" - extract subject before predicate
        is_x = re.search(r"^is (?:a |an |the )?([a-z][a-z\s\-]+?)(?:\s+(?:alive|dead|real|true|valid|a\s|an\s|the\s|like|similar|related)|\s*$)", q)
        if is_x:
            return self._clean_entity_name(is_x.group(1).strip())

        # Pattern 3: "can [a/an/the] X be/do Y" - extract subject
        can_x = re.search(r"^can (?:a |an |the )?([a-z][a-z\s\-]+?)\s+(?:be|do|have|make|get|become)", q)
        if can_x:
            return self._clean_entity_name(can_x.group(1).strip())

        # Pattern 4: "why is X similar/related/like Y" - this is a comparison, return first entity
        why_similar = re.search(r"^why (?:is|are) (?:a |an |the )?([a-z][a-z\s\-]+?)\s+(?:similar|related|like|close)", q)
        if why_similar:
            return self._clean_entity_name(why_similar.group(1).strip())

        # Pattern 5: "about X" or "describe X" or "explain X"
        about = re.search(r"(?:about|describe|explain|tell me about) (?:a |an |the )?(.+)", q)
        if about:
            return self._clean_entity_name(about.group(1).strip())

        # Pattern 6: "classify X" or "what is the classification of X"
        classify = re.search(r"(?:classify|classification of) (?:a |an |the )?(.+)", q)
        if classify:
            return self._clean_entity_name(classify.group(1).strip())

        # Fallback: extract noun phrases using simple heuristic
        # Remove common question words and verbs
        stopwords = {
            "what", "is", "are", "was", "were", "be", "been", "being",
            "the", "a", "an", "this", "that", "these", "those",
            "can", "could", "will", "would", "should", "may", "might",
            "do", "does", "did", "have", "has", "had",
            "how", "why", "when", "where", "who", "which",
            "it", "its", "they", "them", "their",
            "to", "of", "in", "on", "at", "for", "with", "by",
            "and", "or", "but", "if", "then", "so",
            "more", "most", "like", "similar", "related", "alive", "trained",
        }

        words = re.findall(r"[a-z][a-z\-]+", q)
        content_words = [w for w in words if w not in stopwords and len(w) > 1]

        if content_words:
            # Take first 1-3 content words as the entity
            entity = " ".join(content_words[:3])
            return self._clean_entity_name(entity)

        return ""

    def _clean_entity_name(self, name: str) -> str:
        """Clean extracted entity name, removing query artifacts."""
        # Remove leading/trailing punctuation and whitespace
        name = name.strip(" ?.,!;:")

        # Skip obvious query fragments that shouldn't be classified
        query_words = {
            "related to", "connected to", "similar to", "compared to",
            "entities in", "pattern as", "in database", "in the",
            "neighborhood of", "properties of", "classification of",
        }
        name_lower = name.lower()
        for fragment in query_words:
            if name_lower.startswith(fragment):
                # Extract the actual entity after the fragment
                remainder = name[len(fragment):].strip()
                if remainder:
                    return remainder

        return name

    def _is_valid_entity_name(self, name: str) -> bool:
        """Check if a name looks like a valid entity (not query noise)."""
        if not name or len(name) < 2:
            return False

        name_lower = name.lower()

        # Reject names that are clearly query fragments
        invalid_patterns = [
            "entities", "database", "corpus", "factory",
            "similar", "related", "connected", "compared",
            "neighborhood", "properties", "classification",
            "what is", "what are", "how do", "why is",
            "pattern", "hex code", "trait",
            "compare ", "list ", "search ", "find ",
        ]
        for pattern in invalid_patterns:
            if pattern in name_lower:
                return False

        # Reject if it ends with a question mark (query, not entity)
        if name.endswith("?"):
            return False

        # Reject if it's mostly stopwords
        stopwords = {"the", "a", "an", "is", "are", "of", "in", "to", "for", "with", "and", "or"}
        words = name_lower.split()
        if len(words) > 0:
            stopword_ratio = sum(1 for w in words if w in stopwords) / len(words)
            if stopword_ratio > 0.6:
                return False

        return True

    def _extract_entity_pair(self, query: str) -> list[str]:
        """Extract two entity names from a comparison query."""
        q = query.strip().rstrip("?").lower()

        # Pattern 1: "why is X similar/related/like Y" (check FIRST - most specific)
        why_similar = re.search(
            r"why (?:is|are) (?:a |an |the )?([a-z][a-z\s\-]+?)\s+(?:similar|related|like|close|comparable)\s+to\s+(?:a |an |the )?([a-z][a-z\s\-]+?)$",
            q,
        )
        if why_similar:
            return [
                self._clean_entity_name(why_similar.group(1).strip()),
                self._clean_entity_name(why_similar.group(2).strip()),
            ]

        # Pattern 2: "how is X different from Y" or "how does X compare to Y"
        how_diff = re.search(
            r"how (?:is|are|does|do) (?:a |an |the )?([a-z][a-z\s\-]+?)\s+(?:different from|compare to|relate to|similar to)\s+(?:a |an |the )?([a-z][a-z\s\-]+?)$",
            q,
        )
        if how_diff:
            return [
                self._clean_entity_name(how_diff.group(1).strip()),
                self._clean_entity_name(how_diff.group(2).strip()),
            ]

        # Pattern 3: "compare X and/vs/with Y" or "X vs Y" (explicit comparison)
        clean_query = re.sub(r"^compare\s+", "", q, flags=re.IGNORECASE)
        vs_match = re.search(
            r"(?:a |an |the )?([a-z][a-z\s\-]+?)\s+(?:vs\.?|versus|and|compared to|with)\s+(?:a |an |the )?([a-z][a-z\s\-]+?)$",
            clean_query,
        )
        if vs_match:
            return [
                self._clean_entity_name(vs_match.group(1).strip()),
                self._clean_entity_name(vs_match.group(2).strip()),
            ]

        # Pattern 4: "between X and Y"
        between = re.search(
            r"between\s+(?:a |an |the )?([a-z][a-z\s\-]+?)\s+and\s+(?:a |an |the )?([a-z][a-z\s\-]+?)$",
            q,
        )
        if between:
            return [
                self._clean_entity_name(between.group(1).strip()),
                self._clean_entity_name(between.group(2).strip()),
            ]

        # Pattern 5: "is X more like Y or Z" - take first two
        more_like = re.search(
            r"is (?:a |an |the )?([a-z][a-z\s\-]+?)\s+more (?:like|similar to)\s+(?:a |an |the )?([a-z][a-z\s\-]+?)\s+or",
            q,
        )
        if more_like:
            return [
                self._clean_entity_name(more_like.group(1).strip()),
                self._clean_entity_name(more_like.group(2).strip()),
            ]

        # Pattern 6: Simple "X to Y" at end (e.g., "compare democracy to religion")
        simple_to = re.search(
            r"(?:a |an |the )?([a-z][a-z\s\-]+?)\s+to\s+(?:a |an |the )?([a-z][a-z\s\-]+?)$",
            clean_query,
        )
        if simple_to:
            return [
                self._clean_entity_name(simple_to.group(1).strip()),
                self._clean_entity_name(simple_to.group(2).strip()),
            ]

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
            f"**Similarity Metrics:**",
            f"- Hamming distance: {analysis.hamming_distance} bits differ",
            f"- Jaccard similarity: {analysis.jaccard_similarity:.0%} (shared traits / all traits)",
            f"- Simple similarity: {analysis.similarity_score:.0%} ((32 - hamming) / 32)",
            "",
            f"**Trait Breakdown:**",
            f"- Shared traits: {len(analysis.shared_traits)}",
            f"- {name_a} only: {len(analysis.traits_a_only)}",
            f"- {name_b} only: {len(analysis.traits_b_only)}",
            "",
        ]

        if inheritance.is_valid:
            lines.append(f"{name_a} could be considered a type of {name_b} (inheritance score: {inheritance.inheritance_score:.0%})")
        elif analysis.jaccard_similarity >= 0.7:
            lines.append(f"These entities share most of their traits and likely have similar properties.")
        elif analysis.jaccard_similarity >= 0.5:
            lines.append(f"These entities have moderate trait overlap — some property transfer is reasonable.")
        elif analysis.jaccard_similarity >= 0.3:
            lines.append(f"These entities have limited trait overlap — they share some properties but differ significantly.")
        else:
            lines.append(f"These entities have very low trait overlap ({analysis.jaccard_similarity:.0%}) — they are fundamentally different.")

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
