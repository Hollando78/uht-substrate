"""MCP server entry point for UHT Substrate Agent."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import FastMCP
from pydantic import BaseModel

from .config.logging import configure_logging, get_logger
from .config.settings import get_settings
from .graph.connection import Neo4jConnection
from .graph.repository import GraphRepository
from .priors.inference import TRAIT_NAMES, PriorInferenceEngine
from .reasoning.engine import ReasoningEngine
from .uht_client.client import UHTClient

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Get settings
settings = get_settings()


# Application context for dependency injection
class AppContext:
    """Application context holding shared resources."""

    neo4j: Optional[Neo4jConnection] = None
    graph: Optional[GraphRepository] = None
    uht: Optional[UHTClient] = None
    inference: Optional[PriorInferenceEngine] = None
    engine: Optional[ReasoningEngine] = None


ctx = AppContext()


async def _startup():
    """Initialize application resources."""
    logger.info("Starting UHT Substrate Agent")
    ctx.neo4j = Neo4jConnection(settings)
    await ctx.neo4j.connect()
    await ctx.neo4j.initialize_schema()
    ctx.graph = GraphRepository(ctx.neo4j)
    ctx.uht = UHTClient(settings)
    ctx.inference = PriorInferenceEngine()
    ctx.engine = ReasoningEngine(ctx.graph, ctx.uht, ctx.inference)
    logger.info("UHT Substrate Agent ready")


async def _shutdown():
    """Cleanup application resources."""
    try:
        logger.info("Shutting down UHT Substrate Agent")
    except ValueError:
        pass
    if ctx.uht:
        await ctx.uht.close()
    if ctx.neo4j:
        await ctx.neo4j.close()


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Manage application lifecycle for FastMCP."""
    await _startup()
    yield
    await _shutdown()


@asynccontextmanager
async def combined_lifespan(app) -> AsyncIterator[None]:
    """Manage application lifecycle for combined Starlette app."""
    await _startup()
    yield
    await _shutdown()


# =============================================================================
# Helper: Store Factory entities in local graph
# =============================================================================


async def _store_factory_entities(
    entities: list[Any],
    source: str = "uht_factory",
) -> int:
    """
    Store entities returned from Factory API in the local Neo4j graph.

    This grows the local knowledge graph as the LLM explores concepts,
    enabling faster lookups and relationship discovery.

    Args:
        entities: List of Entity, ClassificationResult, SemanticSearchResult,
                  or SimilarityResult objects
        source: Source tag for provenance

    Returns:
        Number of entities stored
    """
    if not ctx.graph:
        return 0

    from .uht_client.models import (
        ClassificationResult,
        Entity,
        SemanticSearchResult,
        SimilarityResult,
    )

    stored = 0
    stored_names: list[str] = []
    for e in entities:
        try:
            name: str | None = None
            # Convert various result types to Entity-like for storage
            if isinstance(e, (ClassificationResult, Entity)):
                name = e.entity if isinstance(e, ClassificationResult) else e.name
                await ctx.graph.upsert_entity(e, source=source)
                stored += 1
            elif isinstance(e, SemanticSearchResult):
                name = e.name
                # Create minimal Entity from search result
                entity = Entity(
                    uuid=e.uuid,
                    name=e.name,
                    hex_code=e.hex_code,
                    description=e.description,
                    created_at=__import__("datetime").datetime.utcnow(),
                )
                await ctx.graph.upsert_entity(entity, source=source)
                stored += 1
            elif isinstance(e, SimilarityResult):
                name = e.name
                # SimilarityResult has entity fields directly
                entity = Entity(
                    uuid=e.uuid,
                    name=e.name,
                    hex_code=e.hex_code,
                    description=e.description,
                    binary=e.binary,
                    created_at=__import__("datetime").datetime.utcnow(),
                )
                await ctx.graph.upsert_entity(entity, source=source)
                stored += 1
            elif isinstance(e, dict) and "uuid" in e and "hex_code" in e:
                name = e.get("name", "Unknown")
                # Raw dict with entity data
                entity = Entity(
                    uuid=e["uuid"],
                    name=name,
                    hex_code=e.get("hex_code") or e.get("uht_code", "00000000"),
                    description=e.get("description"),
                    created_at=__import__("datetime").datetime.utcnow(),
                )
                await ctx.graph.upsert_entity(entity, source=source)
                stored += 1
            if name:
                stored_names.append(name)
        except Exception as ex:
            logger.debug("Failed to store entity", error=str(ex))
            continue

    # Retroactively bind unbound facts referencing any newly stored entities
    for name in stored_names:
        try:
            await ctx.graph.bind_pending_facts_for_entity(name)
        except Exception:
            pass  # Best-effort binding

    if stored > 0:
        logger.debug("Stored Factory entities in local graph", count=stored)

    return stored


# Initialize MCP server
mcp = FastMCP(
    name="UHT Substrate Agent",
    instructions="""
# UHT Substrate Agent

A reasoning substrate grounded in the Universal Hex Taxonomy (UHT). Every entity gets a 32-bit classification (8-char hex code) across four layers:
- **Physical** (bits 1-8): Material properties, boundaries, energy
- **Functional** (bits 9-16): Capabilities, interfaces, state
- **Abstract** (bits 17-24): Symbolic, temporal, rule-governed
- **Social** (bits 25-32): Cultural, economic, institutional

## Core Tools
- `classify_entity(entity)` → hex code + inferred properties
- `compare_entities(entity_a, entity_b)` → Jaccard similarity, shared/unique traits
- `infer_properties(entity)` → axiom-derived properties grouped by confidence
- `semantic_search(query)` → embedding-based conceptual similarity
- `list_entities()` → browse local knowledge graph
- `disambiguate_term(term)` → word senses with UHT classifications

## Reasoning Patterns

**Pattern 1: Analogical Transfer** — "What is X most like?"
```
1. semantic_search(X) or list_entities() to get candidates
2. compare_entities(X, candidate) for each
3. Rank by Jaccard similarity; highest = best structural analogy
```

**Pattern 2: Category Membership** — "Is X a Y?" (e.g., "Is a virus alive?")
```
1. classify_entity(X)
2. classify_entity(exemplar_of_Y) for 2-3 exemplars (e.g., "bacterium", "cat")
3. compare_entities(X, exemplar) for each
4. If X clusters with exemplars (Jaccard > 50%), X is likely a Y
```

**Pattern 3: Property Inheritance** — "Can X do what Y does?"
```
1. compare_entities(X, Y)
2. Check if the relevant trait is in shared_traits or traits_Y_only
3. If shared → property transfers; if Y-only → property doesn't transfer
```

**Pattern 4: Disambiguation** — "Which sense of word W fits context C?"
```
1. disambiguate_term(W) to get senses
2. classify_entity(context_word) for key context words
3. compare_entities(sense, context_word) for each sense
4. Highest cumulative Jaccard = best-fitting sense
```

**Pattern 5: Ontological Surprise** — "What's unexpectedly similar?"
```
1. list_entities() to get corpus
2. compare_entities(X, Y) across domain boundaries
3. High Jaccard (>50%) between different domains = structural insight
```

## Key Metrics
- **Jaccard similarity**: shared_traits / (shared + unique_a + unique_b) — meaningful semantic overlap
- **Hamming distance**: bit differences — raw divergence
- **Inheritance score**: child_has_parent_traits / parent_traits — IS_A relationship strength

## Tips
- Jaccard > 70%: very similar, property transfer safe
- Jaccard 30-70%: partial overlap, check specific traits
- Jaccard < 30%: fundamentally different
- 0% Jaccard between two entities = strong exclusion signal (useful for disambiguation)
""",
    lifespan=app_lifespan,
)


# =============================================================================
# Classification Tools
# =============================================================================

# Keyword → trait bit mapping for property-to-trait inference.
# Each keyword maps to one or more trait bit positions (1-32).
KEYWORD_TRAIT_MAP: dict[str, list[int]] = {
    # Physical layer (1-8)
    "physical": [1], "tangible": [1], "material": [1, 7], "object": [1],
    "manufactured": [2], "synthetic": [2], "human-made": [2], "artificial": [2],
    "biological": [3], "living": [3], "organic": [3], "biomimetic": [3],
    "powered": [4], "electric": [4], "energy": [4], "battery": [4],
    "structural": [5], "load-bearing": [5], "support": [5],
    "observable": [6], "visible": [6], "measurable": [6], "detectable": [6],
    "medium": [7], "substance": [7], "fluid": [7],
    "active": [8], "motion": [8], "dynamic": [8], "moving": [8],
    # Functional layer (9-16)
    "designed": [9], "intentional": [9], "purposeful": [9], "engineered": [9],
    "output": [10], "effect": [10], "produces": [10], "emits": [10],
    "signal": [11, 18], "process": [11], "logic": [11], "compute": [11], "information": [11, 18],
    "transform": [12], "change": [12], "alter": [12], "modify": [12], "convert": [12],
    "interactive": [13], "user": [13], "human-operated": [13], "interface": [13],
    "integrated": [14], "system": [14], "component": [14], "subsystem": [14],
    "autonomous": [15], "independent": [15], "self-governing": [15],
    "essential": [16], "critical": [16], "vital": [16],
    # Abstract layer (17-24)
    "symbolic": [17], "represent": [17], "signify": [17], "denote": [17],
    "meaning": [17], "belief": [17], "concept": [17], "idea": [17],
    "communicate": [18], "inform": [18], "convey": [18], "express": [18],
    "rule": [19], "governed": [19], "formal": [19], "protocol": [19],
    "compositional": [20], "combine": [20], "modular": [20], "discrete": [20],
    "conflicting": [20], "tension": [20], "contradiction": [20],
    "normative": [21], "prescriptive": [21], "ought": [21], "should": [21],
    "meta": [22], "self-referential": [22], "reflexive": [22],
    "cognitive": [22, 17], "psychological": [22, 17], "mental": [22, 17],
    "temporal": [23], "time": [23], "sequence": [23], "duration": [23],
    "motivates": [8, 12], "drives": [8, 12], "causes": [10, 12],
    "behavior": [8, 13], "behavioural": [8, 13], "behavioral": [8, 13],
    "discomfort": [10, 12], "stress": [10, 12],
    "digital": [24], "virtual": [24], "software": [24], "cyber": [24],
    # Social layer (25-32)
    "social": [25], "collective": [25], "communal": [25],
    "cultural": [25, 31], "tradition": [31], "ceremonial": [31],
    "psychology": [25, 17], "sociological": [25], "anthropological": [25],
    "institutional": [26], "organizational": [26], "bureaucratic": [26],
    "identity": [27], "personal": [27], "belonging": [27],
    "regulated": [28], "legal": [28], "compliance": [28], "law": [28],
    "economic": [29], "market": [29], "monetary": [29], "financial": [29],
    "political": [30], "governance": [30], "power": [30], "policy": [30],
    "ritual": [31], "ceremony": [31],
    "ethical": [32], "moral": [32], "rights": [32], "justice": [32],
}

# Stop words to exclude from token matching
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "and", "but", "or", "nor", "not", "no",
    "so", "yet", "both", "either", "neither", "each", "every", "all",
    "any", "few", "more", "most", "other", "some", "such", "than",
    "too", "very", "just", "that", "this", "these", "those", "it", "its",
})


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase word set, excluding stop words."""
    import re
    words = set(re.findall(r"[a-z]+", text.lower()))
    return words - _STOP_WORDS


async def _map_properties_to_traits(
    properties: list[str],
) -> list[dict[str, Any]]:
    """Map natural language properties to candidate UHT trait bits.

    Uses a two-pass strategy:
    1. Direct keyword lookup from KEYWORD_TRAIT_MAP (high confidence)
    2. Token overlap between property text and trait definitions (lower confidence)

    Args:
        properties: List of natural language property strings

    Returns:
        List of mappings, one per property, each with candidate traits
    """
    # Fetch trait definitions for token overlap matching
    trait_defs: list[Any] = []
    if ctx.uht:
        try:
            trait_defs = await ctx.uht.get_traits()
        except Exception:
            pass

    # Build trait text corpus for token overlap
    trait_texts: dict[int, set[str]] = {}
    for td in trait_defs:
        text = f"{td.name} {td.short_description} {td.expanded_definition}"
        trait_texts[td.bit_position] = _tokenize(text)

    results = []
    for prop in properties:
        prop_tokens = _tokenize(prop)
        candidates: dict[int, dict[str, Any]] = {}

        # Pass 1: Direct keyword matches (high confidence)
        for token in prop_tokens:
            if token in KEYWORD_TRAIT_MAP:
                for bit in KEYWORD_TRAIT_MAP[token]:
                    if bit not in candidates:
                        candidates[bit] = {
                            "bit": bit,
                            "name": TRAIT_NAMES.get(bit, f"Bit {bit}"),
                            "rationale": f"keyword match: '{token}'",
                            "mapping_confidence": 0.85,
                        }
                    else:
                        # Multiple keyword hits increase confidence
                        c = candidates[bit]
                        c["mapping_confidence"] = min(0.95, c["mapping_confidence"] + 0.1)
                        c["rationale"] += f", '{token}'"

        # Pass 2: Token overlap with trait definitions (lower confidence)
        if prop_tokens and trait_texts:
            for bit, trait_tokens in trait_texts.items():
                if bit in candidates:
                    continue  # Already matched via keyword
                overlap = prop_tokens & trait_tokens
                if overlap:
                    jaccard = len(overlap) / len(prop_tokens | trait_tokens)
                    if jaccard > 0.03:
                        candidates[bit] = {
                            "bit": bit,
                            "name": TRAIT_NAMES.get(bit, f"Bit {bit}"),
                            "rationale": f"definition overlap: {', '.join(sorted(overlap))}",
                            "mapping_confidence": round(min(0.7, jaccard * 5), 2),
                        }

        # Sort by confidence descending
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda c: c["mapping_confidence"],
            reverse=True,
        )

        results.append({
            "property": prop,
            "candidate_traits": sorted_candidates,
        })

    return results


async def _resolve_classification(
    entity_name: str,
    context: str | None = None,
    force_refresh: bool = False,
    namespace: str | None = None,
) -> "ClassificationResult":
    """Resolve an entity classification using local-first strategy.

    Lookup order:
    1. Local Neo4j graph (free, instant)
    2. UHT Factory corpus search (cheap GET, 16k+ entities)
    3. UHT Factory API classify (expensive POST, 32 parallel evaluators)

    force_refresh=True skips steps 1-2 and always calls classify.
    context/namespace are only used when a fresh classify is needed.

    Returns a ClassificationResult (from local, Factory corpus, or fresh).
    Stores the result locally if it came from an API call.
    """
    from uht_substrate.uht_client.models import ClassificationResult

    # Step 1: Check local graph (unless force_refresh)
    if not force_refresh and ctx.graph:
        local = await ctx.graph.get_classification_by_name(entity_name)
        if local:
            logger.debug("Classification resolved from local graph", entity=entity_name)
            return local

    if not ctx.uht:
        raise RuntimeError("UHT client not initialized")

    # Step 2: Search Factory corpus (unless force_refresh)
    if not force_refresh:
        try:
            existing = await ctx.uht.search_entities(query=entity_name, limit=5)
            entity_lower = entity_name.lower()
            for e in existing:
                if e.name.lower() == entity_lower:
                    logger.debug("Classification resolved from Factory corpus", entity=entity_name)
                    binary = e.binary or _hex_to_binary(e.hex_code)
                    # Use traits from search if available, else derive from hex
                    traits = e.traits
                    if not traits:
                        from uht_substrate.uht_client.models import TraitValue
                        traits = [
                            TraitValue(
                                bit_position=bit,
                                name=TRAIT_NAMES.get(bit, f"Bit {bit}"),
                                present=binary[bit - 1] == "1",
                                confidence=1.0 if binary[bit - 1] == "1" else 0.0,
                                justification=None,
                            )
                            for bit in range(1, 33)
                        ]
                    result = ClassificationResult(
                        uuid=e.uuid,
                        name=e.name,
                        hex_code=e.hex_code,
                        binary=binary,
                        traits=traits,
                        created_at=e.created_at,
                    )
                    # Cache locally
                    if ctx.graph:
                        await ctx.graph.upsert_entity(
                            result,
                            source="uht_factory",
                            namespace=namespace,
                        )
                    return result
        except Exception as e:
            logger.warning("Factory corpus lookup failed, falling through to classify", error=str(e), entity=entity_name)

    # Step 3: Fresh classification via API
    result = await ctx.uht.classify(
        entity=entity_name,
        context=context,
        force_refresh=force_refresh,
        namespace=namespace,
    )

    # Store in local graph for future lookups
    if ctx.graph:
        await ctx.graph.upsert_entity(
            result,
            source="uht_factory",
            description=context,
            namespace=namespace,
        )

    return result


def _hex_to_binary(hex_code: str) -> str:
    """Convert 8-char hex code to 32-bit binary string."""
    return bin(int(hex_code, 16))[2:].zfill(32)


@mcp.tool()
async def classify_entity(
    entity: str,
    context: str = "",
    namespace: str = "",
    force_refresh: bool = False,
    use_semantic_priors: bool = False,
) -> dict[str, Any]:
    """
    Classify an entity using the Universal Hex Taxonomy (UHT).

    Returns an 8-character hex code encoding 32 binary traits across four layers:
    - Physical (bits 1-8): extent, location, mass, boundaries, energy, structure
    - Functional (bits 9-16): detection, state, signals, interfaces, autonomy
    - Abstract (bits 17-24): symbolic, conventional, temporal, rule-governed
    - Social (bits 25-32): cultural, economic, institutional, political

    The hex code is the foundation for all UHT reasoning. Use this to:
    - Ground an entity before comparing it to others
    - Get inferred properties from trait axioms
    - Build up a local knowledge graph for fast lookups

    Args:
        entity: The entity to classify (e.g., "hammer", "democracy", "virus")
        context: Optional context/description to guide classification
                 (e.g., "a programming language" or "the boundary between two systems")
        namespace: Optional namespace code (e.g., "SE", "SE:aerospace").
                   If provided, entity is assigned to this namespace.
                   If empty, assigned to "global" namespace.
        force_refresh: Skip local cache and request fresh classification.
                       Note: Factory corpus entities may still return cached results
                       from the Factory API. For truly fresh classification of
                       corpus entities, include distinctive context.
        use_semantic_priors: Run semantic triangle decomposition first and use
                            essential properties as classification priors. This
                            enriches the context with ontological analysis before
                            trait evaluation.

    Returns:
        hex_code: 8-char hex (e.g., "C6880008")
        properties: Inferred properties from trait axioms
        traits: Individual trait evaluations
    """
    if not ctx.uht:
        return {"error": "UHT client not initialized"}

    try:
        # Semantic priors: run triangle decomposition and map to traits
        semantic_priors: list[dict[str, Any]] | None = None
        if use_semantic_priors:
            try:
                triangle = await ctx.uht.get_semantic_triangle(entity)
                essential_props = triangle.thought.essential_properties
                category = triangle.thought.category

                if essential_props:
                    mappings = await _map_properties_to_traits(essential_props)
                    semantic_priors = []
                    suggested_traits: list[str] = []

                    for m in mappings:
                        candidates = m["candidate_traits"]
                        semantic_priors.append({
                            "essential_property": m["property"],
                            "candidate_traits": [c["bit"] for c in candidates],
                            "trait_names": [c["name"] for c in candidates],
                            "mapping_confidence": round(
                                max((c["mapping_confidence"] for c in candidates), default=0.0),
                                2,
                            ),
                        })
                        suggested_traits.extend(c["name"] for c in candidates)

                    # Build enhanced context
                    props_str = "; ".join(essential_props)
                    traits_str = ", ".join(sorted(set(suggested_traits)))
                    prior_context = (
                        f"Semantic analysis identifies these essential properties: {props_str}. "
                        f"Category: {category}. "
                        f"These suggest traits: {traits_str}."
                    )
                    context = f"{prior_context} {context}".strip() if context else prior_context
            except Exception as e:
                log.warning("Semantic priors failed, continuing without", error=str(e))
        # Determine the entity name to send to Factory API
        # When force_refresh + namespace, use "entity@namespace" format to bypass
        # Factory's corpus cache for polysemous terms
        classify_name = entity
        if force_refresh and namespace:
            classify_name = f"{entity}@{namespace}"
            log.info(
                "Using namespace-qualified name for fresh classification",
                original=entity,
                qualified=classify_name,
            )

        # Resolve classification: local graph first, then API
        result = await _resolve_classification(
            entity_name=classify_name,
            context=context if context else None,
            force_refresh=force_refresh,
            namespace=namespace if namespace else None,
        )

        # If we used a namespace-qualified name, re-store under original name
        if classify_name != entity and ctx.graph:
            from uht_substrate.uht_client.models import ClassificationResult
            store_result = ClassificationResult(
                uuid=result.uuid,
                name=entity,
                hex_code=result.hex_code,
                binary=result.binary,
                traits=result.traits,
                created_at=result.created_at,
            )
            await ctx.graph.upsert_entity(
                store_result,
                source="uht_factory",
                description=context or None,
                namespace=namespace or None,
            )

            # Retroactively bind any unbound facts referencing this entity
            await ctx.graph.bind_pending_facts_for_entity(entity)

        # Get inferred properties from trait axioms
        inferred_properties = []
        if ctx.inference:
            props = ctx.inference.infer_properties(result.hex_code, entity)
            inferred_properties = [
                {
                    "property": p.property_name,
                    "confidence": p.confidence,
                    "source": p.source_axiom_name,
                    "reasoning": p.reasoning_trace,
                }
                for p in props[:15]
            ]

        response = {
            "entity": entity,
            "context_used": context if context else None,
            "namespace": namespace if namespace else "global",
            "force_refresh": force_refresh,
            "hex_code": result.hex_code,
            "binary": result.binary,
            "traits": [
                {
                    "bit": t.bit_position,
                    "name": t.name,
                    "present": t.present,
                    "confidence": t.confidence,
                    "justification": t.justification,
                }
                for t in result.traits
            ],
            "inferred_properties": inferred_properties,
        }

        if semantic_priors is not None:
            response["semantic_priors"] = semantic_priors

        return response
    except Exception as e:
        log.error("classify_entity failed", entity=entity, error=str(e))
        return {"error": f"Classification failed: {str(e)}"}


@mcp.tool()
async def find_similar_entities(
    entity: str,
    limit: int = 5,
    min_shared_traits: int = 20,
) -> dict[str, Any]:
    """
    [EXPERIMENTAL] Find entities similar to the given entity.

    NOTE: This tool often returns empty results. For finding similar entities,
    use semantic_search() instead — it searches the full 16k+ Factory corpus
    and reliably returns results.

    Args:
        entity: Entity to find similar items for
        limit: Maximum number of results (1-20)
        min_shared_traits: Minimum shared traits required (20-32, default 20)

    Returns:
        List of similar entities (may be empty due to threshold/corpus issues)
    """
    if not ctx.uht or not ctx.graph:
        return {"error": "Engine not initialized"}

    limit = min(max(limit, 1), 20)
    min_shared_traits = min(max(min_shared_traits, 20), 32)

    try:
        entity_data = await _resolve_classification(entity)

        # Find similar via UHT API (requires Factory UUID)
        similar_results = await ctx.uht.find_similar(
            uuid=entity_data.uuid,
            threshold=min_shared_traits,
            limit=limit,
        )

        # Store returned entities in local graph for future lookups
        await _store_factory_entities(similar_results, source="uht_factory")

        return {
            "entity": entity,
            "hex_code": entity_data.hex_code,
            "similar_entities": [
                {
                    "name": r.name,
                    "hex_code": r.hex_code,
                    "similarity_score": r.similarity_score,
                    "shared_traits": r.shared_trait_count,
                }
                for r in similar_results
            ],
            "count": len(similar_results),
        }
    except Exception as e:
        return {"error": f"Failed to find similar entities: {e}"}


@mcp.tool()
async def list_entities(
    hex_pattern: str = "",
    name_contains: str = "",
    namespace: str = "",
    limit: int = 50,
    offset: int = 0,
    source: str = "both",
) -> dict[str, Any]:
    """
    List classified entities from the local graph and/or UHT Factory corpus.

    Supports pagination via limit/offset. Response includes `total` count
    and `has_more` flag so you can page through large result sets.

    Args:
        hex_pattern: Filter by hex code pattern (e.g., "C688" for tools)
        name_contains: Filter by name substring (case-insensitive)
        namespace: Filter by namespace (e.g., "SE", "SE:aerospace"). Includes descendants.
        limit: Maximum number of results (1-100)
        offset: Skip first N results for pagination (default 0)
        source: "local" (Neo4j only), "factory" (UHT API only), or "both"

    Returns:
        List of entities with names, hex codes, and source
    """
    limit = min(max(limit, 1), 100)
    offset = max(offset, 0)
    results = []
    total = 0

    # Query local Neo4j graph
    if source in ("local", "both") and ctx.graph:
        local_entities, local_total = await ctx.graph.list_entities(
            name_contains=name_contains if name_contains else None,
            hex_pattern=hex_pattern if hex_pattern else None,
            namespace=namespace if namespace else None,
            limit=limit,
            offset=offset,
        )
        total = local_total
        for e in local_entities:
            results.append({
                "name": e.name,
                "hex_code": e.hex_code,
                "source": "local",
                "created_at": str(e.created_at) if e.created_at else None,
            })

    # Query UHT Factory API
    if source in ("factory", "both") and ctx.uht:
        try:
            factory_entities = await ctx.uht.search_entities(
                query=name_contains if name_contains else None,
                uht_pattern=hex_pattern if hex_pattern else None,
                limit=limit,
                offset=offset,
            )

            # Store Factory entities in local graph for future lookups
            await _store_factory_entities(factory_entities, source="uht_factory")

            for e in factory_entities:
                # Avoid duplicates from local
                if not any(r["name"] == e.name for r in results):
                    results.append({
                        "name": e.name,
                        "hex_code": e.hex_code,
                        "source": "factory",
                        "created_at": str(e.created_at) if e.created_at else None,
                    })
        except Exception:
            # Factory query failed, continue with local results
            pass

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "count": len(results),
        "has_more": offset + limit < total,
        "entities": results[:limit],
        "filters": {
            "hex_pattern": hex_pattern or None,
            "name_contains": name_contains or None,
            "namespace": namespace or None,
            "source": source,
        },
    }


@mcp.tool()
async def search_by_traits(
    # Physical layer (bits 1-8)
    physical_object: bool | None = None,
    synthetic: bool | None = None,
    biological: bool | None = None,
    powered: bool | None = None,
    structural: bool | None = None,
    observable: bool | None = None,
    physical_medium: bool | None = None,
    active: bool | None = None,
    # Functional layer (bits 9-16)
    intentionally_designed: bool | None = None,
    outputs_effect: bool | None = None,
    processes_signals: bool | None = None,
    state_transforming: bool | None = None,
    human_interactive: bool | None = None,
    system_integrated: bool | None = None,
    functionally_autonomous: bool | None = None,
    system_essential: bool | None = None,
    # Abstract layer (bits 17-24)
    symbolic: bool | None = None,
    signalling: bool | None = None,
    rule_governed: bool | None = None,
    compositional: bool | None = None,
    normative: bool | None = None,
    meta: bool | None = None,
    temporal: bool | None = None,
    digital_virtual: bool | None = None,
    # Social layer (bits 25-32)
    social_construct: bool | None = None,
    institutionally_defined: bool | None = None,
    identity_linked: bool | None = None,
    regulated: bool | None = None,
    economically_significant: bool | None = None,
    politicised: bool | None = None,
    ritualised: bool | None = None,
    ethically_significant: bool | None = None,
    # Limit
    limit: int = 20,
) -> dict[str, Any]:
    """
    Search for entities matching specific trait constraints.

    Use this to find entities with particular combinations of traits.
    Pass True to require a trait, False to exclude it, or omit to ignore.

    Example: Find biological, non-synthetic entities:
      biological=True, synthetic=False

    Example: Find symbolic, rule-governed abstractions:
      symbolic=True, rule_governed=True, physical_object=False

    Example: Find powered, human-interactive devices:
      powered=True, human_interactive=True

    Args:
        physical_object: Bit 1 - Has physical form/mass
        synthetic: Bit 2 - Human-made/manufactured
        biological: Bit 3 - Has biological origin/structure
        powered: Bit 4 - Requires energy input
        structural: Bit 5 - Load-bearing/structural
        observable: Bit 6 - Can be directly observed
        physical_medium: Bit 7 - Transmits/contains physical substance
        active: Bit 8 - Exhibits autonomous motion/change
        intentionally_designed: Bit 9 - Created with purpose
        outputs_effect: Bit 10 - Produces observable effects
        processes_signals: Bit 11 - Processes information/signals
        state_transforming: Bit 12 - Changes state of other entities
        human_interactive: Bit 13 - Designed for human use
        system_integrated: Bit 14 - Part of larger system
        functionally_autonomous: Bit 15 - Operates independently
        system_essential: Bit 16 - Critical to system function
        symbolic: Bit 17 - Represents or signifies something
        signalling: Bit 18 - Conveys information
        rule_governed: Bit 19 - Follows explicit rules
        compositional: Bit 20 - Made of discrete combinable parts
        normative: Bit 21 - Prescribes how things should be
        meta: Bit 22 - About other concepts/systems
        temporal: Bit 23 - Inherently time-related
        digital_virtual: Bit 24 - Exists in digital/virtual form
        social_construct: Bit 25 - Exists through collective agreement
        institutionally_defined: Bit 26 - Defined by institutions
        identity_linked: Bit 27 - Tied to personal/group identity
        regulated: Bit 28 - Subject to formal regulation
        economically_significant: Bit 29 - Has economic value/role
        politicised: Bit 30 - Subject to political contestation
        ritualised: Bit 31 - Involves ritualistic practices
        ethically_significant: Bit 32 - Raises ethical considerations
        limit: Maximum results (1-100)

    Returns:
        List of matching entities with hex codes
    """
    if not ctx.uht:
        return {"error": "UHT client not initialized"}

    # Build 32-char pattern with X for wildcards
    pattern = ["X"] * 32

    trait_map = {
        1: physical_object,
        2: synthetic,
        3: biological,
        4: powered,
        5: structural,
        6: observable,
        7: physical_medium,
        8: active,
        9: intentionally_designed,
        10: outputs_effect,
        11: processes_signals,
        12: state_transforming,
        13: human_interactive,
        14: system_integrated,
        15: functionally_autonomous,
        16: system_essential,
        17: symbolic,
        18: signalling,
        19: rule_governed,
        20: compositional,
        21: normative,
        22: meta,
        23: temporal,
        24: digital_virtual,
        25: social_construct,
        26: institutionally_defined,
        27: identity_linked,
        28: regulated,
        29: economically_significant,
        30: politicised,
        31: ritualised,
        32: ethically_significant,
    }

    for bit_pos, value in trait_map.items():
        if value is not None:
            pattern[bit_pos - 1] = "1" if value else "0"

    pattern_str = "".join(pattern)

    # Only search if at least one constraint specified
    if pattern_str == "X" * 32:
        return {"error": "Specify at least one trait constraint"}

    try:
        entities = await ctx.uht.search_by_pattern(pattern_str, limit=min(limit, 100))

        # Store returned entities in local graph for future lookups
        await _store_factory_entities(entities, source="uht_factory")

        return {
            "pattern": pattern_str,
            "count": len(entities),
            "entities": [
                {"name": e.name, "hex_code": e.hex_code}
                for e in entities
            ],
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def delete_entity(
    name: str,
    source: str = "local",
) -> dict[str, Any]:
    """
    Delete an entity from the local knowledge graph.

    Use this to clean up junk entities (accidental classifications of query text).
    Note: This only deletes from the local Neo4j cache, not from the UHT Factory corpus.

    Args:
        name: Exact name of the entity to delete
        source: Currently only "local" is supported

    Returns:
        Confirmation of deletion
    """
    if source != "local":
        return {"error": "Only local deletion is currently supported"}

    if not ctx.graph:
        return {"error": "Graph not initialized"}

    # Find the entity first
    entity = await ctx.graph.find_entity_by_name(name)
    if not entity:
        return {"error": f"Entity '{name}' not found in local graph"}

    # Delete it
    deleted = await ctx.graph.delete_entity(entity.uuid)

    return {
        "deleted": deleted,
        "name": name,
        "uuid": entity.uuid,
        "hex_code": entity.hex_code,
    }


# =============================================================================
# Namespace Tools
# =============================================================================


@mcp.tool()
async def create_namespace(
    code: str,
    name: str,
    description: str = "",
) -> dict[str, Any]:
    """
    Create a new namespace for organizing entities.

    Hierarchical namespaces use colon separators (e.g., "SE:aerospace:propulsion").
    Parent namespaces are automatically created if they don't exist.

    Args:
        code: Unique namespace code (e.g., "SE", "SE:aerospace", "BIO:genomics")
        name: Human-readable name
        description: Optional description

    Returns:
        Created namespace details
    """
    if not ctx.graph:
        return {"error": "Graph not initialized"}

    try:
        ns = await ctx.graph.create_namespace(
            code=code,
            name=name,
            description=description if description else None,
        )

        return {
            "code": ns.code,
            "name": ns.name,
            "description": ns.description,
            "is_root": ns.is_root,
            "created_at": str(ns.created_at),
        }
    except Exception as e:
        log.error("create_namespace failed", code=code, error=str(e))
        return {"error": f"Failed to create namespace: {str(e)}"}


@mcp.tool()
async def list_namespaces(
    parent: str = "",
    include_descendants: bool = False,
) -> dict[str, Any]:
    """
    List namespaces in the knowledge graph.

    Args:
        parent: List children of this namespace (empty for root namespaces)
        include_descendants: Include entire subtree under parent

    Returns:
        List of namespaces with their hierarchy information
    """
    if not ctx.graph:
        return {"error": "Graph not initialized"}

    try:
        namespaces = await ctx.graph.list_namespaces(
            parent_code=parent if parent else None,
            include_descendants=include_descendants,
        )

        results = []
        for ns in namespaces:
            entity_count = await ctx.graph.count_entities_in_namespace(ns.code)
            results.append({
                "code": ns.code,
                "name": ns.name,
                "description": ns.description,
                "is_root": ns.is_root,
                "entity_count": entity_count,
            })

        return {
            "count": len(results),
            "parent": parent if parent else None,
            "namespaces": results,
        }
    except Exception as e:
        log.error("list_namespaces failed", error=str(e))
        return {"error": f"Failed to list namespaces: {str(e)}"}


@mcp.tool()
async def assign_to_namespace(
    entity_name: str,
    namespace: str,
    primary: bool = True,
) -> dict[str, Any]:
    """
    Assign an existing entity to a namespace.

    Args:
        entity_name: Name of the entity to assign
        namespace: Namespace code to assign the entity to
        primary: Whether this is the entity's primary namespace

    Returns:
        Confirmation with entity and namespace details
    """
    if not ctx.graph:
        return {"error": "Graph not initialized"}

    try:
        # Find the entity
        entity = await ctx.graph.find_entity_by_name(entity_name)
        if not entity:
            return {"error": f"Entity '{entity_name}' not found in local graph"}

        # Verify namespace exists
        ns = await ctx.graph.get_namespace(namespace)
        if not ns:
            return {"error": f"Namespace '{namespace}' not found. Create it first."}

        # Assign entity to namespace
        await ctx.graph.assign_entity_to_namespace(
            entity_uuid=entity.uuid,
            namespace_code=namespace,
            primary=primary,
        )

        return {
            "entity": entity_name,
            "namespace": namespace,
            "primary": primary,
            "success": True,
        }
    except Exception as e:
        log.error("assign_to_namespace failed", error=str(e))
        return {"error": f"Failed to assign entity: {str(e)}"}


# =============================================================================
# Reasoning Tools
# =============================================================================


@mcp.tool()
async def compare_entities(
    entity_a: str,
    entity_b: str,
    store_similarity: bool = False,
) -> dict[str, Any]:
    """
    Compare two entities using UHT trait analysis.

    This is the workhorse tool for semantic reasoning. Returns:
    - **Jaccard similarity**: shared_traits / all_traits — the meaningful metric
    - **Hamming distance**: raw bit differences (0-32)
    - **Trait breakdown**: which traits are shared vs unique to each entity

    Use Jaccard to determine:
    - >70%: Very similar, safe to transfer properties between them
    - 50-70%: Moderate overlap, check specific traits before transferring
    - 30-50%: Limited overlap, different in significant ways
    - <30%: Fundamentally different entities
    - 0%: No shared traits — strong exclusion signal for disambiguation

    Inheritance check: If A has most of B's traits, A could be "a type of" B.

    Args:
        entity_a: First entity (will be classified if not already known)
        entity_b: Second entity (will be classified if not already known)
        store_similarity: If True and Jaccard >= 0.70, auto-store a computed
            SIMILAR_TO fact and create a SIMILAR_TO Neo4j edge

    Returns:
        hex_codes: {entity_a: "...", entity_b: "..."}
        similarity: {jaccard_similarity, hamming_distance, shared_traits, traits_a_only, traits_b_only}
        comparison: Human-readable analysis
    """
    if not ctx.engine:
        return {"error": "Engine not initialized"}

    result = await ctx.engine.reason(
        query=f"Compare {entity_a} and {entity_b}",
    )

    # Get detailed similarity metrics if we have both hex codes
    similarity_metrics = {}
    trait_diff = {}
    analysis = None
    if ctx.inference and len(result.hex_codes) >= 2:
        hex_a = result.hex_codes.get(entity_a)
        hex_b = result.hex_codes.get(entity_b)
        if hex_a and hex_b:
            analysis = ctx.inference.analyze_similarity(hex_a, hex_b, entity_a, entity_b)
            similarity_metrics = {
                "hamming_distance": analysis.hamming_distance,
                "jaccard_similarity": round(analysis.jaccard_similarity, 3),
                "simple_similarity": round(analysis.similarity_score, 3),
                "shared_trait_count": len(analysis.shared_traits),
                "traits_a_only_count": len(analysis.traits_a_only),
                "traits_b_only_count": len(analysis.traits_b_only),
            }

            # Build justification lookups from classifications (will hit cache)
            justifications_a: dict[int, str | None] = {}
            justifications_b: dict[int, str | None] = {}
            try:
                class_a = await _resolve_classification(entity_a)
                class_b = await _resolve_classification(entity_b)
                justifications_a = {t.bit_position: t.justification for t in class_a.traits}
                justifications_b = {t.bit_position: t.justification for t in class_b.traits}
            except Exception:
                pass  # Graceful fallback — justifications just won't be included

            # Include trait names with justifications for detailed analysis
            trait_diff = {
                "shared_traits": [
                    {
                        "name": TRAIT_NAMES.get(b, f"Bit {b}"),
                        "justification_a": justifications_a.get(b),
                        "justification_b": justifications_b.get(b),
                    }
                    for b in analysis.shared_traits
                ],
                "traits_a_only": [
                    {
                        "name": TRAIT_NAMES.get(b, f"Bit {b}"),
                        "justification": justifications_a.get(b),
                    }
                    for b in analysis.traits_a_only
                ],
                "traits_b_only": [
                    {
                        "name": TRAIT_NAMES.get(b, f"Bit {b}"),
                        "justification": justifications_b.get(b),
                    }
                    for b in analysis.traits_b_only
                ],
            }

    # Auto-store SIMILAR_TO computed fact if requested and similarity is high
    stored_fact_id = None
    if store_similarity and ctx.graph and analysis and similarity_metrics:
        jaccard = similarity_metrics.get("jaccard_similarity", 0)
        if jaccard >= 0.70:
            try:
                fact = await ctx.graph.store_fact(
                    subject=entity_a,
                    predicate="SIMILAR_TO",
                    obj=entity_b,
                    confidence=jaccard,
                    source="computed",
                    user_id="default",
                )
                stored_fact_id = fact.uuid

                # Also create SIMILAR_TO edge if both entities exist in graph
                entity_a_stored = await ctx.graph.find_entity_by_name(entity_a)
                entity_b_stored = await ctx.graph.find_entity_by_name(entity_b)
                if entity_a_stored and entity_b_stored:
                    await ctx.graph.create_similar_to_relationship(
                        source_uuid=entity_a_stored.uuid,
                        target_uuid=entity_b_stored.uuid,
                        similarity_score=jaccard,
                        shared_traits=sorted(analysis.shared_traits),
                    )
            except Exception as e:
                logger.warning("Failed to store similarity fact", error=str(e))

    return {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "hex_codes": result.hex_codes,
        "similarity": similarity_metrics,
        "trait_diff": trait_diff,
        "comparison": result.answer,
        "confidence": result.confidence,
        "trace_id": result.trace_id,
        "stored_fact_id": stored_fact_id,
    }


@mcp.tool()
async def batch_compare(
    entity: str,
    candidates: list[str],
    store_similarity: bool = False,
) -> dict[str, Any]:
    """
    Compare one entity against multiple candidates, returning ranked results.

    This is the efficient way to find the best match from a set of candidates.
    Returns all comparisons sorted by Jaccard similarity (highest first).

    Typical workflow:
    1. semantic_search("chair") → get candidate names
    2. batch_compare("chair", ["stool", "bench", "sofa", "table"]) → ranked Jaccard table

    Args:
        entity: The entity to compare against all candidates
        candidates: List of candidate entities to compare with (max 20)
        store_similarity: If True, auto-store computed SIMILAR_TO facts for
            candidates with Jaccard >= 0.70

    Returns:
        Ranked list of comparisons sorted by Jaccard similarity
    """
    if not ctx.uht or not ctx.inference:
        return {"error": "Engine not initialized"}

    candidates = candidates[:20]  # Limit to 20 candidates

    # First resolve the main entity classification (local-first)
    try:
        main_class = await _resolve_classification(entity)
    except Exception as e:
        return {"error": f"Failed to classify {entity}: {e}"}

    # Resolve all candidates in parallel (local-first for each)
    import asyncio
    async def classify_candidate(name: str):
        try:
            return (name, await _resolve_classification(name))
        except Exception:
            return (name, None)

    candidate_results = await asyncio.gather(
        *[classify_candidate(c) for c in candidates]
    )

    # Build justification lookup for the main entity
    main_justifications = {t.bit_position: t.justification for t in main_class.traits}

    # Compare and build results
    comparisons = []
    for name, classification in candidate_results:
        if classification is None:
            continue

        analysis = ctx.inference.analyze_similarity(
            main_class.hex_code,
            classification.hex_code,
            entity,
            name,
        )

        cand_justifications = {t.bit_position: t.justification for t in classification.traits}

        comparisons.append({
            "candidate": name,
            "hex_code": classification.hex_code,
            "jaccard_similarity": round(analysis.jaccard_similarity, 3),
            "hamming_distance": analysis.hamming_distance,
            "shared_traits": [
                {
                    "name": TRAIT_NAMES.get(b, f"Bit {b}"),
                    "justification_entity": main_justifications.get(b),
                    "justification_candidate": cand_justifications.get(b),
                }
                for b in analysis.shared_traits
            ],
            "traits_entity_only": [
                {
                    "name": TRAIT_NAMES.get(b, f"Bit {b}"),
                    "justification": main_justifications.get(b),
                }
                for b in analysis.traits_a_only
            ],
            "traits_candidate_only": [
                {
                    "name": TRAIT_NAMES.get(b, f"Bit {b}"),
                    "justification": cand_justifications.get(b),
                }
                for b in analysis.traits_b_only
            ],
        })

    # Sort by Jaccard (highest first)
    comparisons.sort(key=lambda x: x["jaccard_similarity"], reverse=True)

    # Auto-store SIMILAR_TO computed facts for high-similarity candidates
    stored_fact_ids: list[dict[str, str]] = []
    if store_similarity and ctx.graph:
        for comp in comparisons:
            jaccard = comp["jaccard_similarity"]
            if jaccard < 0.70:
                break  # Sorted desc, no more above threshold
            try:
                fact = await ctx.graph.store_fact(
                    subject=entity,
                    predicate="SIMILAR_TO",
                    obj=comp["candidate"],
                    confidence=jaccard,
                    source="computed",
                    user_id="default",
                )
                stored_fact_ids.append({
                    "candidate": comp["candidate"],
                    "fact_id": fact.uuid,
                })

                entity_a_stored = await ctx.graph.find_entity_by_name(entity)
                entity_b_stored = await ctx.graph.find_entity_by_name(comp["candidate"])
                if entity_a_stored and entity_b_stored:
                    shared_bits = [
                        t.bit_position
                        for t in main_class.traits
                        if t.present
                    ]
                    await ctx.graph.create_similar_to_relationship(
                        source_uuid=entity_a_stored.uuid,
                        target_uuid=entity_b_stored.uuid,
                        similarity_score=jaccard,
                        shared_traits=shared_bits,
                    )
            except Exception as e:
                logger.warning("Failed to store batch similarity fact", error=str(e))

    result_dict: dict[str, Any] = {
        "entity": entity,
        "hex_code": main_class.hex_code,
        "comparisons": comparisons,
        "best_match": comparisons[0]["candidate"] if comparisons else None,
        "best_jaccard": comparisons[0]["jaccard_similarity"] if comparisons else None,
    }
    if stored_fact_ids:
        result_dict["stored_similarity_facts"] = stored_fact_ids
    return result_dict


@mcp.tool()
async def infer_properties(
    entity: str,
) -> dict[str, Any]:
    """
    Infer properties of an entity based on its classification.

    Uses trait axioms to derive necessary and typical properties
    from the entity's UHT classification.

    Args:
        entity: Entity to analyze

    Returns:
        Inferred properties grouped by certainty level
    """
    if not ctx.engine:
        return {"error": "Engine not initialized"}

    result = await ctx.engine.reason(
        query=f"What can we infer about {entity}?",
    )

    return {
        "entity": entity,
        "hex_code": result.hex_codes.get(entity),
        "properties": result.inferred_properties,
        "answer": result.answer,
        "confidence": result.confidence,
    }


# =============================================================================
# Exploration Tools
# =============================================================================


@mcp.tool()
async def explore_neighborhood(
    entity: str,
    metric: str = "embedding",
    limit: int = 10,
    min_similarity: float = 0.3,
) -> dict[str, Any]:
    """
    [EXPERIMENTAL] Explore the semantic neighborhood of an entity.

    NOTE: This tool often returns empty results. For finding related entities,
    use semantic_search() instead — it searches the full 16k+ Factory corpus
    and reliably returns results with similarity scores.

    Args:
        entity: Entity to explore from
        metric: Similarity metric ("embedding", "hamming", or "hybrid")
        limit: Maximum neighbors to return (5-50)
        min_similarity: Minimum similarity threshold (0.0-1.0, default 0.3)

    Returns:
        Neighbors (may be empty due to indexing/threshold issues)
    """
    if not ctx.uht or not ctx.graph:
        return {"error": "UHT client not initialized"}

    limit = min(max(limit, 5), 50)
    min_similarity = min(max(min_similarity, 0.0), 1.0)

    try:
        entity_data = await _resolve_classification(entity)

        # Get neighborhood from UHT (requires Factory UUID)
        neighborhood = await ctx.uht.get_neighborhood(
            uuid=entity_data.uuid,
            metric=metric,
            k=limit,
            min_similarity=min_similarity,
        )

        # Store neighbor entities in local graph for future lookups
        neighbor_dicts = [
            {"uuid": n.uuid, "name": n.name, "hex_code": n.hex_code}
            for n in neighborhood.nodes
            if n.hex_code  # Only store nodes with hex codes
        ]
        await _store_factory_entities(neighbor_dicts, source="uht_factory")

        return {
            "entity": entity,
            "hex_code": entity_data.hex_code,
            "neighbors": [
                {
                    "name": node.name,
                    "hex_code": node.hex_code,
                    "type": node.node_type,
                }
                for node in neighborhood.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relationship": edge.relationship,
                    "weight": edge.weight,
                }
                for edge in neighborhood.edges
            ],
            "count": len(neighborhood.nodes),
        }
    except Exception as e:
        return {"error": f"Failed to explore neighborhood: {e}"}


@mcp.tool()
async def disambiguate_term(
    term: str,
    language: str = "en",
) -> dict[str, Any]:
    """
    Disambiguate a polysemous term into its different senses.

    Returns all known meanings of the word with their UHT classifications.

    Args:
        term: Word to disambiguate
        language: Language code (en, fr, de)

    Returns:
        List of senses with definitions and classifications
    """
    if not ctx.uht:
        return {"error": "UHT client not initialized"}

    result = await ctx.uht.disambiguate(term, language)

    return {
        "term": term,
        "language": result.language,
        "sense_count": len(result.senses),
        "senses": [
            {
                "definition": s.definition,
                "hex_code": s.hex_code,
                "examples": s.examples[:3] if s.examples else [],
            }
            for s in result.senses
        ],
    }


@mcp.tool()
async def get_semantic_triangle(
    text: str,
) -> dict[str, Any]:
    """
    Get the Ogden-Richards semantic triangle for a term.

    Decomposes a term into:
    - Symbol: The linguistic form (word/phrase, polysemy, intended sense)
    - Thought: The mental concept (definition, essential properties, category)
    - Referent: The real-world thing (description, typical instances, boundaries)

    Args:
        text: Text to analyze

    Returns:
        Semantic triangle decomposition
    """
    if not ctx.uht:
        return {"error": "UHT client not initialized"}

    triangle = await ctx.uht.get_semantic_triangle(text)

    return {
        "text": text,
        "symbol": {
            "form": triangle.symbol.form,
            "polysemy_detected": triangle.symbol.polysemy_detected,
            "intended_sense": triangle.symbol.intended_sense,
            "other_senses": triangle.symbol.other_senses,
        },
        "thought": {
            "definition": triangle.thought.definition,
            "essential_properties": triangle.thought.essential_properties,
            "category": triangle.thought.category,
        },
        "referent": {
            "description": triangle.referent.description,
            "typical_instances": triangle.referent.typical_instances,
            "ontological_status": triangle.referent.ontological_status,
        },
        "disambiguation_confidence": triangle.disambiguation_confidence,
        "enriched_context": triangle.enriched_context,
    }


@mcp.tool()
async def map_properties_to_traits(
    properties: list[str],
) -> dict[str, Any]:
    """Map natural language properties to candidate UHT trait bits.

    Takes a list of natural language property strings (from the semantic triangle,
    a domain expert, or any other source) and returns candidate trait mappings.

    This is the most composable approach — the client decides what to do with
    the mappings. They can feed them into classify_entity context, use them
    for audit, or let a domain expert override.

    Typical workflow:
    1. get_semantic_triangle("cognitive dissonance") → essential_properties
    2. map_properties_to_traits(essential_properties) → candidate traits
    3. classify_entity("cognitive dissonance", context + mapped priors)

    Args:
        properties: List of natural language property strings
                    (e.g., ["involves conflicting beliefs", "causes discomfort"])

    Returns:
        Per-property mappings with candidate trait bits, names, and rationale
    """
    mappings = await _map_properties_to_traits(properties)

    # Collect all unique suggested bits
    all_bits: set[int] = set()
    for m in mappings:
        for c in m["candidate_traits"]:
            all_bits.add(c["bit"])

    # Build suggested hex code from all mapped bits
    binary = ["0"] * 32
    for bit in all_bits:
        binary[bit - 1] = "1"
    binary_str = "".join(binary)
    suggested_hex = format(int(binary_str, 2), "08X")

    return {
        "properties": properties,
        "mappings": mappings,
        "all_candidate_bits": sorted(all_bits),
        "all_candidate_trait_names": [TRAIT_NAMES.get(b, f"Bit {b}") for b in sorted(all_bits)],
        "suggested_hex_code": suggested_hex,
    }


# =============================================================================
# Context Management Tools
# =============================================================================


@mcp.tool()
async def store_fact(
    subject: str,
    predicate: str,
    object_value: str,
    user_id: str = "default",
    namespace: str = "",
) -> dict[str, Any]:
    """
    Store a fact in the knowledge graph for later use.

    Facts are associated with a user and can be retrieved to provide
    context for future reasoning.

    Args:
        subject: Subject of the fact
        predicate: Relationship/predicate
        object_value: Object of the fact
        user_id: User identifier
        namespace: Optional namespace tag (e.g. "CLAUDE:projects")

    Returns:
        Confirmation of stored fact
    """
    if not ctx.graph:
        return {"error": "Graph not initialized"}

    from .graph.schema import PredicateTaxonomy

    if not PredicateTaxonomy.is_user_settable(predicate):
        return {
            "error": f"Predicate '{predicate}' is in the 'computed' category "
            "and cannot be set directly. Use a different predicate."
        }

    try:
        fact = await ctx.graph.store_fact(
            subject=subject,
            predicate=predicate,
            obj=object_value,
            confidence=1.0,
            source="asserted",
            user_id=user_id,
            namespace=namespace or None,
        )
    except ValueError as e:
        return {"error": str(e)}

    is_duplicate = getattr(fact, "_is_duplicate", False)
    return {
        "fact_id": fact.uuid,
        "subject": fact.subject,
        "predicate": fact.predicate,
        "object": fact.object,
        "category": fact.category,
        "is_custom_predicate": fact.is_custom_predicate,
        "bound": fact.bound,
        "subject_entity_uuid": fact.subject_entity_uuid,
        "object_entity_uuid": fact.object_entity_uuid,
        "namespace": fact.namespace,
        "stored": not is_duplicate,
        "duplicate": is_duplicate,
    }


@mcp.tool()
async def store_facts_bulk(
    facts: list[dict[str, str]],
) -> dict[str, Any]:
    """
    Store multiple facts in a single call.

    Accepts an array of fact objects and stores them sequentially.
    Each fact follows the same behaviour as store_fact. Returns an
    array of results in the same order as the input.

    Args:
        facts: Array of fact objects, each with keys:
               subject (required), predicate (required),
               object_value (required), user_id (optional, defaults to "default"),
               namespace (optional)

    Returns:
        Array of results in the same order as input
    """
    if not ctx.graph:
        return {"error": "Graph not initialized"}

    if not facts:
        return {"error": "facts array is empty"}

    from .graph.schema import PredicateTaxonomy

    results = []
    stored_count = 0
    error_count = 0

    for i, item in enumerate(facts):
        # Validate required fields
        subject = item.get("subject", "")
        predicate = item.get("predicate", "")
        object_value = item.get("object_value", "")
        user_id = item.get("user_id", "default")
        ns = item.get("namespace", "")

        if not subject or not predicate or not object_value:
            results.append({
                "index": i,
                "error": "Missing required field(s): subject, predicate, object_value",
            })
            error_count += 1
            continue

        if not PredicateTaxonomy.is_user_settable(predicate):
            results.append({
                "index": i,
                "error": f"Predicate '{predicate}' is in the 'computed' category",
            })
            error_count += 1
            continue

        try:
            fact = await ctx.graph.store_fact(
                subject=subject,
                predicate=predicate,
                obj=object_value,
                confidence=1.0,
                source="asserted",
                user_id=user_id,
                namespace=ns or None,
            )
            is_duplicate = getattr(fact, "_is_duplicate", False)
            results.append({
                "index": i,
                "fact_id": fact.uuid,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
                "category": fact.category,
                "bound": fact.bound,
                "namespace": fact.namespace,
                "stored": not is_duplicate,
                "duplicate": is_duplicate,
            })
            stored_count += 1
        except ValueError as e:
            results.append({"index": i, "error": str(e)})
            error_count += 1

    return {
        "total": len(facts),
        "stored": stored_count,
        "errors": error_count,
        "results": results,
    }


@mcp.tool()
async def upsert_fact(
    subject: str,
    predicate: str,
    object_value: str,
    user_id: str = "default",
    namespace: str = "",
) -> dict[str, Any]:
    """
    Upsert a fact: match on (subject, predicate, user_id).

    If a fact with that combination already exists, update its object_value.
    If not, create a new fact. Returns the fact with a flag indicating
    whether it was created or updated.

    Args:
        subject: Subject of the fact
        predicate: Relationship/predicate
        object_value: Object of the fact
        user_id: User identifier
        namespace: Optional namespace tag (e.g. "CLAUDE:projects")

    Returns:
        Fact details with created/updated flags
    """
    if not ctx.graph:
        return {"error": "Graph not initialized"}

    from .graph.schema import PredicateTaxonomy

    if not PredicateTaxonomy.is_user_settable(predicate):
        return {
            "error": f"Predicate '{predicate}' is in the 'computed' category "
            "and cannot be set directly. Use a different predicate."
        }

    try:
        fact, was_created = await ctx.graph.upsert_fact(
            subject=subject,
            predicate=predicate,
            obj=object_value,
            confidence=1.0,
            source="asserted",
            user_id=user_id,
            namespace=namespace or None,
        )
    except ValueError as e:
        return {"error": str(e)}

    return {
        "fact_id": fact.uuid,
        "subject": fact.subject,
        "predicate": fact.predicate,
        "object": fact.object,
        "category": fact.category,
        "is_custom_predicate": fact.is_custom_predicate,
        "bound": fact.bound,
        "subject_entity_uuid": fact.subject_entity_uuid,
        "object_entity_uuid": fact.object_entity_uuid,
        "namespace": fact.namespace,
        "created": was_created,
        "updated": not was_created,
    }


@mcp.tool()
async def query_facts(
    subject: str = "",
    object_value: str = "",
    predicate: str = "",
    category: str = "",
    user_id: str = "",
    namespace: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    """
    Query facts with flexible filters. At least one filter is required.

    Filters can be combined. Category must be one of: compositional, causal,
    temporal, functional, associative, computed.

    Example queries:
    - "What is spark plug PART_OF?" -> query_facts(subject="spark plug", predicate="PART_OF")
    - "All parts of engine?" -> query_facts(object_value="engine", category="compositional")
    - "What does virus cause?" -> query_facts(subject="virus", category="causal")

    Args:
        subject: Filter by fact subject (case-insensitive)
        object_value: Filter by fact object (case-insensitive)
        predicate: Filter by exact predicate
        category: Filter by predicate category
        user_id: Scope to facts owned by this user
        namespace: Filter by namespace (e.g., "SE", "SE:aerospace"). Includes descendants.
        limit: Maximum results (1-100)

    Returns:
        List of matching facts with binding info
    """
    if not ctx.graph:
        return {"error": "Graph not initialized"}

    if not any([subject, object_value, predicate, category, user_id, namespace]):
        return {
            "error": "At least one filter (subject, object_value, predicate, "
            "category, user_id, namespace) is required"
        }

    limit = max(1, min(100, limit))

    facts = await ctx.graph.query_facts(
        subject=subject or None,
        object_value=object_value or None,
        predicate=predicate or None,
        category=category or None,
        user_id=user_id or None,
        namespace=namespace or None,
        limit=limit,
    )

    return {
        "count": len(facts),
        "facts": [
            {
                "uuid": f.uuid,
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "confidence": f.confidence,
                "source": f.source,
                "category": f.category,
                "is_custom_predicate": f.is_custom_predicate,
                "bound": f.bound,
                "subject_entity_uuid": f.subject_entity_uuid,
                "object_entity_uuid": f.object_entity_uuid,
                "created_at": str(f.created_at),
            }
            for f in facts
        ],
    }


@mcp.tool()
async def update_fact(
    fact_id: str,
    subject: str = "",
    predicate: str = "",
    object_value: str = "",
) -> dict[str, Any]:
    """
    Update an existing fact's fields.

    Only provided (non-empty) fields are updated. The fact's category
    is automatically re-computed if the predicate changes. Entity binding
    is re-attempted if subject or object changes.

    Args:
        fact_id: UUID of the fact to update
        subject: New subject (empty to keep current)
        predicate: New predicate (empty to keep current)
        object_value: New object (empty to keep current)

    Returns:
        Updated fact details
    """
    if not ctx.graph:
        return {"error": "Graph not initialized"}

    if predicate:
        from .graph.schema import PredicateTaxonomy

        if not PredicateTaxonomy.is_user_settable(predicate):
            return {"error": f"Predicate '{predicate}' is in the 'computed' category"}

    try:
        updated = await ctx.graph.update_fact(
            uuid=fact_id,
            subject=subject or None,
            predicate=predicate or None,
            obj=object_value or None,
        )
    except ValueError as e:
        return {"error": str(e)}

    if not updated:
        return {"error": f"Fact '{fact_id}' not found"}

    return {
        "fact_id": updated.uuid,
        "subject": updated.subject,
        "predicate": updated.predicate,
        "object": updated.object,
        "confidence": updated.confidence,
        "category": updated.category,
        "is_custom_predicate": updated.is_custom_predicate,
        "bound": updated.bound,
        "updated": True,
    }


@mcp.tool()
async def delete_fact(
    fact_id: str,
) -> dict[str, Any]:
    """
    Delete a fact from the knowledge graph.

    This permanently removes the fact and all its relationships
    (user ownership, entity bindings, reasoning trace links).

    Args:
        fact_id: UUID of the fact to delete

    Returns:
        Confirmation of deletion
    """
    if not ctx.graph:
        return {"error": "Graph not initialized"}

    deleted = await ctx.graph.delete_fact(fact_id)

    if not deleted:
        return {"error": f"Fact '{fact_id}' not found"}

    return {"fact_id": fact_id, "deleted": True}


@mcp.tool()
async def get_user_context(
    user_id: str = "default",
) -> dict[str, Any]:
    """
    Retrieve stored facts and preferences for a user.

    Facts are grouped by predicate category (compositional, causal,
    temporal, functional, associative, computed) with a summary of
    totals, bound/unbound counts, and breakdown by source.

    Args:
        user_id: User identifier

    Returns:
        User's stored facts and preferences
    """
    if not ctx.graph:
        return {"error": "Graph not initialized"}

    grouped = await ctx.graph.get_user_facts_grouped(user_id, limit=100)
    preferences = await ctx.graph.get_user_preferences(user_id)

    return {
        "user_id": user_id,
        "facts": grouped["facts_by_category"],
        "summary": grouped["summary"],
        "preferences": preferences,
    }


@mcp.tool()
async def get_namespace_context(
    namespace: str,
    user_id: str = "",
) -> dict[str, Any]:
    """
    Get all entities and facts under a namespace subtree.

    Returns every entity assigned to the given namespace (or any descendant
    namespace), along with their hex codes and all facts whose subject is
    one of those entities. Essentially a single-call dump of everything
    stored under a namespace subtree.

    Args:
        namespace: Namespace code (e.g. "CLAUDE", "SE:aerospace")
        user_id: Optional user ID to scope facts to (empty for all facts)

    Returns:
        Entities with hex codes, facts, and namespace tree
    """
    if not ctx.graph:
        return {"error": "Graph not initialized"}

    # Validate namespace exists
    ns = await ctx.graph.get_namespace(namespace)
    if not ns:
        return {"error": f"Namespace '{namespace}' not found"}

    # Get entities + facts
    context = await ctx.graph.get_namespace_context(
        namespace_code=namespace,
        user_id=user_id or None,
    )

    # Get namespace tree (include_descendants with *0.. already includes root)
    all_namespaces = await ctx.graph.list_namespaces(
        parent_code=namespace,
        include_descendants=True,
    )

    return {
        "namespace": namespace,
        "namespaces": [
            {
                "code": n.code,
                "name": n.name,
                "description": n.description,
                "is_root": n.is_root,
            }
            for n in all_namespaces
        ],
        "entities": [
            {
                "uuid": e.uuid,
                "name": e.name,
                "hex_code": e.hex_code,
                "description": e.description,
                "source": e.source,
            }
            for e in context["entities"]
        ],
        "facts": [
            {
                "uuid": f.uuid,
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "confidence": f.confidence,
                "source": f.source,
                "category": f.category,
                "namespace": f.namespace,
                "bound": f.bound,
                "created_at": str(f.created_at),
            }
            for f in context["facts"]
        ],
        "summary": {
            "namespace_count": len(all_namespaces),
            "entity_count": len(context["entities"]),
            "fact_count": len(context["facts"]),
        },
    }


@mcp.tool()
async def semantic_search(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Search for entities using semantic similarity (embeddings).

    Unlike text search, this finds conceptually similar entities even if
    they don't share exact words. For example, "transportation" might find
    "bicycle", "car", "airplane".

    Args:
        query: Natural language search query
        limit: Maximum number of results (1-50)

    Returns:
        List of semantically similar entities with their classifications
    """
    if not ctx.uht:
        return {"error": "UHT client not initialized"}

    limit = max(1, min(50, limit))

    try:
        results = await ctx.uht.semantic_search(query=query, limit=limit)

        # Store returned entities in local graph for future lookups
        await _store_factory_entities(results, source="uht_factory")

        return {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "name": r.name,
                    "hex_code": r.hex_code,
                    "description": r.description,
                    "uuid": r.uuid,
                    "similarity_score": r.similarity_score,
                }
                for r in results
            ],
        }
    except Exception as e:
        return {"error": f"Semantic search failed: {e}"}


@mcp.tool()
async def get_patterns() -> dict[str, Any]:
    """
    Get UHT reasoning patterns for orchestrating tools.

    Returns documented patterns for composing UHT tools to answer
    complex questions. Each pattern shows which tools to call and
    how to interpret results.

    Returns:
        Reasoning patterns with examples and interpretation guides
    """
    return {
        "patterns": [
            {
                "name": "Analogical Transfer",
                "question_type": "What is X most like?",
                "steps": [
                    "semantic_search(X) to get candidates",
                    "compare_entities(X, candidate) for top 3-5",
                    "Rank by Jaccard similarity (highest = best analogy)",
                ],
                "interpretation": {
                    ">0.70": "Strong analogy, safe to transfer properties",
                    "0.50-0.70": "Moderate analogy, check specific traits",
                    "<0.50": "Weak analogy, use with caveats",
                },
            },
            {
                "name": "Category Membership",
                "question_type": "Is X a Y? (e.g., Is a virus alive?)",
                "steps": [
                    "classify_entity(X)",
                    "classify_entity(exemplar) for 2-3 clear examples of Y",
                    "compare_entities(X, exemplar) for each",
                    "If avg Jaccard > 0.50, X is likely a Y",
                ],
                "interpretation": {
                    ">0.50 with multiple exemplars": "Likely member",
                    "0.30-0.50": "Boundary case, explain differing traits",
                    "<0.30": "Probably not a member",
                },
            },
            {
                "name": "Property Inheritance",
                "question_type": "Can X do what Y does?",
                "steps": [
                    "compare_entities(X, Y)",
                    "Identify trait relevant to property P",
                    "Check if trait in shared_traits or traits_Y_only",
                ],
                "interpretation": {
                    "trait in shared_traits": "Property likely transfers",
                    "trait in traits_Y_only": "Property doesn't transfer",
                },
            },
            {
                "name": "Disambiguation",
                "question_type": "Which sense of word W fits context?",
                "steps": [
                    "disambiguate_term(W) to get senses with hex codes",
                    "classify_entity(context_word) for key context words",
                    "compare_entities(sense, context_word) for each",
                    "Highest cumulative Jaccard = best-fitting sense",
                ],
                "interpretation": {
                    "big Jaccard gap": "Clear winner",
                    "similar scores": "Ambiguous, need more context",
                    "0% with one sense": "Strong exclusion signal",
                },
            },
            {
                "name": "Ontological Surprise",
                "question_type": "What's unexpectedly similar across domains?",
                "steps": [
                    "list_entities() or semantic_search() for corpus",
                    "compare_entities(X, Y) across domain boundaries",
                    "Flag pairs with Jaccard > 0.50 from different domains",
                ],
                "interpretation": {
                    "cross-domain >0.50": "Genuine structural similarity",
                    "shared traits explain WHY": "Not metaphor, real overlap",
                },
            },
        ],
        "key_metrics": {
            "jaccard_similarity": "shared / (shared + unique_A + unique_B) — semantic overlap",
            "hamming_distance": "bit differences (0-32) — raw divergence",
            "inheritance_score": "child_has_parent_traits / parent_traits — IS_A strength",
        },
        "jaccard_guide": {
            "0.80+": "Nearly identical (sofa vs couch)",
            "0.60-0.79": "Same category (chair vs stool)",
            "0.40-0.59": "Related but different (chair vs table)",
            "0.20-0.39": "Weak connection (chair vs tree)",
            "0.00-0.19": "Fundamentally different",
            "0.00": "No overlap — strong exclusion signal",
        },
        "fact_management": {
            "name": "Relational Knowledge",
            "description": "Store, query, and manage facts (relationships between entities) "
            "with a controlled predicate taxonomy and automatic entity binding.",
            "predicate_categories": {
                "compositional": "PART_OF, CONTAINS, MADE_OF, COMPONENT_OF",
                "causal": "CAUSES, ENABLES, PREVENTS, INHIBITS",
                "temporal": "PRECEDES, FOLLOWS, DURING, CONCURRENT_WITH",
                "functional": "TREATS, REGULATES, PRODUCES, CONSUMES, TRANSFORMS",
                "associative": "RELATED_TO, USED_WITH, DERIVED_FROM, ANALOGOUS_TO (+ custom predicates)",
                "computed": "SIMILAR_TO, INHERITS_FROM, DISTINCT_FROM (system-only)",
            },
            "tools": {
                "store_fact(subject, predicate, object_value)": "Store a typed, categorized fact. Auto-binds to entities if they exist.",
                "query_facts(subject?, object_value?, predicate?, category?)": "Bidirectional lookup by any combination of filters.",
                "update_fact(fact_id, ...)": "Modify an existing fact. Re-categorizes and re-binds.",
                "delete_fact(fact_id)": "Remove a fact and all its edges.",
                "get_user_context(user_id)": "All facts grouped by category with summary stats.",
            },
            "entity_binding": "When both subject and object match classified entities, "
            "a RELATED_TO edge is created between them in Neo4j. "
            "Classifying an entity retroactively binds any pending facts.",
            "store_similarity": "compare_entities(A, B, store_similarity=True) auto-stores a "
            "computed SIMILAR_TO fact when Jaccard >= 0.70. Also works with batch_compare.",
        },
    }


@mcp.tool()
async def get_info() -> dict[str, Any]:
    """
    Get information about the UHT system and this MCP server.

    Returns an overview of what the Universal Hex Taxonomy is, how it works,
    and how to use this MCP server effectively.

    Returns:
        System description, key concepts, and usage guidance
    """
    return {
        "name": "UHT Substrate Agent",
        "version": "0.1.0",
        "description": "An MCP server providing semantic reasoning grounded in the Universal Hex Taxonomy (UHT)",
        "what_is_uht": {
            "summary": "A 32-bit classification system that encodes the essential properties of any concept as an 8-character hex code",
            "purpose": "Provides a universal coordinate system for meaning, enabling precise semantic comparison, property inference, and analogical reasoning",
            "key_insight": "Two entities with similar hex codes share similar properties — not by coincidence, but because the classification captures what they fundamentally ARE",
        },
        "the_32_traits": {
            "overview": "Every entity is evaluated against 32 binary traits, each asking a yes/no question about the entity's nature",
            "layers": {
                "physical (bits 1-8)": "Material existence — Does it have mass? Boundaries? Location? Energy?",
                "functional (bits 9-16)": "Capabilities — Can it act? Be consumed? Store state? Interface with things?",
                "abstract (bits 17-24)": "Conceptual nature — Is it symbolic? Rule-governed? Temporal? Compositional?",
                "social (bits 25-32)": "Human context — Is it cultural? Economic? Institutional? Political?",
            },
            "encoding": "32 bits → 8 hex characters. Example: 'hammer' = C6880008 (physical tool, no abstract/social traits)",
        },
        "why_jaccard_not_hamming": {
            "hamming_distance": "Counts differing bits (0-32). Problem: treats all differences equally",
            "jaccard_similarity": "shared_traits / all_traits. Better: measures meaningful overlap",
            "example": "A rock (8 traits) vs democracy (8 traits) — Hamming sees 16 differences, but Jaccard correctly shows 0% overlap because they share ZERO traits",
            "rule": "Use Jaccard for semantic similarity. Hamming is just raw divergence.",
        },
        "core_tools": {
            "reliable": [
                {"tool": "classify_entity", "use": "Get hex code and inferred properties for any concept"},
                {"tool": "compare_entities", "use": "Compare two entities — Jaccard, shared/unique traits. Set store_similarity=True to persist high-similarity results."},
                {"tool": "batch_compare", "use": "Compare one entity against many candidates. Set store_similarity=True to persist."},
                {"tool": "infer_properties", "use": "Derive properties from classification via trait axioms"},
                {"tool": "semantic_search", "use": "Find similar entities in 16k+ Factory corpus (USE THIS for analogies)"},
                {"tool": "disambiguate_term", "use": "Get word senses for polysemous terms"},
                {"tool": "list_entities", "use": "Browse local knowledge graph"},
                {"tool": "get_patterns", "use": "Get reasoning patterns for complex questions"},
                {"tool": "get_traits", "use": "Get definitions of all 32 traits"},
                {"tool": "get_info", "use": "Get this overview"},
            ],
            "fact_management": [
                {"tool": "store_fact", "use": "Store a typed fact (auto-categorized, auto-bound to entities)"},
                {"tool": "upsert_fact", "use": "Idempotent store: match on (subject, predicate, user_id), update object_value if exists, create if not"},
                {"tool": "store_facts_bulk", "use": "Store multiple facts in one call — array of {subject, predicate, object_value, user_id?}"},
                {"tool": "query_facts", "use": "Query facts by subject, object, predicate, category, or namespace (includes descendants)"},
                {"tool": "update_fact", "use": "Modify a fact (re-categorizes and re-binds)"},
                {"tool": "delete_fact", "use": "Remove a fact and all its edges"},
                {"tool": "get_user_context", "use": "All facts grouped by category with summary stats"},
                {"tool": "get_namespace_context", "use": "Single-call dump of all entities and facts under a namespace subtree"},
            ],
            "also_reliable": [
                {"tool": "find_similar_entities", "use": "Find entities similar to a given entity by UUID"},
                {"tool": "search_by_traits", "use": "Search Factory corpus by trait pattern (16k+ entities)"},
            ],
            "experimental": [
                {"tool": "explore_neighborhood", "status": "Graph exploration — may return sparse results"},
            ],
        },
        "recommended_workflow": {
            "find_analogies": "semantic_search(X) → compare_entities(X, candidate) → rank by Jaccard",
            "classify_and_compare": "classify_entity(A), classify_entity(B) → compare_entities(A, B)",
            "check_category": "classify_entity(target) + classify_entity(exemplars) → compare all",
        },
        "when_to_use_uht": [
            "Analogical reasoning: 'What is X most like?'",
            "Category membership: 'Is X a Y?' (boundary cases like 'Is a virus alive?')",
            "Property transfer: 'Can X do what Y does?'",
            "Disambiguation: 'Which sense of word W fits this context?'",
            "Ontological discovery: 'What's unexpectedly similar across domains?'",
        ],
        "factory_corpus": "16,000+ pre-classified entities at factory.universalhex.org",
    }


@mcp.tool()
async def get_traits() -> dict[str, Any]:
    """
    Get definitions of all 32 UHT traits.

    Returns the complete trait ontology organized by layer, with
    descriptions of what each trait means and examples.

    Returns:
        All 32 trait definitions grouped by layer
    """
    if not ctx.uht:
        return {"error": "UHT client not initialized"}

    try:
        traits = await ctx.uht.get_traits()

        # Group by layer
        layers = {
            "physical": [],
            "functional": [],
            "abstract": [],
            "social": [],
        }

        for trait in traits:
            layer_name = trait.layer.value
            layers[layer_name].append({
                "bit": trait.bit_position,
                "name": trait.name,
                "short_description": trait.short_description,
                "expanded_definition": trait.expanded_definition,
                "url": trait.url,
            })

        return {
            "total_traits": 32,
            "encoding": "Each trait = 1 bit. 32 bits = 8 hex characters.",
            "layers": {
                "physical (bits 1-8)": {
                    "description": "Material and spatial properties",
                    "traits": layers["physical"],
                },
                "functional (bits 9-16)": {
                    "description": "Capabilities and behaviors",
                    "traits": layers["functional"],
                },
                "abstract (bits 17-24)": {
                    "description": "Conceptual and symbolic properties",
                    "traits": layers["abstract"],
                },
                "social (bits 25-32)": {
                    "description": "Cultural and institutional properties",
                    "traits": layers["social"],
                },
            },
        }
    except Exception as e:
        return {"error": f"Failed to fetch traits: {e}"}


# =============================================================================
# Information Resources
# =============================================================================


@mcp.resource("uht://traits")
async def get_all_traits() -> str:
    """Get all 32 UHT trait definitions."""
    if not ctx.uht:
        return "UHT client not initialized"

    traits = await ctx.uht.get_traits()

    lines = ["# UHT Trait Definitions", ""]
    for trait in traits:
        lines.extend([
            f"## Bit {trait.bit_position}: {trait.name}",
            f"Layer: {trait.layer.value}",
            f"Description: {trait.description}",
            "",
        ])

    return "\n".join(lines)


@mcp.resource("uht://axioms")
async def get_all_axioms() -> str:
    """Get all trait axioms."""
    if not ctx.inference:
        return "Inference engine not initialized"

    all_traits = ctx.inference.axioms.get_all_traits()

    lines = ["# UHT Trait Axioms", ""]
    for trait in all_traits:
        lines.extend([
            f"## {trait.name} (Bit {trait.bit_position})",
            "",
        ])
        for axiom in trait.axioms:
            lines.append(f"- [{axiom.axiom_type}] {axiom.statement}")
            lines.append(f"  Property: {axiom.property} (confidence: {axiom.confidence:.0%})")
        lines.append("")

    return "\n".join(lines)


@mcp.resource("uht://heuristics")
async def get_all_heuristics() -> str:
    """Get all reasoning heuristics."""
    if not ctx.inference:
        return "Inference engine not initialized"

    heuristics = ctx.inference.heuristics.get_all()

    lines = ["# Reasoning Heuristics", ""]
    for h in heuristics:
        lines.extend([
            f"## {h.name} (Priority: {h.priority})",
            f"**Description:** {h.description}",
            f"**Applicability:** {h.applicability}",
            "",
        ])

    return "\n".join(lines)


@mcp.resource("uht://ontology")
async def get_all_ontology() -> str:
    """Get all ontological commitments.

    Ontological commitments are foundational assumptions about how entities
    relate to each other and how properties transfer between them.
    """
    if not ctx.inference:
        return "Inference engine not initialized"

    commitments = ctx.inference.ontology.get_all()

    if not commitments:
        return "# Ontological Commitments\n\nNo commitments loaded."

    lines = ["# Ontological Commitments", ""]

    # Group by category
    by_category: dict[str, list] = {}
    for c in commitments:
        if c.category not in by_category:
            by_category[c.category] = []
        by_category[c.category].append(c)

    for category, cat_commitments in by_category.items():
        lines.append(f"## {category.replace('_', ' ').title()}")
        lines.append("")
        for c in cat_commitments:
            lines.append(f"### {c.name}")
            lines.append(f"**Statement:** {c.statement}")
            if c.confidence < 1.0:
                lines.append(f"**Confidence:** {c.confidence:.0%}")
            if c.implications:
                lines.append("**Implications:**")
                for impl in c.implications:
                    lines.append(f"  - {impl}")
            lines.append("")

    return "\n".join(lines)


@mcp.resource("graph://statistics")
async def get_graph_statistics() -> str:
    """Get knowledge graph statistics."""
    if not ctx.graph:
        return "Graph not initialized"

    stats = await ctx.graph.get_statistics()

    lines = ["# Knowledge Graph Statistics", ""]
    lines.append("## Node Counts")
    for label, count in stats.get("nodes", {}).items():
        lines.append(f"- {label}: {count}")

    lines.append("")
    lines.append("## Relationship Counts")
    for rel_type, count in stats.get("relationships", {}).items():
        lines.append(f"- {rel_type}: {count}")

    return "\n".join(lines)


@mcp.resource("uht://patterns")
async def get_reasoning_patterns() -> str:
    """Get orchestration patterns for UHT-based reasoning."""
    return """# UHT Reasoning Patterns

These patterns show how to compose UHT Substrate tools for common reasoning tasks.
Each pattern uses primitive tools (classify_entity, compare_entities, etc.) to answer
higher-level questions that would otherwise require ad-hoc reasoning.

---

## Pattern 1: Analogical Transfer

**Question type:** "What is X most like?" / "Find me something similar to X"

**When to use:** User wants conceptually similar entities, needs analogies, or wants
to understand X by comparison.

**Steps:**
1. `semantic_search(X)` — get embedding-based candidates
2. For top 3-5 candidates: `compare_entities(X, candidate)`
3. Rank by **Jaccard similarity** (not Hamming distance)
4. Highest Jaccard = best structural analogy

**Example:**
```
User: "What's a chair most like?"
1. semantic_search("chair") → [stool, bench, sofa, throne, seat]
2. compare_entities("chair", "stool") → Jaccard 0.72
3. compare_entities("chair", "sofa") → Jaccard 0.68
4. Answer: "A chair is most like a stool (72% trait overlap)"
```

**Interpretation:**
- Jaccard > 0.7: Strong analogy, safe to transfer properties
- Jaccard 0.5-0.7: Moderate analogy, check specific traits
- Jaccard < 0.5: Weak analogy, only use with caveats

---

## Pattern 2: Category Membership

**Question type:** "Is X a Y?" / "Does X belong to category Y?"

**When to use:** Boundary cases, novel entities, philosophical questions like
"Is a virus alive?" or "Is a corporation a person?"

**Steps:**
1. `classify_entity(X)` — get target classification
2. `classify_entity(exemplar_1)`, `classify_entity(exemplar_2)` — classify 2-3 clear exemplars of Y
3. `compare_entities(X, exemplar)` for each
4. If X clusters with exemplars (avg Jaccard > 0.5), X is likely a Y

**Example:**
```
User: "Is a virus alive?"
1. classify_entity("virus") → hex code
2. classify_entity("bacterium") → living exemplar 1
3. classify_entity("cat") → living exemplar 2
4. compare_entities("virus", "bacterium") → Jaccard 0.38
5. compare_entities("virus", "cat") → Jaccard 0.25
6. Answer: "Virus shares only 38%/25% traits with living things.
   It's a boundary case — alive in some ways, not in others."
```

**Interpretation:**
- If X has Jaccard > 0.5 with multiple exemplars → likely member
- If X has Jaccard 0.3-0.5 → boundary case, explain which traits differ
- If X has Jaccard < 0.3 → probably not a member

---

## Pattern 3: Property Inheritance

**Question type:** "Can X do what Y does?" / "Does X have property P?"

**When to use:** Checking if properties transfer between similar entities,
or inferring capabilities from known similar entities.

**Steps:**
1. `compare_entities(X, Y)` — get detailed trait comparison
2. Identify the trait relevant to property P
3. Check if trait is in `shared_traits` or `traits_Y_only`

**Example:**
```
User: "Can a robot feel emotions like a human?"
1. compare_entities("robot", "human")
2. Look for emotional/consciousness traits (Abstract layer, bits 17-24)
3. If "sentient" in traits_Y_only → "Robots lack the sentience trait humans have"
4. If "sentient" in shared_traits → property transfers (unlikely for robots)
```

**Interpretation:**
- Trait in shared_traits → property likely transfers
- Trait in unique_Y_only → property doesn't transfer (explain why)
- Neither has trait → property absent from both

---

## Pattern 4: Disambiguation

**Question type:** "Which sense of word W fits context C?"

**When to use:** Polysemous words where context determines meaning
(bank, crane, Python, etc.)

**Steps:**
1. `disambiguate_term(W)` — get all senses with hex codes
2. `classify_entity(context_word)` for 2-3 key context words
3. `compare_entities(sense, context_word)` for each sense × context word
4. Sum Jaccard scores; highest total = best-fitting sense

**Example:**
```
User: "Which 'bank' in 'The crane flew over the bank'?"
1. disambiguate_term("bank") → [financial_bank: 0x..., river_bank: 0x...]
2. classify_entity("crane") → bird sense (assuming context suggests bird)
3. compare_entities("financial_bank", "crane") → Jaccard 0.05
4. compare_entities("river_bank", "crane") → Jaccard 0.28
5. Answer: "River bank — it shares more traits with the bird sense of crane"
```

**Interpretation:**
- Big Jaccard gap between senses → clear winner
- Similar Jaccard → ambiguous, may need more context
- 0% Jaccard with one sense → strong exclusion signal

---

## Pattern 5: Ontological Surprise

**Question type:** "What's unexpectedly similar?" / "Find surprising connections"

**When to use:** Discovery mode, finding non-obvious structural similarities
across domain boundaries, generating insights.

**Steps:**
1. `list_entities()` or `semantic_search()` — get corpus
2. `compare_entities(X, Y)` across domain boundaries
3. Flag pairs with high Jaccard (>0.5) from different domains

**Example:**
```
User: "What's surprising about language and DNA?"
1. compare_entities("language", "DNA")
2. Result: Jaccard 0.56 — unexpectedly high for different domains
3. Shared traits: symbolic, compositional, rule-governed, information-carrying
4. Answer: "Language and DNA share 56% of traits — both are symbolic systems
   with compositional rules that carry information. This isn't a metaphor;
   they're structurally similar."
```

**Interpretation:**
- Cross-domain Jaccard > 0.5 → genuine structural similarity
- Shared traits explain WHY they're similar
- Can use this for analogical reasoning between domains

---

## Key Metrics Reference

| Metric | Definition | Use For |
|--------|------------|---------|
| Jaccard Similarity | shared / (shared + unique_A + unique_B) | Semantic overlap |
| Hamming Distance | Count of differing bits (0-32) | Raw divergence |
| Inheritance Score | child_has_parent_traits / parent_traits | IS_A relationships |

## Jaccard Interpretation Guide

| Score | Interpretation | Example |
|-------|---------------|---------|
| 0.80+ | Nearly identical | "sofa" vs "couch" |
| 0.60-0.79 | Same category | "chair" vs "stool" |
| 0.40-0.59 | Related, different | "chair" vs "table" |
| 0.20-0.39 | Weak connection | "chair" vs "tree" |
| 0.00-0.19 | Fundamentally different | "chair" vs "democracy" |
| 0.00 | No overlap | Strong exclusion signal |
"""


# =============================================================================
# REST API (for ChatGPT, curl, etc.)
# =============================================================================


class ClassifyRequest(BaseModel):
    """Request model for classify endpoint."""
    entity: str
    context: str = ""
    use_semantic_priors: bool = False


class CompareRequest(BaseModel):
    """Request model for compare endpoint."""
    entity_a: str
    entity_b: str


class SearchRequest(BaseModel):
    """Request model for semantic search endpoint."""
    query: str
    limit: int = 10


class DisambiguateRequest(BaseModel):
    """Request model for disambiguate endpoint."""
    term: str
    language: str = "en"


class BatchCompareRequest(BaseModel):
    """Request model for batch compare endpoint."""
    entity: str
    candidates: list[str]


class MapPropertiesToTraitsRequest(BaseModel):
    """Request model for map-properties-to-traits endpoint."""
    properties: list[str]


# Create FastAPI app for REST endpoints
rest_api = FastAPI(
    title="UHT Substrate API",
    description="REST API for Universal Hex Taxonomy semantic reasoning",
    version="0.1.0",
)

# Add CORS middleware for browser access
rest_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@rest_api.get("/")
async def api_root():
    """API root with links to endpoints."""
    return {
        "name": "UHT Substrate REST API",
        "endpoints": {
            "GET /info": "System information and overview",
            "GET /traits": "All 32 trait definitions",
            "GET /patterns": "Reasoning patterns for tool orchestration",
            "POST /classify": "Classify an entity",
            "POST /compare": "Compare two entities",
            "POST /batch-compare": "Compare entity against multiple candidates",
            "POST /search": "Semantic search for similar entities",
            "POST /disambiguate": "Get word senses for polysemous terms",
            "POST /map-properties-to-traits": "Map properties to UHT trait bits",
            "POST /semantic-triangle": "Get Ogden-Richards semantic triangle",
            "GET /entities": "List entities in local graph",
            "GET /search/traits": "Search entities by trait pattern",
            "POST /entities/delete": "Delete entity from local graph",
            "POST /entities/infer": "Infer properties from classification",
            "POST /entities/explore": "Explore semantic neighborhood",
            "POST /entities/find-similar": "Find similar entities",
            "POST /namespaces/create": "Create a namespace",
            "POST /namespaces/list": "List namespaces",
            "POST /namespaces/assign": "Assign entity to namespace",
            "POST /facts/store": "Store a fact",
            "POST /facts/store-bulk": "Store multiple facts",
            "POST /facts/upsert": "Upsert a fact",
            "POST /facts/query": "Query facts",
            "POST /facts/update": "Update a fact",
            "POST /facts/delete": "Delete a fact",
            "POST /facts/user-context": "Get user context",
            "POST /facts/namespace-context": "Get namespace context",
        },
        "mcp_endpoint": "/mcp (for MCP clients)",
    }


@rest_api.get("/info")
async def api_info():
    """Get system information."""
    return await get_info()


@rest_api.get("/traits")
async def api_traits():
    """Get all 32 trait definitions."""
    return await get_traits()


@rest_api.get("/patterns")
async def api_patterns():
    """Get reasoning patterns."""
    return await get_patterns()


@rest_api.post("/classify")
async def api_classify(request: ClassifyRequest):
    """Classify an entity and get its hex code."""
    return await classify_entity(
        request.entity, request.context, use_semantic_priors=request.use_semantic_priors,
    )


@rest_api.post("/compare")
async def api_compare(request: CompareRequest):
    """Compare two entities."""
    return await compare_entities(request.entity_a, request.entity_b)


@rest_api.post("/batch-compare")
async def api_batch_compare(request: BatchCompareRequest):
    """Compare entity against multiple candidates, ranked by Jaccard."""
    return await batch_compare(request.entity, request.candidates)


@rest_api.post("/search")
async def api_search(request: SearchRequest):
    """Semantic search for similar entities."""
    return await semantic_search(request.query, request.limit)


@rest_api.post("/disambiguate")
async def api_disambiguate(request: DisambiguateRequest):
    """Disambiguate a polysemous term."""
    return await disambiguate_term(request.term, request.language)


@rest_api.post("/map-properties-to-traits")
async def api_map_properties_to_traits(request: MapPropertiesToTraitsRequest):
    """Map natural language properties to candidate UHT trait bits."""
    return await map_properties_to_traits(request.properties)


@rest_api.get("/entities")
async def api_list_entities(
    name_contains: str = Query(default="", description="Filter by name substring"),
    hex_pattern: str = Query(default="", description="Filter by hex pattern"),
    limit: int = Query(default=50, ge=1, le=100, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Skip N results"),
):
    """List entities in local graph."""
    return await list_entities(hex_pattern, name_contains, "", limit, offset, "both")


@rest_api.get("/search/traits")
async def api_search_by_traits(
    # Physical layer
    physical_object: bool | None = Query(default=None, description="Bit 1"),
    synthetic: bool | None = Query(default=None, description="Bit 2"),
    biological: bool | None = Query(default=None, description="Bit 3"),
    powered: bool | None = Query(default=None, description="Bit 4"),
    structural: bool | None = Query(default=None, description="Bit 5"),
    observable: bool | None = Query(default=None, description="Bit 6"),
    physical_medium: bool | None = Query(default=None, description="Bit 7"),
    active: bool | None = Query(default=None, description="Bit 8"),
    # Functional layer
    intentionally_designed: bool | None = Query(default=None, description="Bit 9"),
    outputs_effect: bool | None = Query(default=None, description="Bit 10"),
    processes_signals: bool | None = Query(default=None, description="Bit 11"),
    state_transforming: bool | None = Query(default=None, description="Bit 12"),
    human_interactive: bool | None = Query(default=None, description="Bit 13"),
    system_integrated: bool | None = Query(default=None, description="Bit 14"),
    functionally_autonomous: bool | None = Query(default=None, description="Bit 15"),
    system_essential: bool | None = Query(default=None, description="Bit 16"),
    # Abstract layer
    symbolic: bool | None = Query(default=None, description="Bit 17"),
    signalling: bool | None = Query(default=None, description="Bit 18"),
    rule_governed: bool | None = Query(default=None, description="Bit 19"),
    compositional: bool | None = Query(default=None, description="Bit 20"),
    normative: bool | None = Query(default=None, description="Bit 21"),
    meta: bool | None = Query(default=None, description="Bit 22"),
    temporal: bool | None = Query(default=None, description="Bit 23"),
    digital_virtual: bool | None = Query(default=None, description="Bit 24"),
    # Social layer
    social_construct: bool | None = Query(default=None, description="Bit 25"),
    institutionally_defined: bool | None = Query(default=None, description="Bit 26"),
    identity_linked: bool | None = Query(default=None, description="Bit 27"),
    regulated: bool | None = Query(default=None, description="Bit 28"),
    economically_significant: bool | None = Query(default=None, description="Bit 29"),
    politicised: bool | None = Query(default=None, description="Bit 30"),
    ritualised: bool | None = Query(default=None, description="Bit 31"),
    ethically_significant: bool | None = Query(default=None, description="Bit 32"),
    limit: int = Query(default=20, ge=1, le=100, description="Max results"),
):
    """Search Factory corpus by trait pattern."""
    return await search_by_traits(
        physical_object=physical_object,
        synthetic=synthetic,
        biological=biological,
        powered=powered,
        structural=structural,
        observable=observable,
        physical_medium=physical_medium,
        active=active,
        intentionally_designed=intentionally_designed,
        outputs_effect=outputs_effect,
        processes_signals=processes_signals,
        state_transforming=state_transforming,
        human_interactive=human_interactive,
        system_integrated=system_integrated,
        functionally_autonomous=functionally_autonomous,
        system_essential=system_essential,
        symbolic=symbolic,
        signalling=signalling,
        rule_governed=rule_governed,
        compositional=compositional,
        normative=normative,
        meta=meta,
        temporal=temporal,
        digital_virtual=digital_virtual,
        social_construct=social_construct,
        institutionally_defined=institutionally_defined,
        identity_linked=identity_linked,
        regulated=regulated,
        economically_significant=economically_significant,
        politicised=politicised,
        ritualised=ritualised,
        ethically_significant=ethically_significant,
        limit=limit,
    )


# --- Entity management endpoints ---


class DeleteEntityRequest(BaseModel):
    """Request model for delete entity endpoint."""
    name: str
    source: str = "local"


class InferPropertiesRequest(BaseModel):
    """Request model for infer properties endpoint."""
    entity: str


class ExploreNeighborhoodRequest(BaseModel):
    """Request model for explore neighborhood endpoint."""
    entity: str
    metric: str = "embedding"
    limit: int = 10
    min_similarity: float = 0.3


class FindSimilarRequest(BaseModel):
    """Request model for find similar entities endpoint."""
    entity: str
    limit: int = 5
    min_shared_traits: int = 20


@rest_api.post("/entities/delete")
async def api_delete_entity(request: DeleteEntityRequest):
    """Delete an entity from the local knowledge graph."""
    return await delete_entity(request.name, request.source)


@rest_api.post("/entities/infer")
async def api_infer_properties(request: InferPropertiesRequest):
    """Infer properties of an entity from its classification."""
    return await infer_properties(request.entity)


@rest_api.post("/entities/explore")
async def api_explore_neighborhood(request: ExploreNeighborhoodRequest):
    """Explore the semantic neighborhood of an entity."""
    return await explore_neighborhood(
        request.entity, request.metric, request.limit, request.min_similarity,
    )


@rest_api.post("/entities/find-similar")
async def api_find_similar(request: FindSimilarRequest):
    """Find entities similar to the given entity."""
    return await find_similar_entities(request.entity, request.limit, request.min_shared_traits)


# --- Semantic analysis endpoints ---


class SemanticTriangleRequest(BaseModel):
    """Request model for semantic triangle endpoint."""
    text: str


@rest_api.post("/semantic-triangle")
async def api_semantic_triangle(request: SemanticTriangleRequest):
    """Get the Ogden-Richards semantic triangle for a term."""
    return await get_semantic_triangle(request.text)


# --- Namespace endpoints ---


class CreateNamespaceRequest(BaseModel):
    """Request model for create namespace endpoint."""
    code: str
    name: str
    description: str = ""


class ListNamespacesRequest(BaseModel):
    """Request model for list namespaces endpoint."""
    parent: str = ""
    include_descendants: bool = False


class AssignNamespaceRequest(BaseModel):
    """Request model for assign to namespace endpoint."""
    entity_name: str
    namespace: str
    primary: bool = True


@rest_api.post("/namespaces/create")
async def api_create_namespace(request: CreateNamespaceRequest):
    """Create a new namespace."""
    return await create_namespace(request.code, request.name, request.description)


@rest_api.post("/namespaces/list")
async def api_list_namespaces(request: ListNamespacesRequest):
    """List namespaces."""
    return await list_namespaces(request.parent, request.include_descendants)


@rest_api.post("/namespaces/assign")
async def api_assign_namespace(request: AssignNamespaceRequest):
    """Assign an entity to a namespace."""
    return await assign_to_namespace(request.entity_name, request.namespace, request.primary)


# --- Fact endpoints ---


class StoreFactRequest(BaseModel):
    """Request model for store fact endpoint."""
    subject: str
    predicate: str
    object_value: str
    user_id: str = "default"
    namespace: str = ""


class StoreFactsBulkRequest(BaseModel):
    """Request model for store facts bulk endpoint."""
    facts: list[dict[str, str]]


class UpsertFactRequest(BaseModel):
    """Request model for upsert fact endpoint."""
    subject: str
    predicate: str
    object_value: str
    user_id: str = "default"
    namespace: str = ""


class QueryFactsRequest(BaseModel):
    """Request model for query facts endpoint."""
    subject: str = ""
    object_value: str = ""
    predicate: str = ""
    category: str = ""
    user_id: str = ""
    namespace: str = ""
    limit: int = 20


class UpdateFactRequest(BaseModel):
    """Request model for update fact endpoint."""
    fact_id: str
    subject: str = ""
    predicate: str = ""
    object_value: str = ""


class DeleteFactRequest(BaseModel):
    """Request model for delete fact endpoint."""
    fact_id: str


class UserContextRequest(BaseModel):
    """Request model for get user context endpoint."""
    user_id: str = "default"


class NamespaceContextRequest(BaseModel):
    """Request model for get namespace context endpoint."""
    namespace: str
    user_id: str = ""


@rest_api.post("/facts/store")
async def api_store_fact(request: StoreFactRequest):
    """Store a fact in the knowledge graph."""
    return await store_fact(
        request.subject, request.predicate, request.object_value,
        request.user_id, request.namespace,
    )


@rest_api.post("/facts/store-bulk")
async def api_store_facts_bulk(request: StoreFactsBulkRequest):
    """Store multiple facts in a single call."""
    return await store_facts_bulk(request.facts)


@rest_api.post("/facts/upsert")
async def api_upsert_fact(request: UpsertFactRequest):
    """Upsert a fact (create or update)."""
    return await upsert_fact(
        request.subject, request.predicate, request.object_value,
        request.user_id, request.namespace,
    )


@rest_api.post("/facts/query")
async def api_query_facts(request: QueryFactsRequest):
    """Query facts with flexible filters."""
    return await query_facts(
        request.subject, request.object_value, request.predicate,
        request.category, request.user_id, request.namespace, request.limit,
    )


@rest_api.post("/facts/update")
async def api_update_fact(request: UpdateFactRequest):
    """Update an existing fact."""
    return await update_fact(
        request.fact_id, request.subject, request.predicate, request.object_value,
    )


@rest_api.post("/facts/delete")
async def api_delete_fact(request: DeleteFactRequest):
    """Delete a fact from the knowledge graph."""
    return await delete_fact(request.fact_id)


@rest_api.post("/facts/user-context")
async def api_user_context(request: UserContextRequest):
    """Get stored facts and preferences for a user."""
    return await get_user_context(request.user_id)


@rest_api.post("/facts/namespace-context")
async def api_namespace_context(request: NamespaceContextRequest):
    """Get all entities and facts under a namespace."""
    return await get_namespace_context(request.namespace, request.user_id)


# =============================================================================
# Main Entry Point
# =============================================================================


def create_combined_app():
    """Create a combined ASGI app with both REST API and MCP server."""
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    from starlette.routing import Mount

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        """Require Bearer token on all requests when api_key is configured."""

        async def dispatch(self, request, call_next):
            if not settings.api_key:
                return await call_next(request)
            # Check Authorization header first, then ?token= query param
            auth = request.headers.get("Authorization", "")
            if auth == f"Bearer {settings.api_key}":
                return await call_next(request)
            if request.query_params.get("token") == settings.api_key:
                return await call_next(request)
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # Get the MCP ASGI app
    mcp_app = mcp.http_app(path="/mcp")

    @asynccontextmanager
    async def chained_lifespan(app):
        """Chain our lifespan with the MCP lifespan."""
        # Start our resources
        await _startup()

        # Run MCP's lifespan (initializes task group)
        async with mcp_app.lifespan(app):
            yield

        # Cleanup our resources
        await _shutdown()

    # Create combined app with REST API at /api and MCP at root
    combined = Starlette(
        routes=[
            Mount("/api", app=rest_api),
            Mount("/", app=mcp_app),
        ],
        middleware=[Middleware(BearerAuthMiddleware)],
        lifespan=chained_lifespan,
    )

    return combined


def main() -> None:
    """Run the MCP server."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="UHT Substrate Agent MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol: stdio (default) or sse (web)",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Shortcut for --transport sse (run as HTTP server)",
    )
    parser.add_argument(
        "--host",
        default=settings.server_host,
        help=f"Host to bind to (default: {settings.server_host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.server_port,
        help=f"Port to bind to (default: {settings.server_port})",
    )

    args = parser.parse_args()

    transport = "sse" if args.web else args.transport

    logger.info(
        "Starting UHT Substrate Agent",
        transport=transport,
        host=args.host,
        port=args.port,
    )

    if transport == "sse":
        # Run combined HTTP server with REST API + MCP
        # REST API: /api/*
        # MCP: /mcp
        combined_app = create_combined_app()
        uvicorn.run(
            combined_app,
            host=args.host,
            port=args.port,
            log_level="info",
        )
    else:
        # Run with stdio transport (default for MCP clients)
        mcp.run()


if __name__ == "__main__":
    main()
