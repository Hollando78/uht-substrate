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
from .priors.inference import PriorInferenceEngine
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


@mcp.tool()
async def classify_entity(
    entity: str,
    context: str = "",
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

    If the entity exists in the Factory corpus (16,000+ entities), returns cached
    classification. Otherwise runs fresh classification (slower, uses GPT-4).

    Args:
        entity: The entity to classify (e.g., "hammer", "democracy", "virus")
        context: Optional disambiguation context (e.g., "the programming language")

    Returns:
        hex_code: 8-char hex (e.g., "C6880008")
        properties: Inferred properties from trait axioms
        confidence: Classification confidence
    """
    if not ctx.engine:
        return {"error": "Engine not initialized"}

    result = await ctx.engine.reason(
        query=f"What is {entity}?",
        additional_context=context if context else None,
    )

    return {
        "entity": entity,
        "hex_code": result.hex_codes.get(entity, "unknown"),
        "answer": result.answer,
        "confidence": result.confidence,
        "properties": result.inferred_properties[:10],
        "trace_id": result.trace_id,
    }


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
        # First try to find existing entity in Factory corpus
        existing = await ctx.uht.search_entities(query=entity, limit=5)

        # Look for exact or close name match
        entity_data = None
        entity_lower = entity.lower()
        for e in existing:
            if e.name.lower() == entity_lower:
                entity_data = e
                break

        # If no exact match, check local graph
        if not entity_data:
            local_entity = await ctx.graph.find_entity_by_name(entity)
            if local_entity:
                # Use local entity's UUID to get from Factory
                try:
                    entity_data = await ctx.uht.get_entity(local_entity.uuid)
                except Exception:
                    pass

        # If still no match, classify fresh (last resort)
        if not entity_data:
            entity_data = await ctx.uht.classify(entity)

        # Find similar via UHT API
        similar_results = await ctx.uht.find_similar(
            uuid=entity_data.uuid,
            threshold=min_shared_traits,
            limit=limit,
        )

        return {
            "entity": entity,
            "hex_code": entity_data.hex_code,
            "similar_entities": [
                {
                    "name": r.entity.name,
                    "hex_code": r.entity.hex_code,
                    "similarity_score": r.similarity_score,
                    "hamming_distance": r.hamming_distance,
                    "shared_traits": len(r.shared_traits),
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
    limit: int = 50,
    source: str = "both",
) -> dict[str, Any]:
    """
    List classified entities from the local graph and/or UHT Factory corpus.

    Args:
        hex_pattern: Filter by hex code pattern (e.g., "C688" for tools)
        name_contains: Filter by name substring (case-insensitive)
        limit: Maximum number of results (1-100)
        source: "local" (Neo4j only), "factory" (UHT API only), or "both"

    Returns:
        List of entities with names, hex codes, and source
    """
    limit = min(max(limit, 1), 100)
    results = []

    # Query local Neo4j graph
    if source in ("local", "both") and ctx.graph:
        local_entities = await ctx.graph.list_entities(
            name_contains=name_contains if name_contains else None,
            hex_pattern=hex_pattern if hex_pattern else None,
            limit=limit,
        )
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
            )
            for e in factory_entities:
                # Avoid duplicates from local
                if not any(r["name"] == e.name for r in results):
                    results.append({
                        "name": e.name,
                        "hex_code": e.hex_code,
                        "source": "factory",
                        "created_at": str(e.created_at) if e.created_at else None,
                    })
        except Exception as e:
            # Factory query failed, continue with local results
            pass

    return {
        "count": len(results),
        "entities": results[:limit],
        "filters": {
            "hex_pattern": hex_pattern or None,
            "name_contains": name_contains or None,
            "source": source,
        },
    }


@mcp.tool()
async def search_by_traits(
    physical_object: bool | None = None,
    synthetic: bool | None = None,
    biological: bool | None = None,
    structural: bool | None = None,
    observable: bool | None = None,
    consumable: bool | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Search for entities matching specific trait constraints.

    Use this to find entities with particular combinations of traits.
    Pass True to require a trait, False to exclude it, or omit to ignore.

    Example: Find non-living, non-edible, human-made objects:
      synthetic=True, biological=False, consumable=False

    Args:
        physical_object: Bit 1 - Has physical form/mass
        synthetic: Bit 2 - Human-made/manufactured
        biological: Bit 3 - Has biological origin
        structural: Bit 5 - Load-bearing/structural
        observable: Bit 6 - Can be directly observed
        consumable: Bit 12 - Can be consumed/eaten
        limit: Maximum results (1-100)

    Returns:
        List of matching entities with hex codes
    """
    if not ctx.uht:
        return {"error": "UHT client not initialized"}

    # Build 32-char pattern with X for wildcards
    # Bit positions: 1=physical, 2=synthetic, 3=biological, 5=structural, 6=observable, 12=consumable
    pattern = ["X"] * 32

    trait_map = {
        1: physical_object,
        2: synthetic,
        3: biological,
        5: structural,
        6: observable,
        12: consumable,
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
# Reasoning Tools
# =============================================================================


@mcp.tool()
async def reason_about(
    query: str,
    context: str = "",
) -> dict[str, Any]:
    """
    [DEPRECATED] General-purpose reasoning about entities.

    This tool is deprecated. Instead, use the primitive tools directly:
    - "What is X?" → classify_entity(X)
    - "Compare X and Y" → compare_entities(X, Y)
    - "What properties does X have?" → infer_properties(X)

    See the server instructions for reasoning patterns that compose these tools.

    Args:
        query: Question to reason about
        context: Optional additional context

    Returns:
        Reasoned answer with confidence and supporting evidence
    """
    if not ctx.engine:
        return {"error": "Engine not initialized"}

    result = await ctx.engine.reason(
        query=query,
        additional_context=context if context else None,
    )

    return {
        "query": query,
        "answer": result.answer,
        "confidence": result.confidence,
        "sources": result.sources,
        "inferred_properties": result.inferred_properties[:10],
        "trace_id": result.trace_id,
    }


@mcp.tool()
async def compare_entities(
    entity_a: str,
    entity_b: str,
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
            # Include actual trait names for detailed analysis
            trait_diff = {
                "shared_traits": analysis.get_shared_trait_names(),
                "traits_a_only": analysis.get_traits_a_only_names(),
                "traits_b_only": analysis.get_traits_b_only_names(),
            }

    return {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "hex_codes": result.hex_codes,
        "similarity": similarity_metrics,
        "trait_diff": trait_diff,
        "comparison": result.answer,
        "confidence": result.confidence,
        "trace_id": result.trace_id,
    }


@mcp.tool()
async def batch_compare(
    entity: str,
    candidates: list[str],
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

    Returns:
        Ranked list of comparisons sorted by Jaccard similarity
    """
    if not ctx.uht or not ctx.inference:
        return {"error": "Engine not initialized"}

    candidates = candidates[:20]  # Limit to 20 candidates

    # First classify the main entity
    try:
        main_class = await ctx.uht.classify(entity)
    except Exception as e:
        return {"error": f"Failed to classify {entity}: {e}"}

    # Classify all candidates in parallel
    import asyncio
    async def classify_candidate(name: str):
        try:
            return (name, await ctx.uht.classify(name))
        except Exception:
            return (name, None)

    candidate_results = await asyncio.gather(
        *[classify_candidate(c) for c in candidates]
    )

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

        comparisons.append({
            "candidate": name,
            "hex_code": classification.hex_code,
            "jaccard_similarity": round(analysis.jaccard_similarity, 3),
            "hamming_distance": analysis.hamming_distance,
            "shared_traits": analysis.get_shared_trait_names(),
            "traits_entity_only": analysis.get_traits_a_only_names(),
            "traits_candidate_only": analysis.get_traits_b_only_names(),
        })

    # Sort by Jaccard (highest first)
    comparisons.sort(key=lambda x: x["jaccard_similarity"], reverse=True)

    return {
        "entity": entity,
        "hex_code": main_class.hex_code,
        "comparisons": comparisons,
        "best_match": comparisons[0]["candidate"] if comparisons else None,
        "best_jaccard": comparisons[0]["jaccard_similarity"] if comparisons else None,
    }


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
        # First try to find existing entity in Factory corpus
        existing = await ctx.uht.search_entities(query=entity, limit=5)

        # Look for exact or close name match
        entity_data = None
        entity_lower = entity.lower()
        for e in existing:
            if e.name.lower() == entity_lower:
                entity_data = e
                break

        # If no exact match, check local graph
        if not entity_data:
            local_entity = await ctx.graph.find_entity_by_name(entity)
            if local_entity:
                try:
                    entity_data = await ctx.uht.get_entity(local_entity.uuid)
                except Exception:
                    pass

        # If still no match, classify fresh (last resort)
        if not entity_data:
            entity_data = await ctx.uht.classify(entity)

        # Get neighborhood from UHT
        neighborhood = await ctx.uht.get_neighborhood(
            uuid=entity_data.uuid,
            metric=metric,
            k=limit,
            min_similarity=min_similarity,
        )

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


# =============================================================================
# Context Management Tools
# =============================================================================


@mcp.tool()
async def store_fact(
    subject: str,
    predicate: str,
    object_value: str,
    user_id: str = "default",
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

    Returns:
        Confirmation of stored fact
    """
    if not ctx.graph:
        return {"error": "Graph not initialized"}

    fact = await ctx.graph.store_fact(
        subject=subject,
        predicate=predicate,
        obj=object_value,
        confidence=1.0,
        source="user",
        user_id=user_id,
    )

    return {
        "fact_id": fact.uuid,
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
        "stored": True,
    }


@mcp.tool()
async def get_user_context(
    user_id: str = "default",
) -> dict[str, Any]:
    """
    Retrieve stored facts and preferences for a user.

    Args:
        user_id: User identifier

    Returns:
        User's stored facts and preferences
    """
    if not ctx.graph:
        return {"error": "Graph not initialized"}

    facts = await ctx.graph.get_user_facts(user_id, limit=20)
    preferences = await ctx.graph.get_user_preferences(user_id)

    return {
        "user_id": user_id,
        "facts": [
            {
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
            }
            for f in facts
        ],
        "preferences": preferences,
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
                {"tool": "compare_entities", "use": "Compare two entities — Jaccard, shared/unique traits"},
                {"tool": "infer_properties", "use": "Derive properties from classification via trait axioms"},
                {"tool": "semantic_search", "use": "Find similar entities in 16k+ Factory corpus (USE THIS for analogies)"},
                {"tool": "disambiguate_term", "use": "Get word senses for polysemous terms"},
                {"tool": "list_entities", "use": "Browse local knowledge graph"},
                {"tool": "get_patterns", "use": "Get reasoning patterns for complex questions"},
                {"tool": "get_traits", "use": "Get definitions of all 32 traits"},
                {"tool": "get_info", "use": "Get this overview"},
            ],
            "experimental": [
                {"tool": "find_similar_entities", "status": "Returns empty — use semantic_search instead"},
                {"tool": "explore_neighborhood", "status": "Returns empty — use semantic_search instead"},
                {"tool": "reason_about", "status": "Deprecated — use patterns with core tools instead"},
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
            "GET /entities": "List entities in local graph",
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
    return await classify_entity(request.entity, request.context)


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


@rest_api.get("/entities")
async def api_list_entities(
    name_contains: str = Query(default="", description="Filter by name substring"),
    hex_pattern: str = Query(default="", description="Filter by hex pattern"),
    limit: int = Query(default=50, ge=1, le=100, description="Max results"),
):
    """List entities in local graph."""
    return await list_entities(hex_pattern, name_contains, limit, "both")


# =============================================================================
# Main Entry Point
# =============================================================================


def create_combined_app():
    """Create a combined ASGI app with both REST API and MCP server."""
    from starlette.applications import Starlette
    from starlette.routing import Mount

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
