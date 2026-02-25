"""MCP server entry point for UHT Substrate Agent."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastmcp import FastMCP

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


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Manage application lifecycle."""
    logger.info("Starting UHT Substrate Agent")

    # Initialize connections
    ctx.neo4j = Neo4jConnection(settings)
    await ctx.neo4j.connect()
    await ctx.neo4j.initialize_schema()

    ctx.graph = GraphRepository(ctx.neo4j)
    ctx.uht = UHTClient(settings)
    ctx.inference = PriorInferenceEngine()
    ctx.engine = ReasoningEngine(ctx.graph, ctx.uht, ctx.inference)

    logger.info("UHT Substrate Agent ready")

    yield

    # Cleanup
    logger.info("Shutting down UHT Substrate Agent")
    if ctx.uht:
        await ctx.uht.close()
    if ctx.neo4j:
        await ctx.neo4j.close()


# Initialize MCP server
mcp = FastMCP(
    name="UHT Substrate Agent",
    instructions="""
    I am the UHT Substrate Agent, a reasoning assistant grounded in the Universal Hex Taxonomy (UHT).

    I can help you:
    - Classify any entity using UHT's 32-bit classification system
    - Explore semantic relationships between concepts
    - Infer properties based on trait axioms
    - Remember facts and preferences across conversations
    - Disambiguate polysemous terms

    The UHT system classifies entities across four layers:
    - Physical (bits 1-8): Material properties
    - Functional (bits 9-16): Capabilities and behaviors
    - Abstract (bits 17-24): Symbolic and conceptual aspects
    - Social (bits 25-32): Cultural and social dimensions

    Each entity gets an 8-character hex code that captures its essential properties.
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
    Classify an entity using the Universal Hex Taxonomy.

    The entity will be evaluated against 32 binary traits organized into
    Physical, Functional, Abstract, and Social layers, resulting in an
    8-character hex code that captures its essential properties.

    Args:
        entity: The entity to classify (e.g., "bicycle", "democracy", "Python")
        context: Optional context to guide classification

    Returns:
        Classification result with hex code, properties, and explanation
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
) -> dict[str, Any]:
    """
    Find entities semantically similar to the given entity.

    Similarity is based on shared traits (low Hamming distance between hex codes).

    Args:
        entity: Entity to find similar items for
        limit: Maximum number of results (1-20)

    Returns:
        List of similar entities with similarity scores
    """
    if not ctx.engine or not ctx.uht:
        return {"error": "Engine not initialized"}

    limit = min(max(limit, 1), 20)

    result = await ctx.engine.reason(query=f"What is related to {entity}?")

    return {
        "entity": entity,
        "hex_code": result.hex_codes.get(entity),
        "similar_entities": result.entities_used,
        "answer": result.answer,
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
    Apply reasoning to answer questions about entities and concepts.

    Uses trait axioms, ontological commitments, and heuristics to derive
    conclusions from classifications.

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
    Compare two entities and explain their relationship.

    Analyzes similarity based on shared traits, checks for inheritance
    relationships, and explains key differences.

    Args:
        entity_a: First entity to compare
        entity_b: Second entity to compare

    Returns:
        Comparison analysis with similarity metrics and explanation
    """
    if not ctx.engine:
        return {"error": "Engine not initialized"}

    result = await ctx.engine.reason(
        query=f"Compare {entity_a} and {entity_b}",
    )

    return {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "hex_codes": result.hex_codes,
        "comparison": result.answer,
        "confidence": result.confidence,
        "trace_id": result.trace_id,
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
) -> dict[str, Any]:
    """
    Explore the semantic neighborhood of an entity.

    Finds related entities in both the local knowledge graph and
    the UHT Factory corpus.

    Args:
        entity: Entity to explore from

    Returns:
        Neighborhood graph with related entities
    """
    if not ctx.engine:
        return {"error": "Engine not initialized"}

    result = await ctx.engine.reason(
        query=f"What is connected to {entity}?",
    )

    return {
        "entity": entity,
        "hex_code": result.hex_codes.get(entity),
        "related_entities": result.entities_used,
        "answer": result.answer,
    }


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
        "senses": [
            {
                "sense_id": s.sense_id,
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
    - Symbol: The linguistic form (word/phrase)
    - Referent: The real-world thing it refers to
    - Reference: The mental concept/meaning

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
        "symbol": triangle.symbol,
        "referent": triangle.referent,
        "reference": triangle.reference,
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


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run the MCP server."""
    import sys

    logger.info(
        "Starting UHT Substrate Agent MCP server",
        host=settings.server_host,
        port=settings.server_port,
    )

    # Run with stdio transport (default for MCP)
    mcp.run()


if __name__ == "__main__":
    main()
