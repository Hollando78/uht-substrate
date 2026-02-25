"""Exploration tools for MCP server."""

from typing import Any

from ..reasoning.engine import ReasoningEngine
from ..uht_client.client import UHTClient


async def explore_neighborhood(
    engine: ReasoningEngine,
    entity: str,
    depth: int = 2,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Explore the semantic neighborhood of an entity.

    Args:
        engine: Reasoning engine
        entity: Entity to explore from
        depth: How many hops to explore
        user_id: Optional user ID

    Returns:
        Neighborhood information
    """
    result = await engine.reason(
        query=f"What is related to {entity}?",
        user_id=user_id,
    )

    return {
        "entity": entity,
        "hex_code": result.hex_codes.get(entity),
        "answer": result.answer,
        "entities_found": result.entities_used,
        "confidence": result.confidence,
    }


async def disambiguate(
    uht: UHTClient,
    term: str,
    context: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """
    Disambiguate a polysemous term.

    Args:
        uht: UHT Factory client
        term: Term to disambiguate
        context: Optional context to help disambiguation
        lang: Language code

    Returns:
        Disambiguation result with senses
    """
    result = await uht.disambiguate(term, lang)

    return {
        "term": term,
        "language": result.language,
        "senses": [
            {
                "sense_id": s.sense_id,
                "definition": s.definition,
                "hex_code": s.hex_code,
                "entity_uuid": s.entity_uuid,
                "examples": s.examples,
            }
            for s in result.senses
        ],
    }


async def semantic_search(
    uht: UHTClient,
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Search for entities by semantic similarity.

    Args:
        uht: UHT Factory client
        query: Search query
        limit: Maximum results

    Returns:
        Matching entities
    """
    entities = await uht.semantic_search(query, limit)

    return {
        "query": query,
        "results": [
            {
                "uuid": e.uuid,
                "name": e.name,
                "hex_code": e.hex_code,
                "description": e.description,
            }
            for e in entities
        ],
    }


async def analyze_hex(
    uht: UHTClient,
    hex_code: str,
) -> dict[str, Any]:
    """
    Analyze a hex code for its trait composition.

    Args:
        uht: UHT Factory client
        hex_code: 8-character hex code

    Returns:
        Analysis of the hex code
    """
    return await uht.analyze_hex(hex_code)


async def get_semantic_triangle(
    uht: UHTClient,
    text: str,
) -> dict[str, Any]:
    """
    Get the semantic triangle decomposition for text.

    Args:
        uht: UHT Factory client
        text: Text to analyze

    Returns:
        Semantic triangle (symbol, referent, reference)
    """
    triangle = await uht.get_semantic_triangle(text)

    return {
        "text": text,
        "symbol": triangle.symbol,
        "referent": triangle.referent,
        "reference": triangle.reference,
    }
