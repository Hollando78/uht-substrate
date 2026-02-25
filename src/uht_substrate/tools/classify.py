"""Classification tools for MCP server."""

from typing import Any

from ..reasoning.engine import ReasoningEngine


async def classify_entity(
    engine: ReasoningEngine,
    entity: str,
    context: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Classify an entity using UHT.

    Args:
        engine: Reasoning engine
        entity: Entity name to classify
        context: Optional context to guide classification
        user_id: Optional user ID for personalization

    Returns:
        Classification result with hex code and properties
    """
    result = await engine.reason(
        query=f"What is {entity}?",
        user_id=user_id,
        additional_context=context,
    )

    return {
        "entity": entity,
        "hex_code": result.hex_codes.get(entity, "unknown"),
        "answer": result.answer,
        "confidence": result.confidence,
        "properties": result.inferred_properties,
        "trace_id": result.trace_id,
    }


async def get_entity(
    engine: ReasoningEngine,
    entity: str,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Get entity from local graph or UHT.

    Args:
        engine: Reasoning engine
        entity: Entity name
        user_id: Optional user ID

    Returns:
        Entity information
    """
    result = await engine.reason(
        query=f"Tell me about {entity}",
        user_id=user_id,
    )

    return {
        "entity": entity,
        "hex_code": result.hex_codes.get(entity),
        "answer": result.answer,
        "confidence": result.confidence,
        "sources": result.sources,
    }


async def find_similar(
    engine: ReasoningEngine,
    entity: str,
    limit: int = 10,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Find entities similar to the given entity.

    Args:
        engine: Reasoning engine
        entity: Entity name
        limit: Maximum results
        user_id: Optional user ID

    Returns:
        Similar entities
    """
    result = await engine.reason(
        query=f"What is similar to {entity}?",
        user_id=user_id,
    )

    return {
        "entity": entity,
        "answer": result.answer,
        "entities_found": result.entities_used,
        "confidence": result.confidence,
    }
