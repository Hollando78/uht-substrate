"""User context management tools for MCP server."""

from typing import Any

from ..graph.repository import GraphRepository


async def store_fact(
    graph: GraphRepository,
    subject: str,
    predicate: str,
    obj: str,
    user_id: str,
) -> dict[str, Any]:
    """
    Store a user-provided fact in the knowledge graph.

    Args:
        graph: Graph repository
        subject: Subject of the fact
        predicate: Predicate (relationship)
        obj: Object of the fact
        user_id: User ID

    Returns:
        Stored fact confirmation
    """
    fact = await graph.store_fact(
        subject=subject,
        predicate=predicate,
        obj=obj,
        confidence=1.0,
        source="user",
        user_id=user_id,
    )

    return {
        "fact_id": fact.uuid,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "stored": True,
    }


async def store_preference(
    graph: GraphRepository,
    key: str,
    value: str,
    user_id: str,
) -> dict[str, Any]:
    """
    Store a user preference.

    Args:
        graph: Graph repository
        key: Preference key
        value: Preference value
        user_id: User ID

    Returns:
        Confirmation
    """
    await graph.store_preference(user_id, key, value)

    return {
        "key": key,
        "value": value,
        "stored": True,
    }


async def get_context(
    graph: GraphRepository,
    user_id: str,
) -> dict[str, Any]:
    """
    Get all stored context for a user.

    Args:
        graph: Graph repository
        user_id: User ID

    Returns:
        User's facts and preferences
    """
    facts = await graph.get_user_facts(user_id, limit=50)
    preferences = await graph.get_user_preferences(user_id)

    return {
        "user_id": user_id,
        "facts": [
            {
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "confidence": f.confidence,
            }
            for f in facts
        ],
        "preferences": preferences,
    }


async def clear_context(
    graph: GraphRepository,
    user_id: str,
) -> dict[str, Any]:
    """
    Clear all stored context for a user.

    Note: This is a placeholder - actual implementation would need
    additional repository methods.

    Args:
        graph: Graph repository
        user_id: User ID

    Returns:
        Confirmation
    """
    # This would need a delete method in the repository
    return {
        "user_id": user_id,
        "cleared": False,
        "message": "Context clearing not yet implemented",
    }
