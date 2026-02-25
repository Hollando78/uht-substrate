"""Reasoning and inference tools for MCP server."""

from typing import Any

from ..reasoning.engine import ReasoningEngine


async def reason_about(
    engine: ReasoningEngine,
    query: str,
    context: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Apply reasoning to answer a question about entities.

    Args:
        engine: Reasoning engine
        query: The question to answer
        context: Optional context
        user_id: Optional user ID

    Returns:
        Reasoning result with answer and trace
    """
    result = await engine.reason(
        query=query,
        user_id=user_id,
        additional_context=context,
    )

    return {
        "query": query,
        "answer": result.answer,
        "confidence": result.confidence,
        "sources": result.sources,
        "inferred_properties": result.inferred_properties,
        "trace_id": result.trace_id,
    }


async def explain_relationship(
    engine: ReasoningEngine,
    entity_a: str,
    entity_b: str,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Explain how two entities relate to each other.

    Args:
        engine: Reasoning engine
        entity_a: First entity
        entity_b: Second entity
        user_id: Optional user ID

    Returns:
        Relationship explanation
    """
    result = await engine.reason(
        query=f"Compare {entity_a} and {entity_b}",
        user_id=user_id,
    )

    return {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "hex_codes": result.hex_codes,
        "explanation": result.answer,
        "confidence": result.confidence,
        "trace_id": result.trace_id,
    }


async def apply_axiom(
    engine: ReasoningEngine,
    entity: str,
    axiom_type: str = "all",
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Apply trait axioms to derive properties for an entity.

    Args:
        engine: Reasoning engine
        entity: Entity name
        axiom_type: Type of axioms (necessary, typical, all)
        user_id: Optional user ID

    Returns:
        Derived properties
    """
    result = await engine.reason(
        query=f"What can we infer about {entity}?",
        user_id=user_id,
    )

    # Filter by axiom type if specified
    properties = result.inferred_properties
    if axiom_type == "necessary":
        properties = [p for p in properties if p.get("confidence", 0) >= 0.99]
    elif axiom_type == "typical":
        properties = [p for p in properties if 0.7 <= p.get("confidence", 0) < 0.99]

    return {
        "entity": entity,
        "axiom_type": axiom_type,
        "hex_code": result.hex_codes.get(entity),
        "properties": properties,
        "confidence": result.confidence,
        "trace_id": result.trace_id,
    }


async def trace_reasoning(
    engine: ReasoningEngine,
    trace_id: str,
) -> dict[str, Any]:
    """
    Get the reasoning trace for a conclusion.

    Args:
        engine: Reasoning engine
        trace_id: Trace UUID

    Returns:
        Full reasoning trace
    """
    trace_details = await engine._graph.get_trace_details(trace_id)

    if not trace_details:
        return {"error": f"Trace {trace_id} not found"}

    return {
        "trace_id": trace_id,
        "query": trace_details.get("rt", {}).get("query"),
        "conclusion": trace_details.get("rt", {}).get("conclusion"),
        "strategy": trace_details.get("rt", {}).get("strategy"),
        "confidence": trace_details.get("rt", {}).get("confidence"),
        "entities_used": [e.get("name") for e in trace_details.get("entities", [])],
        "axioms_applied": [a.get("name") for a in trace_details.get("axioms", [])],
        "facts_derived": len(trace_details.get("facts", [])),
    }
