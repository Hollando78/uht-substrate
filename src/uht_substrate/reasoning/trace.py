"""Reasoning trace management for explainability."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from uuid6 import uuid7

from ..config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TraceStep:
    """A single step in a reasoning trace."""

    step_number: int
    action: str
    description: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    entity_uuid: Optional[str] = None
    axiom_uuid: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a query."""

    uuid: str
    query: str
    strategy: str
    steps: list[TraceStep] = field(default_factory=list)
    conclusion: Optional[str] = None
    confidence: float = 0.0
    entity_uuids: list[str] = field(default_factory=list)
    axiom_uuids: list[str] = field(default_factory=list)
    fact_uuids: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def add_step(
        self,
        action: str,
        description: str,
        entity_uuid: Optional[str] = None,
        axiom_uuid: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """Add a step to the trace."""
        step = TraceStep(
            step_number=len(self.steps) + 1,
            action=action,
            description=description,
            entity_uuid=entity_uuid,
            axiom_uuid=axiom_uuid,
            confidence=confidence,
        )
        self.steps.append(step)

        # Track UUIDs
        if entity_uuid and entity_uuid not in self.entity_uuids:
            self.entity_uuids.append(entity_uuid)
        if axiom_uuid and axiom_uuid not in self.axiom_uuids:
            self.axiom_uuids.append(axiom_uuid)

    def complete(self, conclusion: str, confidence: float) -> None:
        """Mark the trace as complete."""
        self.conclusion = conclusion
        self.confidence = confidence
        self.completed_at = datetime.utcnow()

    def add_fact(self, fact_uuid: str) -> None:
        """Record a derived fact."""
        if fact_uuid not in self.fact_uuids:
            self.fact_uuids.append(fact_uuid)

    def format_summary(self) -> str:
        """Format a human-readable summary of the trace."""
        lines = [
            f"Query: {self.query}",
            f"Strategy: {self.strategy}",
            f"Steps: {len(self.steps)}",
            "",
            "Reasoning:",
        ]

        for step in self.steps:
            prefix = f"  {step.step_number}. [{step.action}]"
            lines.append(f"{prefix} {step.description}")
            if step.confidence is not None:
                lines.append(f"      (confidence: {step.confidence:.0%})")

        if self.conclusion:
            lines.extend([
                "",
                f"Conclusion: {self.conclusion}",
                f"Confidence: {self.confidence:.0%}",
            ])

        return "\n".join(lines)


class ReasoningTraceBuilder:
    """Builder for constructing reasoning traces."""

    def __init__(self, query: str, strategy: str = "unknown"):
        """
        Initialize a new trace builder.

        Args:
            query: The original query
            strategy: The reasoning strategy being used
        """
        self._trace = ReasoningTrace(
            uuid=str(uuid7()),
            query=query,
            strategy=strategy,
        )

    @property
    def id(self) -> str:
        """Get the trace UUID."""
        return self._trace.uuid

    @property
    def trace(self) -> ReasoningTrace:
        """Get the underlying trace."""
        return self._trace

    def set_strategy(self, strategy: str) -> None:
        """Update the strategy name."""
        self._trace.strategy = strategy

    def log(
        self,
        description: str,
        action: str = "info",
        entity_uuid: Optional[str] = None,
        axiom_uuid: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """
        Log a step in the reasoning process.

        Args:
            description: What happened in this step
            action: Type of action (info, lookup, classify, infer, etc.)
            entity_uuid: Entity involved in this step
            axiom_uuid: Axiom applied in this step
            confidence: Confidence of this step
        """
        self._trace.add_step(
            action=action,
            description=description,
            entity_uuid=entity_uuid,
            axiom_uuid=axiom_uuid,
            confidence=confidence,
        )
        logger.debug(
            "Reasoning step",
            trace_id=self._trace.uuid,
            action=action,
            description=description,
        )

    def log_entity_lookup(self, entity_name: str, found: bool, uuid: Optional[str] = None) -> None:
        """Log an entity lookup."""
        if found:
            self.log(
                f"Found entity '{entity_name}' in local graph",
                action="lookup",
                entity_uuid=uuid,
            )
        else:
            self.log(
                f"Entity '{entity_name}' not found in local graph",
                action="lookup",
            )

    def log_classification(
        self,
        entity_name: str,
        hex_code: str,
        uuid: str,
        source: str = "uht_factory",
    ) -> None:
        """Log a classification result."""
        self.log(
            f"Classified '{entity_name}' as {hex_code} (source: {source})",
            action="classify",
            entity_uuid=uuid,
        )

    def log_axiom_application(
        self,
        axiom_name: str,
        axiom_uuid: str,
        property_name: str,
        confidence: float,
    ) -> None:
        """Log an axiom being applied."""
        self.log(
            f"Applied axiom '{axiom_name}' to infer property '{property_name}'",
            action="infer",
            axiom_uuid=axiom_uuid,
            confidence=confidence,
        )

    def log_similarity(
        self,
        entity_a: str,
        entity_b: str,
        similarity: float,
        hamming_distance: int,
    ) -> None:
        """Log a similarity comparison."""
        self.log(
            f"Compared '{entity_a}' with '{entity_b}': "
            f"similarity={similarity:.0%}, hamming_distance={hamming_distance}",
            action="compare",
            confidence=similarity,
        )

    def log_fact_stored(self, fact_uuid: str, subject: str, predicate: str, obj: str) -> None:
        """Log a fact being stored."""
        self._trace.add_fact(fact_uuid)
        self.log(
            f"Stored fact: {subject} {predicate} {obj}",
            action="store",
        )

    def complete(self, conclusion: str, confidence: float) -> ReasoningTrace:
        """
        Complete the trace and return it.

        Args:
            conclusion: The final conclusion
            confidence: Confidence in the conclusion

        Returns:
            The completed trace
        """
        self._trace.complete(conclusion, confidence)
        logger.info(
            "Reasoning complete",
            trace_id=self._trace.uuid,
            steps=len(self._trace.steps),
            confidence=confidence,
        )
        return self._trace
