"""Loader for reasoning heuristics from YAML configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from ..config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Heuristic:
    """A reasoning heuristic."""

    uuid: str
    name: str
    description: str
    priority: int
    applicability: str
    implementation: str
    parameters: dict[str, Any] = field(default_factory=dict)
    rationale: Optional[str] = None


class HeuristicRepository:
    """Repository for loading and querying reasoning heuristics."""

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the repository.

        Args:
            data_path: Path to priors data directory
        """
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent.parent / "data" / "priors"
        self._data_path = data_path
        self._heuristics: dict[str, Heuristic] = {}
        self._by_priority: list[Heuristic] = []
        self._loaded = False

    def load(self) -> None:
        """Load heuristics from YAML file."""
        if self._loaded:
            return

        heuristics_file = self._data_path / "heuristics.yaml"
        if not heuristics_file.exists():
            logger.warning("Heuristics file not found", path=str(heuristics_file))
            return

        with open(heuristics_file) as f:
            data = yaml.safe_load(f)

        for i, h_data in enumerate(data.get("heuristics", [])):
            heuristic = Heuristic(
                uuid=f"heuristic-{i}",
                name=h_data["name"],
                description=h_data["description"],
                priority=h_data["priority"],
                applicability=h_data["applicability"],
                implementation=h_data["implementation"],
                parameters=h_data.get("parameters", {}),
                rationale=h_data.get("rationale"),
            )
            self._heuristics[heuristic.name] = heuristic

        # Sort by priority (descending)
        self._by_priority = sorted(
            self._heuristics.values(),
            key=lambda h: h.priority,
            reverse=True,
        )

        self._loaded = True
        logger.info("Loaded heuristics", count=len(self._heuristics))

    def get(self, name: str) -> Optional[Heuristic]:
        """
        Get heuristic by name.

        Args:
            name: Heuristic name

        Returns:
            Heuristic if found
        """
        self.load()
        return self._heuristics.get(name)

    def get_all(self) -> list[Heuristic]:
        """
        Get all heuristics sorted by priority.

        Returns:
            List of heuristics, highest priority first
        """
        self.load()
        return self._by_priority.copy()

    def get_by_applicability(self, context: str) -> list[Heuristic]:
        """
        Get heuristics that match a given context.

        Args:
            context: Context description to match

        Returns:
            List of applicable heuristics
        """
        self.load()
        context_lower = context.lower()
        return [
            h
            for h in self._by_priority
            if context_lower in h.applicability.lower()
        ]

    def get_confidence_threshold(self) -> float:
        """Get the confidence threshold for assertions."""
        h = self.get("Confidence Threshold")
        if h and "assertion_threshold" in h.parameters:
            return h.parameters["assertion_threshold"]
        return 0.7

    def get_possibility_threshold(self) -> float:
        """Get the confidence threshold for possibilities."""
        h = self.get("Confidence Threshold")
        if h and "possibility_threshold" in h.parameters:
            return h.parameters["possibility_threshold"]
        return 0.5

    def get_freshness_hours(self) -> int:
        """Get how many hours before classification is considered stale."""
        h = self.get("Freshness Check")
        if h and "fresh_hours" in h.parameters:
            return h.parameters["fresh_hours"]
        return 24

    def get_max_hamming_distance(self) -> int:
        """Get maximum Hamming distance for nearest neighbor inference."""
        h = self.get("Nearest Neighbor Property Transfer")
        if h and "max_hamming_distance" in h.parameters:
            return h.parameters["max_hamming_distance"]
        return 4

    def get_confidence_decay_factor(self) -> float:
        """Get the confidence decay factor for inference chains."""
        h = self.get("Confidence Decay")
        if h and "decay_factor" in h.parameters:
            return h.parameters["decay_factor"]
        return 0.9
