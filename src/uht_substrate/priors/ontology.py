"""Loader for ontological commitments from YAML configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from ..config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OntologicalCommitment:
    """A single ontological commitment."""

    uuid: str
    name: str
    statement: str
    category: str
    implications: list[str] = field(default_factory=list)
    confidence: float = 1.0
    confidence_decay: Optional[float] = None


class OntologyRepository:
    """Repository for loading and querying ontological commitments."""

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the repository.

        Args:
            data_path: Path to priors data directory
        """
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent.parent / "data" / "priors"
        self._data_path = data_path
        self._commitments: list[OntologicalCommitment] = []
        self._by_category: dict[str, list[OntologicalCommitment]] = {}
        self._loaded = False

    def load(self) -> None:
        """Load ontological commitments from YAML file."""
        if self._loaded:
            return

        ontology_file = self._data_path / "ontology.yaml"
        if not ontology_file.exists():
            logger.warning("Ontology file not found", path=str(ontology_file))
            return

        with open(ontology_file) as f:
            data = yaml.safe_load(f)

        commitment_id = 0
        for section, commitments in data.items():
            for commitment_data in commitments:
                commitment = OntologicalCommitment(
                    uuid=f"commitment-{commitment_id}",
                    name=commitment_data["name"],
                    statement=commitment_data["statement"],
                    category=commitment_data.get("category", section),
                    implications=commitment_data.get("implications", []),
                    confidence=commitment_data.get("confidence", 1.0),
                    confidence_decay=commitment_data.get("confidence_decay"),
                )
                self._commitments.append(commitment)

                # Index by category
                category = commitment.category
                if category not in self._by_category:
                    self._by_category[category] = []
                self._by_category[category].append(commitment)

                commitment_id += 1

        self._loaded = True
        logger.info("Loaded ontological commitments", count=len(self._commitments))

    def get_all(self) -> list[OntologicalCommitment]:
        """
        Get all ontological commitments.

        Returns:
            List of all commitments
        """
        self.load()
        return self._commitments.copy()

    def get_by_category(self, category: str) -> list[OntologicalCommitment]:
        """
        Get commitments by category.

        Args:
            category: Category name

        Returns:
            List of commitments in that category
        """
        self.load()
        return self._by_category.get(category, [])

    def get_by_name(self, name: str) -> Optional[OntologicalCommitment]:
        """
        Get commitment by name.

        Args:
            name: Commitment name

        Returns:
            Commitment if found
        """
        self.load()
        for c in self._commitments:
            if c.name == name:
                return c
        return None

    def get_categories(self) -> list[str]:
        """
        Get all category names.

        Returns:
            List of category names
        """
        self.load()
        return list(self._by_category.keys())

    def get_inheritance_commitment(self) -> Optional[OntologicalCommitment]:
        """Get the trait inheritance commitment."""
        return self.get_by_name("Trait Inheritance")

    def get_similarity_commitment(self) -> Optional[OntologicalCommitment]:
        """Get the similarity implies shared properties commitment."""
        return self.get_by_name("Similarity Implies Shared Properties")

    def get_transitivity_commitment(self, rel_type: str) -> Optional[OntologicalCommitment]:
        """Get transitivity commitment for a relationship type."""
        if rel_type == "IS_A":
            return self.get_by_name("Transitivity of IS_A")
        elif rel_type == "PART_OF":
            return self.get_by_name("Partial Transitivity of PART_OF")
        return None
