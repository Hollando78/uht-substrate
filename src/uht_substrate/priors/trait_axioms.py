"""Loader for trait axioms from YAML configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from ..config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Axiom:
    """A single axiom derived from a trait."""

    uuid: str
    trait_bit: int
    trait_name: str
    name: str
    statement: str
    axiom_type: str  # necessary, typical, possible
    property: str
    confidence: float = 1.0


@dataclass
class TraitAxioms:
    """Axioms for a single trait."""

    bit_position: int
    name: str
    layer: str
    description: str
    axioms: list[Axiom]


class TraitAxiomRepository:
    """Repository for loading and querying trait axioms."""

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the repository.

        Args:
            data_path: Path to priors data directory
        """
        if data_path is None:
            # Default to package data directory
            data_path = Path(__file__).parent.parent.parent.parent / "data" / "priors"
        self._data_path = data_path
        self._traits: dict[int, TraitAxioms] = {}
        self._loaded = False

    def load(self) -> None:
        """Load trait axioms from YAML file."""
        if self._loaded:
            return

        axiom_file = self._data_path / "trait_axioms.yaml"
        if not axiom_file.exists():
            logger.warning("Trait axioms file not found", path=str(axiom_file))
            return

        with open(axiom_file) as f:
            data = yaml.safe_load(f)

        for key, trait_data in data.items():
            if not key.startswith("trait_"):
                continue

            bit_position = trait_data["bit_position"]
            trait_name = trait_data["name"]
            layer = trait_data["layer"]
            description = trait_data["description"]

            axioms = []
            for i, axiom_data in enumerate(trait_data.get("axioms", [])):
                axiom = Axiom(
                    uuid=f"axiom-{bit_position}-{i}",
                    trait_bit=bit_position,
                    trait_name=trait_name,
                    name=axiom_data["name"],
                    statement=axiom_data["statement"],
                    axiom_type=axiom_data["type"],
                    property=axiom_data["property"],
                    confidence=axiom_data.get("confidence", 1.0),
                )
                axioms.append(axiom)

            self._traits[bit_position] = TraitAxioms(
                bit_position=bit_position,
                name=trait_name,
                layer=layer,
                description=description,
                axioms=axioms,
            )

        self._loaded = True
        logger.info("Loaded trait axioms", count=len(self._traits))

    def get_trait(self, bit_position: int) -> Optional[TraitAxioms]:
        """
        Get trait axioms by bit position.

        Args:
            bit_position: Trait bit position (1-32)

        Returns:
            TraitAxioms if found
        """
        self.load()
        return self._traits.get(bit_position)

    def get_axioms_for_trait(self, bit_position: int) -> list[Axiom]:
        """
        Get all axioms for a trait.

        Args:
            bit_position: Trait bit position (1-32)

        Returns:
            List of axioms for the trait
        """
        trait = self.get_trait(bit_position)
        return trait.axioms if trait else []

    def get_necessary_axioms(self, bit_position: int) -> list[Axiom]:
        """
        Get only necessary axioms for a trait.

        Args:
            bit_position: Trait bit position (1-32)

        Returns:
            List of necessary axioms
        """
        return [a for a in self.get_axioms_for_trait(bit_position) if a.axiom_type == "necessary"]

    def get_typical_axioms(self, bit_position: int) -> list[Axiom]:
        """
        Get only typical axioms for a trait.

        Args:
            bit_position: Trait bit position (1-32)

        Returns:
            List of typical axioms with confidence
        """
        return [a for a in self.get_axioms_for_trait(bit_position) if a.axiom_type == "typical"]

    def get_all_traits(self) -> list[TraitAxioms]:
        """
        Get all trait axioms.

        Returns:
            List of all traits with their axioms
        """
        self.load()
        return list(self._traits.values())

    def get_traits_by_layer(self, layer: str) -> list[TraitAxioms]:
        """
        Get traits for a specific layer.

        Args:
            layer: Layer name (physical, functional, abstract, social)

        Returns:
            List of traits in that layer
        """
        self.load()
        return [t for t in self._traits.values() if t.layer == layer]
