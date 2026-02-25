"""Pydantic models for UHT Factory API responses."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Layer(str, Enum):
    """The four UHT classification layers."""

    PHYSICAL = "physical"  # Bits 1-8
    FUNCTIONAL = "functional"  # Bits 9-16
    ABSTRACT = "abstract"  # Bits 17-24
    SOCIAL = "social"  # Bits 25-32

    @classmethod
    def _missing_(cls, value: object) -> "Layer | None":
        """Handle case-insensitive lookup."""
        if isinstance(value, str):
            lower = value.lower()
            for member in cls:
                if member.value == lower:
                    return member
        return None


class TraitValue(BaseModel):
    """Single trait evaluation result from classification."""

    bit_position: int = Field(ge=1, le=32, alias="trait_bit", description="Bit position 1-32")
    name: str = Field(alias="trait_name", description="Trait name")
    present: bool = Field(alias="applicable", description="Whether trait is present")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    justification: Optional[str] = Field(default=None, description="Reasoning for assessment")

    model_config = {"populate_by_name": True}

    @property
    def layer(self) -> Layer:
        """Determine layer from bit position."""
        if self.bit_position <= 8:
            return Layer.PHYSICAL
        elif self.bit_position <= 16:
            return Layer.FUNCTIONAL
        elif self.bit_position <= 24:
            return Layer.ABSTRACT
        else:
            return Layer.SOCIAL


class ClassificationResult(BaseModel):
    """Result from POST /classify/ endpoint."""

    uuid: str = Field(description="Entity UUID")
    name: str = Field(description="Entity name")
    hex_code: str = Field(alias="uht_code", pattern=r"^[0-9A-Fa-f]{8}$", description="8-char hex code")
    binary: str = Field(alias="binary_representation", pattern=r"^[01]{32}$", description="32-bit binary string")
    traits: list[TraitValue] = Field(default_factory=list, alias="trait_evaluations", description="Evaluated traits")
    created_at: datetime = Field(description="Classification timestamp")

    model_config = {"populate_by_name": True}

    @property
    def entity(self) -> str:
        """Alias for name for backwards compatibility."""
        return self.name

    @property
    def physical_byte(self) -> str:
        """Get physical layer hex byte (bits 1-8)."""
        return self.hex_code[:2]

    @property
    def functional_byte(self) -> str:
        """Get functional layer hex byte (bits 9-16)."""
        return self.hex_code[2:4]

    @property
    def abstract_byte(self) -> str:
        """Get abstract layer hex byte (bits 17-24)."""
        return self.hex_code[4:6]

    @property
    def social_byte(self) -> str:
        """Get social layer hex byte (bits 25-32)."""
        return self.hex_code[6:8]

    def has_trait(self, bit: int) -> bool:
        """Check if trait at bit position is present."""
        if not 1 <= bit <= 32:
            raise ValueError(f"Bit position must be 1-32, got {bit}")
        return self.binary[bit - 1] == "1"

    def get_layer_traits(self, layer: Layer) -> list[TraitValue]:
        """Get all traits for a specific layer."""
        return [t for t in self.traits if t.layer == layer]

    def get_present_traits(self) -> list[TraitValue]:
        """Get all present traits."""
        return [t for t in self.traits if t.present]


class Entity(BaseModel):
    """Entity from UHT Factory."""

    uuid: str = Field(description="Entity UUID")
    name: str = Field(description="Primary name")
    hex_code: str = Field(pattern=r"^[0-9A-Fa-f]{8}$", description="8-char hex code")
    binary: Optional[str] = Field(default=None, description="32-bit binary string")
    description: Optional[str] = Field(default=None, description="Entity description")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    traits: list[TraitValue] = Field(default_factory=list, description="Trait assessments")
    wikidata_id: Optional[str] = Field(default=None, description="Wikidata Q-ID if linked")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update")

    def has_trait(self, bit: int) -> bool:
        """Check if trait at bit position is present."""
        if self.binary:
            return self.binary[bit - 1] == "1"
        return any(t.bit_position == bit and t.present for t in self.traits)


class TraitDefinition(BaseModel):
    """Definition of a single UHT trait."""

    bit_position: int = Field(ge=1, le=32, alias="bit", description="Bit position 1-32")
    name: str = Field(description="Trait name")
    layer: Layer = Field(description="Classification layer")
    description: str = Field(default="", alias="desc", description="What this trait means")
    examples_present: list[str] = Field(default_factory=list, description="Entities with trait")
    examples_absent: list[str] = Field(default_factory=list, description="Entities without trait")
    classifier_prompt: Optional[str] = Field(default=None, description="LLM evaluation prompt")

    model_config = {"populate_by_name": True}


class SemanticTriangle(BaseModel):
    """Ogden-Richards semantic triangle decomposition."""

    symbol: str = Field(description="The word/phrase (linguistic form)")
    referent: str = Field(description="The real-world thing it refers to")
    reference: str = Field(description="The mental concept/meaning")


class SimilarityResult(BaseModel):
    """Result from similarity search."""

    entity: Entity = Field(description="Similar entity")
    similarity_score: float = Field(ge=0.0, le=1.0, description="Similarity score")
    hamming_distance: Optional[int] = Field(default=None, description="Hamming distance")
    shared_traits: list[int] = Field(default_factory=list, description="Shared trait positions")


class NeighborhoodNode(BaseModel):
    """Node in neighborhood graph."""

    uuid: str = Field(description="Entity UUID")
    name: str = Field(description="Entity name")
    hex_code: str = Field(description="8-char hex code")


class NeighborhoodEdge(BaseModel):
    """Edge in neighborhood graph."""

    source: str = Field(description="Source entity UUID")
    target: str = Field(description="Target entity UUID")
    relationship: str = Field(description="Relationship type")
    weight: float = Field(default=1.0, description="Edge weight")


class NeighborhoodResult(BaseModel):
    """Result from neighborhood exploration."""

    center: str = Field(description="Center entity UUID")
    nodes: list[NeighborhoodNode] = Field(default_factory=list, description="Nodes in neighborhood")
    edges: list[NeighborhoodEdge] = Field(default_factory=list, description="Edges between nodes")


class DisambiguationSense(BaseModel):
    """A single sense of a polysemous word."""

    sense_id: str = Field(description="Sense identifier")
    definition: str = Field(description="Sense definition")
    hex_code: Optional[str] = Field(default=None, description="UHT code if classified")
    entity_uuid: Optional[str] = Field(default=None, description="Linked entity UUID")
    examples: list[str] = Field(default_factory=list, description="Usage examples")


class DisambiguationResult(BaseModel):
    """Result from polysemy disambiguation."""

    lemma: str = Field(description="The word being disambiguated")
    language: str = Field(default="en", description="Language code")
    senses: list[DisambiguationSense] = Field(default_factory=list, description="Word senses")


class PreprocessingResult(BaseModel):
    """Result from entity preprocessing."""

    entity_name: str = Field(description="Original entity name")
    normalized_name: str = Field(description="Normalized form")
    semantic_triangle: Optional[SemanticTriangle] = Field(default=None)
    suggested_context: Optional[str] = Field(default=None)
    potential_duplicates: list[Entity] = Field(default_factory=list)
