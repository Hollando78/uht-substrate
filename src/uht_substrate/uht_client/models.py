"""Pydantic models for UHT Factory API responses."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def parse_neo4j_datetime(value: Any) -> datetime:
    """Parse Neo4j's broken datetime serialization format."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # ISO format string
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, dict):
        # Neo4j's weird internal format: _DateTime__date, _DateTime__time
        date_part = value.get("_DateTime__date", {})
        time_part = value.get("_DateTime__time", {})
        return datetime(
            year=date_part.get("_Date__year", 2000),
            month=date_part.get("_Date__month", 1),
            day=date_part.get("_Date__day", 1),
            hour=time_part.get("_Time__hour", 0),
            minute=time_part.get("_Time__minute", 0),
            second=time_part.get("_Time__second", 0),
        )
    return datetime.utcnow()


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

    bit_position: int = Field(ge=1, le=32, description="Bit position 1-32")
    name: str = Field(default="", description="Trait name")
    present: bool = Field(default=False, description="Whether trait is present")
    confidence: float = Field(ge=0.0, le=1.0, default=0.0, description="Confidence score")
    justification: Optional[str] = Field(default=None, description="Reasoning for assessment")

    model_config = {"populate_by_name": True}

    @model_validator(mode="before")
    @classmethod
    def normalize_trait_format(cls, data: Any) -> Any:
        """Handle different API response formats for traits."""
        if not isinstance(data, dict):
            return data

        # Handle entity API format: {bit, name, evaluation: {applicable, confidence}}
        if "bit" in data and "evaluation" in data:
            evaluation = data.get("evaluation", {})
            return {
                "bit_position": data.get("bit"),
                "name": data.get("name", ""),
                "present": evaluation.get("applicable", False),
                "confidence": evaluation.get("confidence", 0.0),
                "justification": evaluation.get("justification"),
            }

        # Handle classification API format: {trait_bit, trait_name, applicable, confidence}
        if "trait_bit" in data:
            return {
                "bit_position": data.get("trait_bit"),
                "name": data.get("trait_name", ""),
                "present": data.get("applicable", False),
                "confidence": data.get("confidence", 0.0),
                "justification": data.get("justification"),
            }

        return data

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

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_created_at(cls, v: Any) -> datetime:
        return parse_neo4j_datetime(v)

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
    hex_code: str = Field(alias="uht_code", pattern=r"^[0-9A-Fa-f]{8}$", description="8-char hex code")
    binary: Optional[str] = Field(default=None, alias="binary_representation", description="32-bit binary string")
    description: Optional[str] = Field(default=None, description="Entity description")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    traits: list[TraitValue] = Field(default_factory=list, description="Trait assessments")
    wikidata_id: Optional[str] = Field(default=None, description="Wikidata Q-ID if linked")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update")

    model_config = {"populate_by_name": True}

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def parse_datetime_fields(cls, v: Any) -> datetime | None:
        if v is None:
            return None
        return parse_neo4j_datetime(v)

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
    short_description: str = Field(default="", description="Brief description")
    expanded_definition: str = Field(default="", description="Full definition")
    url: Optional[str] = Field(default=None, description="Documentation URL")
    examples_present: list[str] = Field(default_factory=list, description="Entities with trait")
    examples_absent: list[str] = Field(default_factory=list, description="Entities without trait")
    classifier_prompt: Optional[str] = Field(default=None, description="LLM evaluation prompt")

    model_config = {"populate_by_name": True}

    @property
    def description(self) -> str:
        """Get description (short_description for backwards compat)."""
        return self.short_description


class SymbolComponent(BaseModel):
    """Symbol component of semantic triangle."""

    form: str = Field(description="The word/phrase")
    polysemy_detected: bool = Field(default=False, description="Whether multiple senses detected")
    intended_sense: str = Field(default="", description="The intended meaning")
    other_senses: list[str] = Field(default_factory=list, description="Alternative senses")


class ThoughtComponent(BaseModel):
    """Thought/reference component of semantic triangle."""

    definition: str = Field(description="Conceptual definition")
    essential_properties: list[str] = Field(default_factory=list, description="Key properties")
    category: str = Field(default="", description="Ontological category")
    distinguishing_features: list[str] = Field(default_factory=list, description="What makes it unique")


class ReferentComponent(BaseModel):
    """Referent component of semantic triangle."""

    description: str = Field(description="Description of the real-world thing")
    typical_instances: list[str] = Field(default_factory=list, description="Common examples")
    boundaries: str = Field(default="", description="What it includes/excludes")
    ontological_status: str = Field(default="", description="Type of existence")


class SemanticTriangle(BaseModel):
    """Ogden-Richards semantic triangle decomposition."""

    symbol: SymbolComponent = Field(description="The word/phrase (linguistic form)")
    thought: ThoughtComponent = Field(description="The mental concept/meaning")
    referent: ReferentComponent = Field(description="The real-world thing it refers to")
    disambiguation_confidence: float = Field(default=0.0, description="Confidence in disambiguation")
    enriched_context: str = Field(default="", description="Summary context")


class SimilarityResult(BaseModel):
    """Result from similarity search.

    Note: Factory API returns similarity_score as shared trait count (0-32),
    not a normalized 0-1 score. We normalize it in the model.
    """

    uuid: str = Field(description="Entity UUID")
    name: str = Field(description="Entity name")
    hex_code: str = Field(alias="uht_code", description="8-char hex code")
    description: Optional[str] = Field(default=None, description="Entity description")
    shared_trait_count: int = Field(alias="similarity_score", ge=0, le=32, description="Number of shared traits")
    binary: Optional[str] = Field(default=None, alias="binary_representation", description="32-bit binary")

    model_config = {"populate_by_name": True}

    @property
    def similarity_score(self) -> float:
        """Get normalized similarity score (0-1)."""
        return self.shared_trait_count / 32

    @property
    def entity(self) -> "Entity":
        """Get entity for backwards compatibility."""
        return Entity(
            uuid=self.uuid,
            name=self.name,
            hex_code=self.hex_code,
            description=self.description,
            binary=self.binary,
            created_at=datetime.utcnow(),  # Placeholder
        )


class SemanticSearchResult(BaseModel):
    """Result from embedding-based semantic search."""

    uuid: str = Field(description="Entity UUID")
    name: str = Field(description="Entity name")
    description: Optional[str] = Field(default=None, description="Entity description")
    hex_code: str = Field(alias="uht_code", description="8-char hex code")
    image_url: Optional[str] = Field(default=None, description="Image URL if available")
    similarity_score: float = Field(ge=0.0, le=1.0, description="Semantic similarity score")

    model_config = {"populate_by_name": True}


class NeighborhoodNode(BaseModel):
    """Node in neighborhood graph."""

    uuid: str = Field(alias="id", description="Entity UUID")
    name: str = Field(description="Entity name")
    hex_code: str = Field(default="", alias="uht_code", description="8-char hex code")
    node_type: str = Field(default="entity", alias="type", description="Node type")
    description: Optional[str] = Field(default=None, description="Node description")

    model_config = {"populate_by_name": True}


class NeighborhoodEdge(BaseModel):
    """Edge in neighborhood graph."""

    source: str = Field(description="Source entity UUID")
    target: str = Field(description="Target entity UUID")
    relationship: str = Field(default="similar", description="Relationship type")
    weight: float = Field(default=1.0, alias="value", description="Edge weight")

    model_config = {"populate_by_name": True}


class NeighborhoodResult(BaseModel):
    """Result from neighborhood exploration."""

    center: NeighborhoodNode = Field(description="Center entity")
    nodes: list[NeighborhoodNode] = Field(default_factory=list, description="Nodes in neighborhood")
    edges: list[NeighborhoodEdge] = Field(default_factory=list, alias="links", description="Edges between nodes")

    model_config = {"populate_by_name": True}


class DisambiguationSense(BaseModel):
    """A single sense of a polysemous word."""

    definition: str = Field(alias="definition_en", description="Sense definition")
    hex_code: Optional[str] = Field(default=None, alias="uht_code", description="UHT code if classified")
    entity_uuid: Optional[str] = Field(default=None, description="Linked entity UUID")
    examples: list[str] = Field(default_factory=list, description="Usage examples")
    traits: list[dict] = Field(default_factory=list, description="Trait evaluations")

    model_config = {"populate_by_name": True}


class DisambiguationWord(BaseModel):
    """Word metadata from disambiguation."""

    lemma: str = Field(description="The word lemma")
    language: str = Field(default="en", description="Language code")
    sense_count: int = Field(default=0, description="Number of senses")
    tier: str = Field(default="", description="Word tier")


class DisambiguationResult(BaseModel):
    """Result from polysemy disambiguation."""

    word: DisambiguationWord = Field(description="Word metadata")
    senses: list[DisambiguationSense] = Field(
        default_factory=list, alias="classified_senses", description="Word senses"
    )

    model_config = {"populate_by_name": True}

    @property
    def lemma(self) -> str:
        """Get lemma from word object for backwards compatibility."""
        return self.word.lemma

    @property
    def language(self) -> str:
        """Get language from word object for backwards compatibility."""
        return self.word.language


class PreprocessingResult(BaseModel):
    """Result from entity preprocessing."""

    entity_name: str = Field(description="Original entity name")
    normalized_name: str = Field(description="Normalized form")
    semantic_triangle: Optional[SemanticTriangle] = Field(default=None)
    suggested_context: Optional[str] = Field(default=None)
    potential_duplicates: list[Entity] = Field(default_factory=list)
