"""Prior-based inference engine for deriving facts from classifications."""

from dataclasses import dataclass, field
from typing import Optional

from ..config.logging import get_logger
from .heuristics import HeuristicRepository
from .ontology import OntologyRepository
from .trait_axioms import TraitAxiomRepository

logger = get_logger(__name__)


@dataclass
class InferredProperty:
    """A property inferred from classification using axioms."""

    property_name: str
    value: bool
    confidence: float
    source_axiom_uuid: str
    source_axiom_name: str
    reasoning_trace: list[str] = field(default_factory=list)


@dataclass
class InheritanceCheck:
    """Result of checking inheritance between entities."""

    is_valid: bool
    inheritance_score: float
    shared_trait_count: int
    parent_trait_count: int
    reasoning: list[str] = field(default_factory=list)


@dataclass
class SimilarityAnalysis:
    """Analysis of similarity between two entities."""

    hamming_distance: int
    jaccard_similarity: float  # |A ∩ B| / |A ∪ B| for present traits
    shared_traits: list[int]
    differing_traits: list[int]
    similarity_score: float  # Simple (32 - hamming) / 32
    can_transfer_properties: bool
    traits_a_only: list[int] = field(default_factory=list)  # Present in A but not B
    traits_b_only: list[int] = field(default_factory=list)  # Present in B but not A
    reasoning: list[str] = field(default_factory=list)


class PriorInferenceEngine:
    """Engine for applying priors to derive new facts from classifications."""

    def __init__(
        self,
        axioms: Optional[TraitAxiomRepository] = None,
        ontology: Optional[OntologyRepository] = None,
        heuristics: Optional[HeuristicRepository] = None,
    ):
        """
        Initialize the inference engine.

        Args:
            axioms: Trait axiom repository
            ontology: Ontology repository
            heuristics: Heuristics repository
        """
        self.axioms = axioms or TraitAxiomRepository()
        self.ontology = ontology or OntologyRepository()
        self.heuristics = heuristics or HeuristicRepository()

    def infer_properties(
        self,
        hex_code: str,
        entity_name: str,
        min_confidence: Optional[float] = None,
    ) -> list[InferredProperty]:
        """
        Derive properties from classification using trait axioms.

        Args:
            hex_code: 8-character hex code
            entity_name: Name of the entity
            min_confidence: Minimum confidence to include (uses heuristic default if None)

        Returns:
            List of inferred properties
        """
        if min_confidence is None:
            min_confidence = self.heuristics.get_possibility_threshold()

        binary = self._hex_to_binary(hex_code)
        inferred: list[InferredProperty] = []

        for bit_pos in range(1, 33):
            if binary[bit_pos - 1] == "1":
                # Trait is present - apply its axioms
                axioms = self.axioms.get_axioms_for_trait(bit_pos)

                for axiom in axioms:
                    confidence = axiom.confidence

                    # Skip if below minimum confidence
                    if confidence < min_confidence:
                        continue

                    # Build reasoning trace
                    reasoning = [
                        f"{entity_name} has trait '{axiom.trait_name}' (bit {bit_pos})",
                        f"Axiom ({axiom.axiom_type}): {axiom.statement}",
                    ]

                    if axiom.axiom_type == "necessary":
                        reasoning.append(f"Therefore: {entity_name} necessarily {axiom.property}")
                    else:
                        reasoning.append(
                            f"Therefore: {entity_name} likely {axiom.property} "
                            f"(confidence: {confidence:.0%})"
                        )

                    inferred.append(
                        InferredProperty(
                            property_name=axiom.property,
                            value=True,
                            confidence=confidence,
                            source_axiom_uuid=axiom.uuid,
                            source_axiom_name=axiom.name,
                            reasoning_trace=reasoning,
                        )
                    )

        logger.debug(
            "Inferred properties from classification",
            entity=entity_name,
            hex_code=hex_code,
            property_count=len(inferred),
        )

        return inferred

    def check_inheritance(
        self,
        child_hex: str,
        parent_hex: str,
        child_name: str = "child",
        parent_name: str = "parent",
    ) -> InheritanceCheck:
        """
        Check if child could inherit from parent based on trait subsumption.

        According to the ontology, if A is-a B, then A should have most of B's traits.

        Args:
            child_hex: Child entity hex code
            parent_hex: Parent entity hex code
            child_name: Name of child entity (for reasoning)
            parent_name: Name of parent entity (for reasoning)

        Returns:
            InheritanceCheck result
        """
        child_binary = self._hex_to_binary(child_hex)
        parent_binary = self._hex_to_binary(parent_hex)

        # Count shared traits where both have the trait
        shared = 0
        parent_traits = 0

        for i in range(32):
            if parent_binary[i] == "1":
                parent_traits += 1
                if child_binary[i] == "1":
                    shared += 1

        # Handle edge case
        if parent_traits == 0:
            return InheritanceCheck(
                is_valid=False,
                inheritance_score=0.0,
                shared_trait_count=0,
                parent_trait_count=0,
                reasoning=[f"{parent_name} has no traits to inherit"],
            )

        inheritance_score = shared / parent_traits

        # Get inheritance commitment for threshold
        inheritance_commitment = self.ontology.get_inheritance_commitment()
        threshold = 0.8  # Default
        if inheritance_commitment:
            threshold = inheritance_commitment.confidence

        is_valid = inheritance_score >= threshold

        reasoning = [
            f"{parent_name} has {parent_traits} trait(s)",
            f"{child_name} shares {shared} of those trait(s)",
            f"Inheritance score: {inheritance_score:.0%}",
            f"Threshold for IS_A: {threshold:.0%}",
            f"Conclusion: IS_A relationship {'valid' if is_valid else 'not valid'}",
        ]

        return InheritanceCheck(
            is_valid=is_valid,
            inheritance_score=inheritance_score,
            shared_trait_count=shared,
            parent_trait_count=parent_traits,
            reasoning=reasoning,
        )

    def analyze_similarity(
        self,
        hex_a: str,
        hex_b: str,
        name_a: str = "entity A",
        name_b: str = "entity B",
    ) -> SimilarityAnalysis:
        """
        Analyze similarity between two entities.

        Args:
            hex_a: First entity hex code
            hex_b: Second entity hex code
            name_a: Name of first entity
            name_b: Name of second entity

        Returns:
            SimilarityAnalysis result
        """
        binary_a = self._hex_to_binary(hex_a)
        binary_b = self._hex_to_binary(hex_b)

        shared_traits: list[int] = []  # 1 in both
        differing_traits: list[int] = []  # Different between A and B
        traits_a_only: list[int] = []  # 1 in A, 0 in B
        traits_b_only: list[int] = []  # 0 in A, 1 in B

        for i in range(32):
            bit_pos = i + 1
            a_has = binary_a[i] == "1"
            b_has = binary_b[i] == "1"

            if a_has and b_has:
                shared_traits.append(bit_pos)
            elif a_has and not b_has:
                traits_a_only.append(bit_pos)
                differing_traits.append(bit_pos)
            elif b_has and not a_has:
                traits_b_only.append(bit_pos)
                differing_traits.append(bit_pos)
            # else: both 0, neither has trait

        hamming_distance = len(differing_traits)
        similarity_score = (32 - hamming_distance) / 32

        # Jaccard similarity: |intersection| / |union| of present traits
        intersection = len(shared_traits)
        union = len(shared_traits) + len(traits_a_only) + len(traits_b_only)
        jaccard_similarity = intersection / union if union > 0 else 1.0

        # Check if we can transfer properties based on Jaccard similarity
        # Jaccard is more meaningful than Hamming because it ignores shared zeros
        min_jaccard = 0.5  # Require at least 50% trait overlap
        can_transfer = jaccard_similarity >= min_jaccard

        reasoning = [
            f"Comparing {name_a} ({hex_a}) with {name_b} ({hex_b})",
            f"Jaccard similarity: {jaccard_similarity:.0%} (shared traits / all traits)",
            f"Hamming distance: {hamming_distance} (bit differences)",
            f"Shared traits: {len(shared_traits)}, {name_a} only: {len(traits_a_only)}, {name_b} only: {len(traits_b_only)}",
        ]

        if can_transfer:
            reasoning.append(
                f"Entities share enough traits (Jaccard >= {min_jaccard:.0%}) for property transfer"
            )
        else:
            reasoning.append(
                f"Entities have low trait overlap (Jaccard < {min_jaccard:.0%}), property transfer not recommended"
            )

        return SimilarityAnalysis(
            hamming_distance=hamming_distance,
            jaccard_similarity=jaccard_similarity,
            shared_traits=shared_traits,
            differing_traits=differing_traits,
            similarity_score=similarity_score,
            can_transfer_properties=can_transfer,
            traits_a_only=traits_a_only,
            traits_b_only=traits_b_only,
            reasoning=reasoning,
        )

    def apply_confidence_decay(
        self,
        initial_confidence: float,
        chain_length: int,
    ) -> float:
        """
        Apply confidence decay for multi-step inferences.

        Args:
            initial_confidence: Starting confidence
            chain_length: Number of inference steps

        Returns:
            Decayed confidence value
        """
        decay_factor = self.heuristics.get_confidence_decay_factor()
        return initial_confidence * (decay_factor ** chain_length)

    def should_assert_as_fact(self, confidence: float) -> bool:
        """
        Check if confidence is high enough to assert as fact.

        Args:
            confidence: Confidence level

        Returns:
            True if should assert as fact
        """
        return confidence >= self.heuristics.get_confidence_threshold()

    def should_store_as_possibility(self, confidence: float) -> bool:
        """
        Check if confidence is high enough to store as possibility.

        Args:
            confidence: Confidence level

        Returns:
            True if should store as possibility
        """
        return confidence >= self.heuristics.get_possibility_threshold()

    def get_layer_for_query(self, query: str) -> Optional[str]:
        """
        Determine which layer is most relevant for a query.

        Based on the Layer Specialization heuristic.

        Args:
            query: The user's query

        Returns:
            Layer name or None if no specific layer
        """
        query_lower = query.lower()

        # Physical layer indicators
        if any(
            phrase in query_lower
            for phrase in ["made of", "material", "physical", "weight", "size", "location"]
        ):
            return "physical"

        # Functional layer indicators
        if any(
            phrase in query_lower
            for phrase in ["can it", "what can", "how does it work", "function", "capability"]
        ):
            return "functional"

        # Abstract layer indicators
        if any(
            phrase in query_lower
            for phrase in ["what does it mean", "symbol", "represent", "concept", "idea"]
        ):
            return "abstract"

        # Social layer indicators
        if any(
            phrase in query_lower
            for phrase in ["who uses", "cultural", "social", "institution", "value", "economic"]
        ):
            return "social"

        return None

    def _hex_to_binary(self, hex_code: str) -> str:
        """Convert 8-char hex code to 32-bit binary string."""
        return bin(int(hex_code, 16))[2:].zfill(32)
