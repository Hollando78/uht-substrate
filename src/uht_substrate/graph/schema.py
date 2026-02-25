"""Neo4j schema definitions for the knowledge graph."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeLabel:
    """Node labels used in the knowledge graph."""

    ENTITY = "Entity"
    TRAIT = "Trait"
    REASONING_TRACE = "ReasoningTrace"
    USER_CONTEXT = "UserContext"
    FACT = "Fact"
    PREFERENCE = "Preference"
    AXIOM = "Axiom"
    ONTOLOGICAL_COMMITMENT = "OntologicalCommitment"
    HEURISTIC = "Heuristic"
    LAYER = "Layer"


@dataclass(frozen=True)
class RelationshipType:
    """Relationship types used in the knowledge graph."""

    # Entity relationships
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    RELATED_TO = "RELATED_TO"
    SIMILAR_TO = "SIMILAR_TO"
    OPPOSITE_OF = "OPPOSITE_OF"

    # Trait relationships
    HAS_TRAIT = "HAS_TRAIT"
    IN_LAYER = "IN_LAYER"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"
    IMPLIES = "IMPLIES"

    # Reasoning trace relationships
    USED_ENTITY = "USED_ENTITY"
    APPLIED_AXIOM = "APPLIED_AXIOM"
    APPLIED_HEURISTIC = "APPLIED_HEURISTIC"
    DERIVED_FACT = "DERIVED_FACT"
    FOLLOWS_FROM = "FOLLOWS_FROM"

    # User context relationships
    OWNS_FACT = "OWNS_FACT"
    HAS_PREFERENCE = "HAS_PREFERENCE"
    INTERESTED_IN = "INTERESTED_IN"

    # Axiom relationships
    APPLIES_TO_TRAIT = "APPLIES_TO_TRAIT"


# Schema constraints and indexes (Cypher statements)
SCHEMA_CONSTRAINTS = """
// Uniqueness constraints
CREATE CONSTRAINT entity_uuid IF NOT EXISTS FOR (e:Entity) REQUIRE e.uuid IS UNIQUE;
CREATE CONSTRAINT trait_bit IF NOT EXISTS FOR (t:Trait) REQUIRE t.bit_position IS UNIQUE;
CREATE CONSTRAINT user_context_id IF NOT EXISTS FOR (uc:UserContext) REQUIRE uc.user_id IS UNIQUE;
CREATE CONSTRAINT axiom_uuid IF NOT EXISTS FOR (a:Axiom) REQUIRE a.uuid IS UNIQUE;
CREATE CONSTRAINT fact_uuid IF NOT EXISTS FOR (f:Fact) REQUIRE f.uuid IS UNIQUE;
CREATE CONSTRAINT preference_uuid IF NOT EXISTS FOR (p:Preference) REQUIRE p.uuid IS UNIQUE;
CREATE CONSTRAINT heuristic_uuid IF NOT EXISTS FOR (h:Heuristic) REQUIRE h.uuid IS UNIQUE;
CREATE CONSTRAINT reasoning_trace_uuid IF NOT EXISTS FOR (rt:ReasoningTrace) REQUIRE rt.uuid IS UNIQUE;
CREATE CONSTRAINT ontological_commitment_uuid IF NOT EXISTS FOR (oc:OntologicalCommitment) REQUIRE oc.uuid IS UNIQUE;
CREATE CONSTRAINT layer_name IF NOT EXISTS FOR (l:Layer) REQUIRE l.name IS UNIQUE;
"""

SCHEMA_INDEXES = """
// Indexes for common lookups
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_hex IF NOT EXISTS FOR (e:Entity) ON (e.hex_code);
CREATE INDEX entity_source IF NOT EXISTS FOR (e:Entity) ON (e.source);
CREATE INDEX trait_layer IF NOT EXISTS FOR (t:Trait) ON (t.layer);
CREATE INDEX trait_name IF NOT EXISTS FOR (t:Trait) ON (e.name);
CREATE INDEX fact_subject IF NOT EXISTS FOR (f:Fact) ON (f.subject);
CREATE INDEX fact_predicate IF NOT EXISTS FOR (f:Fact) ON (f.predicate);
CREATE INDEX reasoning_trace_created IF NOT EXISTS FOR (rt:ReasoningTrace) ON (rt.created_at);
CREATE INDEX axiom_trait IF NOT EXISTS FOR (a:Axiom) ON (a.trait_bit);
CREATE INDEX axiom_type IF NOT EXISTS FOR (a:Axiom) ON (a.axiom_type);
"""

# Initial seed data for the 4 layers
SEED_LAYERS = """
MERGE (l1:Layer {name: 'physical', bit_start: 1, bit_end: 8, description: 'Material properties and physical existence'})
MERGE (l2:Layer {name: 'functional', bit_start: 9, bit_end: 16, description: 'Capabilities and behaviors'})
MERGE (l3:Layer {name: 'abstract', bit_start: 17, bit_end: 24, description: 'Symbolic and conceptual aspects'})
MERGE (l4:Layer {name: 'social', bit_start: 25, bit_end: 32, description: 'Cultural and social dimensions'})
"""


def get_schema_statements() -> list[str]:
    """Get all schema initialization statements."""
    statements = []

    # Parse constraint statements
    for line in SCHEMA_CONSTRAINTS.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("//"):
            statements.append(line)

    # Parse index statements
    for line in SCHEMA_INDEXES.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("//"):
            statements.append(line)

    # Add layer seeds
    statements.append(SEED_LAYERS.strip())

    return statements
