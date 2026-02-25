"""Cypher query templates for the knowledge graph."""

# =============================================================================
# Entity Queries
# =============================================================================

UPSERT_ENTITY = """
MERGE (e:Entity {uuid: $uuid})
ON CREATE SET
    e.name = $name,
    e.hex_code = $hex_code,
    e.binary_code = $binary_code,
    e.description = $description,
    e.source = $source,
    e.created_at = datetime(),
    e.updated_at = datetime()
ON MATCH SET
    e.name = $name,
    e.hex_code = $hex_code,
    e.binary_code = $binary_code,
    e.description = COALESCE($description, e.description),
    e.source = $source,
    e.updated_at = datetime()
RETURN e
"""

FIND_ENTITY_BY_NAME = """
MATCH (e:Entity)
WHERE toLower(e.name) = toLower($name)
RETURN e
LIMIT 1
"""

FIND_ENTITY_BY_UUID = """
MATCH (e:Entity {uuid: $uuid})
RETURN e
"""

SEARCH_ENTITIES = """
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower($query)
RETURN e
ORDER BY e.name
LIMIT $limit
"""

FIND_SIMILAR_ENTITIES = """
MATCH (e1:Entity {uuid: $uuid})
MATCH (e2:Entity)
WHERE e1 <> e2
WITH e1, e2,
     [i IN range(0, 31) |
         CASE WHEN substring(e1.binary_code, i, 1) = substring(e2.binary_code, i, 1)
         THEN 1 ELSE 0 END
     ] as matches
WITH e1, e2, reduce(s = 0, x IN matches | s + x) as shared_traits
WHERE shared_traits >= $min_shared
RETURN e2 as entity, shared_traits, 32 - shared_traits as hamming_distance
ORDER BY shared_traits DESC
LIMIT $limit
"""

DELETE_ENTITY = """
MATCH (e:Entity {uuid: $uuid})
DETACH DELETE e
"""

# =============================================================================
# Trait Queries
# =============================================================================

UPSERT_TRAIT = """
MERGE (t:Trait {bit_position: $bit_position})
ON CREATE SET
    t.name = $name,
    t.layer = $layer,
    t.description = $description
ON MATCH SET
    t.name = $name,
    t.layer = $layer,
    t.description = $description
RETURN t
"""

GET_ALL_TRAITS = """
MATCH (t:Trait)
RETURN t
ORDER BY t.bit_position
"""

CONNECT_ENTITY_TO_TRAIT = """
MATCH (e:Entity {uuid: $entity_uuid})
MATCH (t:Trait {bit_position: $bit_position})
MERGE (e)-[r:HAS_TRAIT]->(t)
ON CREATE SET
    r.confidence = $confidence,
    r.evaluated_at = datetime()
ON MATCH SET
    r.confidence = $confidence,
    r.evaluated_at = datetime()
RETURN r
"""

GET_ENTITY_TRAITS = """
MATCH (e:Entity {uuid: $uuid})-[r:HAS_TRAIT]->(t:Trait)
RETURN t, r.confidence as confidence
ORDER BY t.bit_position
"""

# =============================================================================
# Relationship Queries
# =============================================================================

CREATE_ENTITY_RELATIONSHIP = """
MATCH (e1:Entity {uuid: $source_uuid})
MATCH (e2:Entity {uuid: $target_uuid})
CALL apoc.merge.relationship(e1, $rel_type, {}, $properties, e2, {}) YIELD rel
RETURN rel
"""

# Fallback without APOC
CREATE_IS_A_RELATIONSHIP = """
MATCH (e1:Entity {uuid: $source_uuid})
MATCH (e2:Entity {uuid: $target_uuid})
MERGE (e1)-[r:IS_A]->(e2)
SET r += $properties
RETURN r
"""

CREATE_SIMILAR_TO_RELATIONSHIP = """
MATCH (e1:Entity {uuid: $source_uuid})
MATCH (e2:Entity {uuid: $target_uuid})
MERGE (e1)-[r:SIMILAR_TO]->(e2)
SET r.similarity_score = $similarity_score,
    r.shared_traits = $shared_traits,
    r.created_at = datetime()
RETURN r
"""

CREATE_RELATED_TO_RELATIONSHIP = """
MATCH (e1:Entity {uuid: $source_uuid})
MATCH (e2:Entity {uuid: $target_uuid})
MERGE (e1)-[r:RELATED_TO]->(e2)
SET r += $properties
RETURN r
"""

GET_ENTITY_RELATIONSHIPS = """
MATCH (e:Entity {uuid: $uuid})-[r]-(other:Entity)
RETURN type(r) as rel_type,
       startNode(r) = e as outgoing,
       other,
       properties(r) as properties
"""

# =============================================================================
# Fact Queries
# =============================================================================

CREATE_FACT = """
CREATE (f:Fact {
    uuid: $uuid,
    subject: $subject,
    predicate: $predicate,
    object: $object,
    confidence: $confidence,
    source: $source,
    created_at: datetime()
})
RETURN f
"""

FIND_FACTS_BY_SUBJECT = """
MATCH (f:Fact)
WHERE toLower(f.subject) = toLower($subject)
RETURN f
ORDER BY f.created_at DESC
LIMIT $limit
"""

FIND_FACTS_BY_PREDICATE = """
MATCH (f:Fact)
WHERE f.predicate = $predicate
RETURN f
ORDER BY f.created_at DESC
LIMIT $limit
"""

LINK_FACT_TO_USER = """
MATCH (uc:UserContext {user_id: $user_id})
MATCH (f:Fact {uuid: $fact_uuid})
MERGE (uc)-[:OWNS_FACT]->(f)
RETURN f
"""

GET_USER_FACTS = """
MATCH (uc:UserContext {user_id: $user_id})-[:OWNS_FACT]->(f:Fact)
RETURN f
ORDER BY f.created_at DESC
LIMIT $limit
"""

# =============================================================================
# User Context Queries
# =============================================================================

UPSERT_USER_CONTEXT = """
MERGE (uc:UserContext {user_id: $user_id})
ON CREATE SET
    uc.created_at = datetime(),
    uc.updated_at = datetime()
ON MATCH SET
    uc.updated_at = datetime()
RETURN uc
"""

CREATE_PREFERENCE = """
MERGE (uc:UserContext {user_id: $user_id})
ON CREATE SET uc.created_at = datetime()
CREATE (p:Preference {
    uuid: $uuid,
    key: $key,
    value: $value,
    created_at: datetime()
})
MERGE (uc)-[:HAS_PREFERENCE]->(p)
RETURN p
"""

GET_USER_PREFERENCES = """
MATCH (uc:UserContext {user_id: $user_id})-[:HAS_PREFERENCE]->(p:Preference)
RETURN p.key as key, p.value as value
"""

MARK_USER_INTERESTED_IN = """
MATCH (uc:UserContext {user_id: $user_id})
MATCH (e:Entity {uuid: $entity_uuid})
MERGE (uc)-[r:INTERESTED_IN]->(e)
ON CREATE SET r.first_interaction = datetime()
SET r.last_interaction = datetime(),
    r.interaction_count = COALESCE(r.interaction_count, 0) + 1
RETURN r
"""

# =============================================================================
# Reasoning Trace Queries
# =============================================================================

CREATE_REASONING_TRACE = """
CREATE (rt:ReasoningTrace {
    uuid: $uuid,
    query: $query,
    conclusion: $conclusion,
    strategy: $strategy,
    confidence: $confidence,
    created_at: datetime()
})
RETURN rt
"""

LINK_TRACE_TO_ENTITY = """
MATCH (rt:ReasoningTrace {uuid: $trace_uuid})
MATCH (e:Entity {uuid: $entity_uuid})
MERGE (rt)-[:USED_ENTITY]->(e)
RETURN rt, e
"""

LINK_TRACE_TO_AXIOM = """
MATCH (rt:ReasoningTrace {uuid: $trace_uuid})
MATCH (a:Axiom {uuid: $axiom_uuid})
MERGE (rt)-[:APPLIED_AXIOM]->(a)
RETURN rt, a
"""

LINK_TRACE_TO_FACT = """
MATCH (rt:ReasoningTrace {uuid: $trace_uuid})
MATCH (f:Fact {uuid: $fact_uuid})
MERGE (rt)-[:DERIVED_FACT]->(f)
RETURN rt, f
"""

GET_RECENT_TRACES = """
MATCH (rt:ReasoningTrace)
WHERE rt.created_at >= datetime() - duration({hours: $hours})
RETURN rt
ORDER BY rt.created_at DESC
LIMIT $limit
"""

GET_TRACE_DETAILS = """
MATCH (rt:ReasoningTrace {uuid: $uuid})
OPTIONAL MATCH (rt)-[:USED_ENTITY]->(e:Entity)
OPTIONAL MATCH (rt)-[:APPLIED_AXIOM]->(a:Axiom)
OPTIONAL MATCH (rt)-[:DERIVED_FACT]->(f:Fact)
RETURN rt,
       collect(DISTINCT e) as entities,
       collect(DISTINCT a) as axioms,
       collect(DISTINCT f) as facts
"""

# =============================================================================
# Axiom Queries
# =============================================================================

UPSERT_AXIOM = """
MERGE (a:Axiom {uuid: $uuid})
ON CREATE SET
    a.trait_bit = $trait_bit,
    a.name = $name,
    a.statement = $statement,
    a.axiom_type = $axiom_type,
    a.property = $property,
    a.confidence = $confidence,
    a.created_at = datetime()
ON MATCH SET
    a.name = $name,
    a.statement = $statement,
    a.axiom_type = $axiom_type,
    a.property = $property,
    a.confidence = $confidence
RETURN a
"""

GET_AXIOMS_FOR_TRAIT = """
MATCH (a:Axiom)
WHERE a.trait_bit = $bit_position
RETURN a
ORDER BY a.axiom_type, a.name
"""

GET_ALL_AXIOMS = """
MATCH (a:Axiom)
RETURN a
ORDER BY a.trait_bit, a.axiom_type
"""

# =============================================================================
# Heuristic Queries
# =============================================================================

UPSERT_HEURISTIC = """
MERGE (h:Heuristic {uuid: $uuid})
ON CREATE SET
    h.name = $name,
    h.description = $description,
    h.priority = $priority,
    h.applicability_condition = $applicability_condition,
    h.implementation = $implementation
ON MATCH SET
    h.name = $name,
    h.description = $description,
    h.priority = $priority,
    h.applicability_condition = $applicability_condition,
    h.implementation = $implementation
RETURN h
"""

GET_ALL_HEURISTICS = """
MATCH (h:Heuristic)
RETURN h
ORDER BY h.priority DESC
"""

# =============================================================================
# Ontological Commitment Queries
# =============================================================================

UPSERT_ONTOLOGICAL_COMMITMENT = """
MERGE (oc:OntologicalCommitment {uuid: $uuid})
ON CREATE SET
    oc.name = $name,
    oc.statement = $statement,
    oc.category = $category
ON MATCH SET
    oc.name = $name,
    oc.statement = $statement,
    oc.category = $category
RETURN oc
"""

GET_ALL_ONTOLOGICAL_COMMITMENTS = """
MATCH (oc:OntologicalCommitment)
RETURN oc
ORDER BY oc.category, oc.name
"""

# =============================================================================
# Graph Statistics
# =============================================================================

GET_GRAPH_STATISTICS = """
MATCH (n)
WITH labels(n) as nodeLabels
UNWIND nodeLabels as label
RETURN label, count(*) as count
ORDER BY count DESC
"""

GET_RELATIONSHIP_STATISTICS = """
MATCH ()-[r]->()
RETURN type(r) as relationship_type, count(*) as count
ORDER BY count DESC
"""
