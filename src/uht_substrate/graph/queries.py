"""Cypher query templates for the knowledge graph."""

# =============================================================================
# Entity Queries
# =============================================================================

UPSERT_ENTITY = """
MERGE (e:Entity {name: $name})
ON CREATE SET
    e.uuid = $uuid,
    e.hex_code = $hex_code,
    e.binary_code = $binary_code,
    e.description = $description,
    e.source = $source,
    e.created_at = datetime(),
    e.updated_at = datetime()
ON MATCH SET
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
    r.justification = $justification,
    r.evaluated_at = datetime()
ON MATCH SET
    r.confidence = $confidence,
    r.justification = $justification,
    r.evaluated_at = datetime()
RETURN r
"""

GET_ENTITY_TRAITS = """
MATCH (e:Entity {uuid: $uuid})-[r:HAS_TRAIT]->(t:Trait)
RETURN t, r.confidence as confidence, r.justification as justification
ORDER BY t.bit_position
"""

FIND_ENTITY_WITH_TRAITS_BY_NAME = """
MATCH (e:Entity)
WHERE toLower(e.name) = toLower($name)
OPTIONAL MATCH (e)-[r:HAS_TRAIT]->(t:Trait)
RETURN e, collect({bit_position: t.bit_position, name: t.name, confidence: r.confidence, justification: r.justification}) AS traits
LIMIT 1
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

FIND_DUPLICATE_FACT = """
MATCH (f:Fact)
WHERE toLower(f.subject) = toLower($subject)
  AND f.predicate = $predicate
  AND toLower(f.object) = toLower($object)
RETURN f
LIMIT 1
"""

FIND_FACT_FOR_UPSERT = """
MATCH (uc:UserContext {user_id: $user_id})-[:OWNS_FACT]->(f:Fact)
WHERE toLower(f.subject) = toLower($subject)
  AND f.predicate = $predicate
RETURN f
LIMIT 1
"""

FIND_FACT_FOR_UPSERT_GLOBAL = """
MATCH (f:Fact)
WHERE toLower(f.subject) = toLower($subject)
  AND f.predicate = $predicate
RETURN f
LIMIT 1
"""

CREATE_FACT = """
CREATE (f:Fact {
    uuid: $uuid,
    subject: $subject,
    predicate: $predicate,
    object: $object,
    confidence: $confidence,
    source: $source,
    category: $category,
    is_custom_predicate: $is_custom_predicate,
    bound: false,
    created_at: datetime(),
    updated_at: datetime()
})
RETURN f
"""

FIND_FACT_BY_UUID = """
MATCH (f:Fact {uuid: $uuid})
RETURN f
"""

UPDATE_FACT = """
MATCH (f:Fact {uuid: $uuid})
SET f.subject = COALESCE($subject, f.subject),
    f.predicate = COALESCE($predicate, f.predicate),
    f.object = COALESCE($object, f.object),
    f.confidence = COALESCE($confidence, f.confidence),
    f.category = COALESCE($category, f.category),
    f.is_custom_predicate = COALESCE($is_custom_predicate, f.is_custom_predicate),
    f.updated_at = datetime()
RETURN f
"""

DELETE_FACT = """
MATCH (f:Fact {uuid: $uuid})
WITH f, f.uuid AS deleted_uuid
DETACH DELETE f
RETURN deleted_uuid
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

QUERY_FACTS = """
MATCH (f:Fact)
WHERE ($subject IS NULL OR toLower(f.subject) = toLower($subject))
  AND ($object IS NULL OR toLower(f.object) = toLower($object))
  AND ($predicate IS NULL OR f.predicate = $predicate)
  AND ($category IS NULL OR f.category = $category)
  AND ($source IS NULL OR f.source = $source)
RETURN f
ORDER BY f.created_at DESC
LIMIT $limit
"""

QUERY_USER_FACTS = """
MATCH (uc:UserContext {user_id: $user_id})-[:OWNS_FACT]->(f:Fact)
WHERE ($subject IS NULL OR toLower(f.subject) = toLower($subject))
  AND ($object IS NULL OR toLower(f.object) = toLower($object))
  AND ($predicate IS NULL OR f.predicate = $predicate)
  AND ($category IS NULL OR f.category = $category)
  AND ($source IS NULL OR f.source = $source)
RETURN f
ORDER BY f.created_at DESC
LIMIT $limit
"""

QUERY_FACTS_IN_NAMESPACE = """
MATCH (root:Namespace {code: $namespace_code})
MATCH (root)-[:PARENT_OF*0..]->(ns:Namespace)
MATCH (e:Entity)-[:BELONGS_TO]->(ns)
MATCH (f:Fact)-[:FACT_ABOUT]->(e)
WHERE ($subject IS NULL OR toLower(f.subject) = toLower($subject))
  AND ($object IS NULL OR toLower(f.object) = toLower($object))
  AND ($predicate IS NULL OR f.predicate = $predicate)
  AND ($category IS NULL OR f.category = $category)
  AND ($source IS NULL OR f.source = $source)
RETURN DISTINCT f
ORDER BY f.created_at DESC
LIMIT $limit
"""

QUERY_USER_FACTS_IN_NAMESPACE = """
MATCH (root:Namespace {code: $namespace_code})
MATCH (root)-[:PARENT_OF*0..]->(ns:Namespace)
MATCH (e:Entity)-[:BELONGS_TO]->(ns)
MATCH (uc:UserContext {user_id: $user_id})-[:OWNS_FACT]->(f:Fact)-[:FACT_ABOUT]->(e)
WHERE ($subject IS NULL OR toLower(f.subject) = toLower($subject))
  AND ($object IS NULL OR toLower(f.object) = toLower($object))
  AND ($predicate IS NULL OR f.predicate = $predicate)
  AND ($category IS NULL OR f.category = $category)
  AND ($source IS NULL OR f.source = $source)
RETURN DISTINCT f
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

GET_USER_FACTS_WITH_BINDING = """
MATCH (uc:UserContext {user_id: $user_id})-[:OWNS_FACT]->(f:Fact)
OPTIONAL MATCH (f)-[:FACT_ABOUT]->(subj_entity:Entity)
OPTIONAL MATCH (f)-[:FACT_REFERENCES]->(obj_entity:Entity)
RETURN f,
       subj_entity.name AS subject_entity_name,
       obj_entity.name AS object_entity_name
ORDER BY f.category, f.created_at DESC
LIMIT $limit
"""

# --- Entity Binding Queries ---

BIND_FACT_TO_SUBJECT_ENTITY = """
MATCH (f:Fact {uuid: $fact_uuid})
MATCH (e:Entity)
WHERE toLower(e.name) = toLower($subject)
MERGE (f)-[:FACT_ABOUT]->(e)
RETURN e.uuid AS entity_uuid, e.name AS entity_name
"""

BIND_FACT_TO_OBJECT_ENTITY = """
MATCH (f:Fact {uuid: $fact_uuid})
MATCH (e:Entity)
WHERE toLower(e.name) = toLower($object)
MERGE (f)-[:FACT_REFERENCES]->(e)
RETURN e.uuid AS entity_uuid, e.name AS entity_name
"""

MARK_FACT_BOUND = """
MATCH (f:Fact {uuid: $uuid})
SET f.bound = true,
    f.subject_entity_uuid = $subject_entity_uuid,
    f.object_entity_uuid = $object_entity_uuid,
    f.updated_at = datetime()
RETURN f
"""

FIND_UNBOUND_FACTS = """
MATCH (f:Fact)
WHERE f.bound = false OR f.bound IS NULL
RETURN f
ORDER BY f.created_at ASC
LIMIT $limit
"""

FIND_UNBOUND_FACTS_FOR_ENTITY = """
MATCH (f:Fact)
WHERE (f.bound = false OR f.bound IS NULL)
  AND (toLower(f.subject) = toLower($entity_name)
       OR toLower(f.object) = toLower($entity_name))
RETURN f
ORDER BY f.created_at ASC
"""

CREATE_ENTITY_RELATIONSHIP_FROM_FACT = """
MATCH (e1:Entity {uuid: $source_uuid})
MATCH (e2:Entity {uuid: $target_uuid})
MERGE (e1)-[r:RELATED_TO {predicate: $predicate}]->(e2)
SET r.fact_uuid = $fact_uuid,
    r.category = $category,
    r.source = $source,
    r.user_id = $user_id,
    r.confidence = $confidence,
    r.created_at = datetime()
RETURN r
"""

DELETE_ENTITY_RELATIONSHIP_BY_FACT = """
MATCH (e1:Entity)-[r:RELATED_TO {fact_uuid: $fact_uuid}]->(e2:Entity)
DELETE r
RETURN count(r) AS deleted_count
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
# Namespace Queries
# =============================================================================

CREATE_NAMESPACE = """
MERGE (n:Namespace {code: $code})
ON CREATE SET
    n.uuid = randomUUID(),
    n.name = $name,
    n.description = $description,
    n.created_at = datetime(),
    n.is_root = CASE WHEN $parent_code IS NULL THEN true ELSE false END
ON MATCH SET
    n.name = $name,
    n.description = COALESCE($description, n.description)
WITH n
OPTIONAL MATCH (parent:Namespace {code: $parent_code})
FOREACH (_ IN CASE WHEN parent IS NOT NULL THEN [1] ELSE [] END |
    MERGE (parent)-[:PARENT_OF]->(n)
)
RETURN n
"""

FIND_NAMESPACE_BY_CODE = """
MATCH (n:Namespace {code: $code})
RETURN n
"""

LIST_ROOT_NAMESPACES = """
MATCH (n:Namespace)
WHERE n.is_root = true OR NOT EXISTS { (:Namespace)-[:PARENT_OF]->(n) }
RETURN n
ORDER BY n.code
"""

LIST_NAMESPACE_CHILDREN = """
MATCH (parent:Namespace {code: $parent_code})-[:PARENT_OF]->(n:Namespace)
RETURN n
ORDER BY n.code
"""

LIST_NAMESPACE_DESCENDANTS = """
MATCH (root:Namespace {code: $code})
MATCH (root)-[:PARENT_OF*0..]->(n:Namespace)
RETURN n
ORDER BY n.code
"""

GET_NAMESPACE_PATH = """
MATCH (n:Namespace {code: $code})
MATCH path = (root:Namespace)-[:PARENT_OF*0..]->(n)
WHERE root.is_root = true OR NOT EXISTS { (:Namespace)-[:PARENT_OF]->(root) }
RETURN [node in nodes(path) | node.code] AS path
"""

DELETE_NAMESPACE = """
MATCH (n:Namespace {code: $code})
DETACH DELETE n
"""

DELETE_NAMESPACE_CASCADE = """
MATCH (root:Namespace {code: $code})
OPTIONAL MATCH (root)-[:PARENT_OF*0..]->(descendant:Namespace)
DETACH DELETE descendant
DETACH DELETE root
"""

ASSIGN_ENTITY_TO_NAMESPACE = """
MATCH (e:Entity {uuid: $entity_uuid})
MATCH (n:Namespace {code: $namespace_code})
MERGE (e)-[r:BELONGS_TO]->(n)
SET r.primary = $primary,
    r.assigned_at = datetime()
RETURN r
"""

REMOVE_ENTITY_FROM_NAMESPACE = """
MATCH (e:Entity {uuid: $entity_uuid})-[r:BELONGS_TO]->(n:Namespace {code: $namespace_code})
DELETE r
"""

GET_ENTITY_NAMESPACES = """
MATCH (e:Entity {uuid: $entity_uuid})-[r:BELONGS_TO]->(n:Namespace)
RETURN n, r.primary as is_primary
ORDER BY r.primary DESC, n.code
"""

LIST_ENTITIES_IN_NAMESPACE = """
MATCH (root:Namespace {code: $namespace_code})
MATCH (root)-[:PARENT_OF*0..]->(ns:Namespace)
MATCH (e:Entity)-[:BELONGS_TO]->(ns)
RETURN DISTINCT e
ORDER BY e.created_at DESC
LIMIT $limit
"""

LIST_ENTITIES_IN_NAMESPACE_FILTERED = """
MATCH (root:Namespace {code: $namespace_code})
MATCH (root)-[:PARENT_OF*0..]->(ns:Namespace)
MATCH (e:Entity)-[:BELONGS_TO]->(ns)
WHERE ($name_filter IS NULL OR toLower(e.name) CONTAINS toLower($name_filter))
  AND ($hex_filter IS NULL OR e.hex_code STARTS WITH $hex_filter)
RETURN DISTINCT e
ORDER BY e.created_at DESC
LIMIT $limit
"""

GET_NAMESPACE_CONTEXT = """
MATCH (root:Namespace {code: $namespace_code})
MATCH (root)-[:PARENT_OF*0..]->(ns:Namespace)
MATCH (e:Entity)-[:BELONGS_TO]->(ns)
WITH DISTINCT e
OPTIONAL MATCH (f:Fact)-[:FACT_ABOUT]->(e)
RETURN e, collect(DISTINCT f) AS facts
ORDER BY e.name
"""

GET_NAMESPACE_CONTEXT_USER = """
MATCH (root:Namespace {code: $namespace_code})
MATCH (root)-[:PARENT_OF*0..]->(ns:Namespace)
MATCH (e:Entity)-[:BELONGS_TO]->(ns)
WITH DISTINCT e
OPTIONAL MATCH (uc:UserContext {user_id: $user_id})-[:OWNS_FACT]->(f:Fact)-[:FACT_ABOUT]->(e)
RETURN e, collect(DISTINCT f) AS facts
ORDER BY e.name
"""

COUNT_ENTITIES_IN_NAMESPACE = """
MATCH (root:Namespace {code: $namespace_code})
MATCH (root)-[:PARENT_OF*0..]->(ns:Namespace)
MATCH (e:Entity)-[:BELONGS_TO]->(ns)
RETURN count(DISTINCT e) as entity_count
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
