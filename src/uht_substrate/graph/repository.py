"""Repository for knowledge graph operations."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from uuid6 import uuid7

from ..config.logging import get_logger
from ..uht_client.models import ClassificationResult, Entity, TraitValue
from . import queries
from .connection import Neo4jConnection

logger = get_logger(__name__)


@dataclass
class StoredEntity:
    """Entity as stored in the knowledge graph."""

    uuid: str
    name: str
    hex_code: str
    binary_code: str
    description: Optional[str]
    source: str
    created_at: datetime
    updated_at: datetime


@dataclass
class StoredFact:
    """Fact as stored in the knowledge graph."""

    uuid: str
    subject: str
    predicate: str
    object: str
    confidence: float
    source: str
    created_at: datetime
    category: str = "associative"
    is_custom_predicate: bool = False
    bound: bool = False
    subject_entity_uuid: Optional[str] = None
    object_entity_uuid: Optional[str] = None
    updated_at: Optional[datetime] = None
    _is_duplicate: bool = False  # Transient flag, not persisted


@dataclass
class StoredReasoningTrace:
    """Reasoning trace as stored in the knowledge graph."""

    uuid: str
    query: str
    conclusion: str
    strategy: str
    confidence: float
    created_at: datetime


@dataclass
class StoredNamespace:
    """Namespace as stored in the knowledge graph."""

    uuid: str
    code: str
    name: str
    description: Optional[str]
    is_root: bool
    created_at: datetime
    entity_count: int = 0


class GraphRepository:
    """Repository for knowledge graph CRUD operations."""

    def __init__(self, connection: Neo4jConnection):
        """
        Initialize the repository.

        Args:
            connection: Neo4j connection instance
        """
        self._conn = connection

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def upsert_entity(
        self,
        classification: ClassificationResult | Entity,
        source: str = "uht_factory",
        description: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> StoredEntity:
        """
        Upsert an entity from classification result.

        Args:
            classification: Classification result or Entity
            source: Source of the entity ("uht_factory", "user_defined", "inferred")
            description: Optional description override
            namespace: Optional namespace code (defaults to "global")

        Returns:
            The stored entity
        """
        # Handle both ClassificationResult and Entity
        if isinstance(classification, ClassificationResult):
            uuid = classification.uuid
            name = classification.entity
            hex_code = classification.hex_code
            binary_code = classification.binary
            traits = classification.traits
        else:
            uuid = classification.uuid
            name = classification.name
            hex_code = classification.hex_code
            binary_code = classification.binary or self._hex_to_binary(hex_code)
            traits = classification.traits
            description = description or classification.description

        params = {
            "uuid": uuid,
            "name": name,
            "hex_code": hex_code,
            "binary_code": binary_code,
            "description": description,
            "source": source,
        }

        result = await self._conn.execute_write(queries.UPSERT_ENTITY, params)
        entity_data = result[0]["e"] if result else {}

        logger.debug("Upserted entity", uuid=uuid, name=name)

        # Connect to traits
        for trait in traits:
            if trait.present:
                await self._conn.execute_write(
                    queries.CONNECT_ENTITY_TO_TRAIT,
                    {
                        "entity_uuid": uuid,
                        "bit_position": trait.bit_position,
                        "confidence": trait.confidence,
                        "justification": trait.justification,
                    },
                )

        # Assign to namespace (default to global)
        ns_code = namespace or "global"
        await self.assign_entity_to_namespace(uuid, ns_code, primary=True)

        return StoredEntity(
            uuid=uuid,
            name=name,
            hex_code=hex_code,
            binary_code=binary_code,
            description=description,
            source=source,
            created_at=entity_data.get("created_at", datetime.utcnow()),
            updated_at=entity_data.get("updated_at", datetime.utcnow()),
        )

    async def find_entity_by_name(self, name: str) -> Optional[StoredEntity]:
        """
        Find entity by name (case-insensitive).

        Args:
            name: Entity name to search for

        Returns:
            Entity if found, None otherwise
        """
        result = await self._conn.execute_query(
            queries.FIND_ENTITY_BY_NAME,
            {"name": name},
        )

        if not result:
            return None

        return self._parse_entity(result[0]["e"])

    async def find_entity_by_uuid(self, uuid: str) -> Optional[StoredEntity]:
        """
        Find entity by UUID.

        Args:
            uuid: Entity UUID

        Returns:
            Entity if found, None otherwise
        """
        result = await self._conn.execute_query(
            queries.FIND_ENTITY_BY_UUID,
            {"uuid": uuid},
        )

        if not result:
            return None

        return self._parse_entity(result[0]["e"])

    async def get_classification_by_name(self, name: str) -> Optional["ClassificationResult"]:
        """
        Reconstruct a ClassificationResult from local graph data.

        Returns None if entity not found or has no traits stored.
        """
        from uht_substrate.uht_client.models import ClassificationResult, TraitValue

        result = await self._conn.execute_query(
            queries.FIND_ENTITY_WITH_TRAITS_BY_NAME,
            {"name": name},
        )
        if not result:
            return None

        e = result[0]["e"]
        raw_traits = result[0].get("traits", [])

        # Filter out null entries from OPTIONAL MATCH
        traits = [t for t in raw_traits if t.get("bit_position") is not None]
        if not traits:
            return None

        trait_values = [
            TraitValue(
                bit_position=t["bit_position"],
                name=t.get("name", ""),
                present=True,  # Only present traits are stored as HAS_TRAIT edges
                confidence=t.get("confidence", 1.0),
                justification=t.get("justification"),
            )
            for t in sorted(traits, key=lambda t: t["bit_position"])
        ]

        # Fill in absent traits (bits without HAS_TRAIT edges)
        present_bits = {t.bit_position for t in trait_values}
        for bit in range(1, 33):
            if bit not in present_bits:
                trait_values.append(
                    TraitValue(
                        bit_position=bit,
                        name="",
                        present=False,
                        confidence=0.0,
                        justification=None,
                    )
                )
        trait_values.sort(key=lambda t: t.bit_position)

        return ClassificationResult(
            uuid=e["uuid"],
            name=e["name"],
            hex_code=e["hex_code"],
            binary=e["binary_code"],
            traits=trait_values,
            created_at=e.get("created_at", datetime.utcnow()),
        )

    async def search_entities(self, query: str, limit: int = 100) -> list[StoredEntity]:
        """
        Search entities by name substring.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching entities
        """
        result = await self._conn.execute_query(
            queries.SEARCH_ENTITIES,
            {"query": query, "limit": limit},
        )

        return [self._parse_entity(r["e"]) for r in result]

    async def list_entities(
        self,
        name_contains: Optional[str] = None,
        hex_pattern: Optional[str] = None,
        namespace: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[StoredEntity], int]:
        """
        List entities with optional filters, returning results and total count.

        Args:
            name_contains: Filter by name substring (case-insensitive)
            hex_pattern: Filter by hex code prefix
            namespace: Filter by namespace (includes descendants)
            limit: Maximum results per page
            offset: Number of results to skip (for pagination)

        Returns:
            Tuple of (list of matching entities, total count matching filters)
        """
        # If namespace filter provided, use the namespace-specific method
        if namespace:
            entities = await self.list_entities_in_namespace(
                namespace_code=namespace,
                name_contains=name_contains,
                hex_pattern=hex_pattern,
                limit=limit,
            )
            return entities, len(entities)

        # Build dynamic query
        conditions = []
        params: dict[str, object] = {"limit": limit, "offset": offset}

        if name_contains:
            conditions.append("toLower(e.name) CONTAINS toLower($name_filter)")
            params["name_filter"] = name_contains

        if hex_pattern:
            conditions.append("e.hex_code STARTS WITH $hex_filter")
            params["hex_filter"] = hex_pattern.upper()

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            MATCH (e:Entity)
            {where_clause}
            RETURN e
            ORDER BY e.name ASC
            SKIP $offset
            LIMIT $limit
        """

        count_query = f"""
            MATCH (e:Entity)
            {where_clause}
            RETURN count(e) AS total
        """

        result = await self._conn.execute_query(query, params)
        count_result = await self._conn.execute_query(count_query, params)
        total = count_result[0]["total"] if count_result else 0

        return [self._parse_entity(r["e"]) for r in result], total

    async def find_similar_entities(
        self,
        uuid: str,
        min_shared: int = 24,
        limit: int = 10,
    ) -> list[tuple[StoredEntity, int, int]]:
        """
        Find entities similar to given entity by trait overlap.

        Args:
            uuid: Entity UUID
            min_shared: Minimum shared traits
            limit: Maximum results

        Returns:
            List of (entity, shared_traits, hamming_distance) tuples
        """
        result = await self._conn.execute_query(
            queries.FIND_SIMILAR_ENTITIES,
            {"uuid": uuid, "min_shared": min_shared, "limit": limit},
        )

        return [
            (
                self._parse_entity(r["entity"]),
                r["shared_traits"],
                r["hamming_distance"],
            )
            for r in result
        ]

    async def delete_entity(self, uuid: str) -> bool:
        """
        Delete an entity and all its relationships.

        Args:
            uuid: Entity UUID

        Returns:
            True if entity was deleted
        """
        await self._conn.execute_write(queries.DELETE_ENTITY, {"uuid": uuid})
        logger.info("Deleted entity", uuid=uuid)
        return True

    # =========================================================================
    # Relationship Operations
    # =========================================================================

    async def create_similar_to_relationship(
        self,
        source_uuid: str,
        target_uuid: str,
        similarity_score: float,
        shared_traits: list[int],
    ) -> None:
        """
        Create SIMILAR_TO relationship between entities.

        Args:
            source_uuid: Source entity UUID
            target_uuid: Target entity UUID
            similarity_score: Similarity score (0-1)
            shared_traits: List of shared trait bit positions
        """
        await self._conn.execute_write(
            queries.CREATE_SIMILAR_TO_RELATIONSHIP,
            {
                "source_uuid": source_uuid,
                "target_uuid": target_uuid,
                "similarity_score": similarity_score,
                "shared_traits": shared_traits,
            },
        )
        logger.debug(
            "Created SIMILAR_TO relationship",
            source=source_uuid,
            target=target_uuid,
            score=similarity_score,
        )

    async def create_is_a_relationship(
        self,
        source_uuid: str,
        target_uuid: str,
        confidence: float = 1.0,
    ) -> None:
        """
        Create IS_A relationship between entities.

        Args:
            source_uuid: Source entity UUID (child)
            target_uuid: Target entity UUID (parent)
            confidence: Confidence in the relationship
        """
        await self._conn.execute_write(
            queries.CREATE_IS_A_RELATIONSHIP,
            {
                "source_uuid": source_uuid,
                "target_uuid": target_uuid,
                "properties": {"confidence": confidence},
            },
        )

    async def create_related_to_relationship(
        self,
        source_uuid: str,
        target_uuid: str,
        relation_type: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Create RELATED_TO relationship between entities.

        Args:
            source_uuid: Source entity UUID
            target_uuid: Target entity UUID
            relation_type: Type of relation
            properties: Additional relationship properties
        """
        props = properties or {}
        props["relation_type"] = relation_type

        await self._conn.execute_write(
            queries.CREATE_RELATED_TO_RELATIONSHIP,
            {
                "source_uuid": source_uuid,
                "target_uuid": target_uuid,
                "properties": props,
            },
        )

    async def get_entity_relationships(
        self,
        uuid: str,
    ) -> list[dict[str, Any]]:
        """
        Get all relationships for an entity.

        Args:
            uuid: Entity UUID

        Returns:
            List of relationship info dicts
        """
        return await self._conn.execute_query(
            queries.GET_ENTITY_RELATIONSHIPS,
            {"uuid": uuid},
        )

    # =========================================================================
    # Fact Operations
    # =========================================================================

    async def store_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "asserted",
        user_id: Optional[str] = None,
        trace_uuid: Optional[str] = None,
    ) -> StoredFact:
        """
        Store a fact in the knowledge graph.

        Predicates are normalized to uppercase and categorized against the
        predicate taxonomy. Unknown predicates are stored under 'associative'
        with is_custom_predicate=True. The 'computed' category is reserved
        for system operations and rejected when source is 'asserted'.

        Args:
            subject: Subject of the fact
            predicate: Predicate/relationship
            obj: Object of the fact
            confidence: Confidence level (0-1)
            source: Source of the fact ('asserted', 'computed', 'inferred')
            user_id: Optional user ID to link fact to
            trace_uuid: Optional reasoning trace UUID

        Returns:
            The stored fact

        Raises:
            ValueError: If a computed predicate is used with source='asserted'
        """
        from .schema import PredicateTaxonomy

        normalized_predicate = predicate.strip().upper().replace(" ", "_")
        category, is_custom = PredicateTaxonomy.categorize(normalized_predicate)

        # Check for duplicate (same subject + predicate + object)
        existing = await self._conn.execute_query(
            queries.FIND_DUPLICATE_FACT,
            {
                "subject": subject,
                "predicate": normalized_predicate,
                "object": obj,
            },
        )
        if existing:
            fact = self._parse_fact(existing[0]["f"])
            fact.is_custom_predicate = is_custom  # Ensure current taxonomy
            fact._is_duplicate = True
            logger.debug(
                "Duplicate fact found",
                uuid=fact.uuid,
                subject=subject,
                predicate=normalized_predicate,
            )
            return fact

        if source == "asserted" and category == "computed":
            raise ValueError(
                f"Predicate '{predicate}' is in the 'computed' category "
                "and cannot be set by users. Use a different predicate."
            )

        fact_uuid = str(uuid7())

        params = {
            "uuid": fact_uuid,
            "subject": subject,
            "predicate": normalized_predicate,
            "object": obj,
            "confidence": confidence,
            "source": source,
            "category": category,
            "is_custom_predicate": is_custom,
        }

        result = await self._conn.execute_write(queries.CREATE_FACT, params)
        fact_data = result[0]["f"] if result else {}

        # Link to user if provided
        if user_id:
            await self._conn.execute_write(
                queries.UPSERT_USER_CONTEXT,
                {"user_id": user_id},
            )
            await self._conn.execute_write(
                queries.LINK_FACT_TO_USER,
                {"user_id": user_id, "fact_uuid": fact_uuid},
            )

        # Link to reasoning trace if provided
        if trace_uuid:
            await self._conn.execute_write(
                queries.LINK_TRACE_TO_FACT,
                {"trace_uuid": trace_uuid, "fact_uuid": fact_uuid},
            )

        # Attempt entity binding
        subj_uuid, obj_uuid = await self._try_bind_fact(
            fact_uuid, subject, obj, normalized_predicate, category,
            confidence, source, user_id,
        )

        logger.debug(
            "Stored fact",
            uuid=fact_uuid,
            subject=subject,
            predicate=normalized_predicate,
            category=category,
            bound=bool(subj_uuid and obj_uuid),
        )

        return StoredFact(
            uuid=fact_uuid,
            subject=subject,
            predicate=normalized_predicate,
            object=obj,
            confidence=confidence,
            source=source,
            created_at=fact_data.get("created_at", datetime.utcnow()),
            category=category,
            is_custom_predicate=is_custom,
            bound=bool(subj_uuid and obj_uuid),
            subject_entity_uuid=subj_uuid,
            object_entity_uuid=obj_uuid,
        )

    async def upsert_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "asserted",
        user_id: Optional[str] = None,
    ) -> tuple[StoredFact, bool]:
        """
        Upsert a fact: match on (subject, predicate, user_id).

        If a fact with that combination already exists, update its object
        value. If not, create a new fact.

        Args:
            subject: Subject of the fact
            predicate: Predicate/relationship
            obj: Object of the fact
            confidence: Confidence level (0-1)
            source: Source of the fact
            user_id: Optional user ID to scope the upsert

        Returns:
            Tuple of (StoredFact, was_created: bool)

        Raises:
            ValueError: If a computed predicate is used with source='asserted'
        """
        from .schema import PredicateTaxonomy

        normalized_predicate = predicate.strip().upper().replace(" ", "_")
        category, is_custom = PredicateTaxonomy.categorize(normalized_predicate)

        if source == "asserted" and category == "computed":
            raise ValueError(
                f"Predicate '{predicate}' is in the 'computed' category "
                "and cannot be set by users. Use a different predicate."
            )

        # Look for existing fact matching (subject, predicate, user_id)
        if user_id:
            existing = await self._conn.execute_query(
                queries.FIND_FACT_FOR_UPSERT,
                {
                    "user_id": user_id,
                    "subject": subject,
                    "predicate": normalized_predicate,
                },
            )
        else:
            existing = await self._conn.execute_query(
                queries.FIND_FACT_FOR_UPSERT_GLOBAL,
                {
                    "subject": subject,
                    "predicate": normalized_predicate,
                },
            )

        if existing:
            # Update the existing fact's object value
            existing_fact = self._parse_fact(existing[0]["f"])
            updated = await self.update_fact(
                uuid=existing_fact.uuid,
                obj=obj,
                confidence=confidence,
            )
            if updated:
                return updated, False
            # Fallback: if update failed somehow, return existing
            return existing_fact, False

        # No existing fact — create new one
        fact = await self.store_fact(
            subject=subject,
            predicate=predicate,
            obj=obj,
            confidence=confidence,
            source=source,
            user_id=user_id,
        )
        return fact, True

    async def get_facts_by_subject(
        self,
        subject: str,
        limit: int = 100,
    ) -> list[StoredFact]:
        """
        Get facts by subject.

        Args:
            subject: Subject to search for
            limit: Maximum results

        Returns:
            List of matching facts
        """
        result = await self._conn.execute_query(
            queries.FIND_FACTS_BY_SUBJECT,
            {"subject": subject, "limit": limit},
        )

        return [self._parse_fact(r["f"]) for r in result]

    async def get_user_facts(
        self,
        user_id: str,
        limit: int = 100,
    ) -> list[StoredFact]:
        """
        Get all facts owned by a user.

        Args:
            user_id: User ID
            limit: Maximum results

        Returns:
            List of user's facts
        """
        result = await self._conn.execute_query(
            queries.GET_USER_FACTS,
            {"user_id": user_id, "limit": limit},
        )

        return [self._parse_fact(r["f"]) for r in result]

    async def query_facts(
        self,
        subject: Optional[str] = None,
        object_value: Optional[str] = None,
        predicate: Optional[str] = None,
        category: Optional[str] = None,
        source: Optional[str] = None,
        user_id: Optional[str] = None,
        namespace: Optional[str] = None,
        limit: int = 50,
    ) -> list[StoredFact]:
        """
        Query facts with flexible filters.

        Args:
            subject: Filter by subject (case-insensitive)
            object_value: Filter by object (case-insensitive)
            predicate: Filter by exact predicate
            category: Filter by predicate category
            source: Filter by source type
            user_id: Scope to facts owned by this user
            namespace: Scope to facts whose subject entity belongs
                       to this namespace or its descendants
            limit: Maximum results

        Returns:
            List of matching facts
        """
        # Normalize predicate to uppercase if provided
        if predicate:
            predicate = predicate.strip().upper().replace(" ", "_")

        params = {
            "subject": subject,
            "object": object_value,
            "predicate": predicate,
            "category": category,
            "source": source,
            "limit": limit,
        }

        if namespace:
            if user_id:
                params["user_id"] = user_id
                params["namespace_code"] = namespace
                result = await self._conn.execute_query(
                    queries.QUERY_USER_FACTS_IN_NAMESPACE,
                    params,
                )
            else:
                params["namespace_code"] = namespace
                result = await self._conn.execute_query(
                    queries.QUERY_FACTS_IN_NAMESPACE,
                    params,
                )
        elif user_id:
            params["user_id"] = user_id
            result = await self._conn.execute_query(
                queries.QUERY_USER_FACTS,
                params,
            )
        else:
            result = await self._conn.execute_query(
                queries.QUERY_FACTS,
                params,
            )

        return [self._parse_fact(r["f"]) for r in result]

    async def update_fact(
        self,
        uuid: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> Optional[StoredFact]:
        """
        Update a fact's fields. Only provided (non-None) fields are updated.

        Re-validates predicate against taxonomy if changed. Re-attempts
        entity binding if subject or object changes.

        Args:
            uuid: Fact UUID
            subject: New subject (None to keep current)
            predicate: New predicate (None to keep current)
            obj: New object (None to keep current)
            confidence: New confidence (None to keep current)

        Returns:
            Updated fact, or None if not found
        """
        from .schema import PredicateTaxonomy

        category = None
        is_custom = None
        normalized_predicate = None

        if predicate:
            normalized_predicate = predicate.strip().upper().replace(" ", "_")
            category, is_custom = PredicateTaxonomy.categorize(normalized_predicate)
            if category == "computed":
                raise ValueError(
                    f"Predicate '{predicate}' is in the 'computed' category "
                    "and cannot be set by users."
                )

        result = await self._conn.execute_write(
            queries.UPDATE_FACT,
            {
                "uuid": uuid,
                "subject": subject,
                "predicate": normalized_predicate,
                "object": obj,
                "confidence": confidence,
                "category": category,
                "is_custom_predicate": is_custom,
            },
        )

        if not result:
            return None

        fact = self._parse_fact(result[0]["f"])

        # Re-attempt binding if subject or object changed
        if subject or obj:
            subj_uuid, obj_uuid = await self._try_bind_fact(
                uuid, fact.subject, fact.object, fact.predicate,
                fact.category, fact.confidence, fact.source, None,
            )
            fact.bound = bool(subj_uuid and obj_uuid)
            fact.subject_entity_uuid = subj_uuid
            fact.object_entity_uuid = obj_uuid

        return fact

    async def delete_fact(self, uuid: str) -> bool:
        """
        Delete a fact and all its relationships.

        Removes the Fact node, OWNS_FACT edges, FACT_ABOUT/FACT_REFERENCES
        edges, and any RELATED_TO entity edges linked by fact_uuid.

        Args:
            uuid: Fact UUID

        Returns:
            True if deleted, False if not found
        """
        # First remove any entity relationship created from this fact
        await self._conn.execute_write(
            queries.DELETE_ENTITY_RELATIONSHIP_BY_FACT,
            {"fact_uuid": uuid},
        )

        # Then delete the fact node (DETACH DELETE removes remaining edges)
        result = await self._conn.execute_write(
            queries.DELETE_FACT,
            {"uuid": uuid},
        )
        deleted = bool(result and result[0].get("deleted_uuid"))
        if deleted:
            logger.debug("Deleted fact", uuid=uuid)
        return deleted

    async def bind_pending_facts(self, limit: int = 100) -> dict[str, int]:
        """
        Attempt to bind unbound facts to classified entity nodes.

        Scans facts where bound=false and tries to match subject/object
        to existing Entity nodes by name.

        Args:
            limit: Maximum facts to process

        Returns:
            Counts: {"checked": N, "newly_bound": M}
        """
        result = await self._conn.execute_query(
            queries.FIND_UNBOUND_FACTS,
            {"limit": limit},
        )

        checked = 0
        newly_bound = 0
        for row in result:
            f = row["f"]
            subj_uuid, obj_uuid = await self._try_bind_fact(
                f["uuid"], f["subject"], f["object"],
                f.get("predicate", "RELATED_TO"),
                f.get("category", "associative"),
                f.get("confidence", 1.0),
                f.get("source", "asserted"),
                None,
            )
            checked += 1
            if subj_uuid and obj_uuid:
                newly_bound += 1

        logger.info("Binding pass complete", checked=checked, newly_bound=newly_bound)
        return {"checked": checked, "newly_bound": newly_bound}

    async def bind_pending_facts_for_entity(self, entity_name: str) -> dict[str, int]:
        """
        Attempt to bind unbound facts that reference a specific entity.

        Called after classify_entity to retroactively bind facts that were
        stored before the entity was classified.

        Args:
            entity_name: Name of the newly classified entity

        Returns:
            Counts: {"checked": N, "newly_bound": M}
        """
        result = await self._conn.execute_query(
            queries.FIND_UNBOUND_FACTS_FOR_ENTITY,
            {"entity_name": entity_name},
        )

        checked = 0
        newly_bound = 0
        for row in result:
            f = row["f"]
            subj_uuid, obj_uuid = await self._try_bind_fact(
                f["uuid"], f["subject"], f["object"],
                f.get("predicate", "RELATED_TO"),
                f.get("category", "associative"),
                f.get("confidence", 1.0),
                f.get("source", "asserted"),
                None,
            )
            checked += 1
            if subj_uuid and obj_uuid:
                newly_bound += 1

        if newly_bound:
            logger.info(
                "Retroactive binding complete",
                entity=entity_name,
                checked=checked,
                newly_bound=newly_bound,
            )
        return {"checked": checked, "newly_bound": newly_bound}

    async def get_user_facts_grouped(
        self,
        user_id: str,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Get user facts grouped by category with summary stats.

        Args:
            user_id: User ID
            limit: Maximum facts to return

        Returns:
            Dict with 'facts_by_category' and 'summary'
        """
        result = await self._conn.execute_query(
            queries.GET_USER_FACTS_WITH_BINDING,
            {"user_id": user_id, "limit": limit},
        )

        by_category: dict[str, list[dict[str, Any]]] = {}
        total = 0
        bound_count = 0
        unbound_count = 0
        by_source: dict[str, int] = {}

        for row in result:
            f = row["f"]
            fact_dict = {
                "uuid": f["uuid"],
                "subject": f["subject"],
                "predicate": f["predicate"],
                "object": f["object"],
                "confidence": f.get("confidence", 1.0),
                "source": f.get("source", "unknown"),
                "category": f.get("category", "associative"),
                "is_custom_predicate": f.get("is_custom_predicate", False),
                "bound": f.get("bound", False),
                "subject_entity": row.get("subject_entity_name"),
                "object_entity": row.get("object_entity_name"),
                "created_at": str(f.get("created_at", "")),
            }

            cat = fact_dict["category"]
            by_category.setdefault(cat, []).append(fact_dict)

            total += 1
            if fact_dict["bound"]:
                bound_count += 1
            else:
                unbound_count += 1
            src = fact_dict["source"]
            by_source[src] = by_source.get(src, 0) + 1

        return {
            "facts_by_category": by_category,
            "summary": {
                "total_facts": total,
                "bound": bound_count,
                "unbound": unbound_count,
                "by_source": by_source,
            },
        }

    async def _try_bind_fact(
        self,
        fact_uuid: str,
        subject: str,
        obj: str,
        predicate: str,
        category: str,
        confidence: float,
        source: str,
        user_id: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Attempt to bind a fact to Entity nodes.

        If both subject and object match existing Entity nodes, creates
        a RELATED_TO relationship edge between them and marks the fact
        as bound.

        Args:
            fact_uuid: UUID of the fact to bind
            subject: Subject string to match against Entity names
            obj: Object string to match against Entity names
            predicate: The fact's predicate
            category: The fact's category
            confidence: The fact's confidence
            source: The fact's source
            user_id: Optional user ID for the relationship

        Returns:
            (subject_entity_uuid, object_entity_uuid) -- either can be None
        """
        subj_uuid = None
        obj_uuid = None

        # Try to bind subject
        result = await self._conn.execute_write(
            queries.BIND_FACT_TO_SUBJECT_ENTITY,
            {"fact_uuid": fact_uuid, "subject": subject},
        )
        if result:
            subj_uuid = result[0]["entity_uuid"]

        # Try to bind object
        result = await self._conn.execute_write(
            queries.BIND_FACT_TO_OBJECT_ENTITY,
            {"fact_uuid": fact_uuid, "object": obj},
        )
        if result:
            obj_uuid = result[0]["entity_uuid"]

        # If both bound, create typed entity relationship and mark fact
        if subj_uuid and obj_uuid:
            await self._conn.execute_write(
                queries.CREATE_ENTITY_RELATIONSHIP_FROM_FACT,
                {
                    "source_uuid": subj_uuid,
                    "target_uuid": obj_uuid,
                    "predicate": predicate,
                    "fact_uuid": fact_uuid,
                    "category": category,
                    "source": source,
                    "user_id": user_id or "",
                    "confidence": confidence,
                },
            )

            await self._conn.execute_write(
                queries.MARK_FACT_BOUND,
                {
                    "uuid": fact_uuid,
                    "subject_entity_uuid": subj_uuid,
                    "object_entity_uuid": obj_uuid,
                },
            )

        return subj_uuid, obj_uuid

    # =========================================================================
    # User Context Operations
    # =========================================================================

    async def store_preference(
        self,
        user_id: str,
        key: str,
        value: str,
    ) -> None:
        """
        Store a user preference.

        Args:
            user_id: User ID
            key: Preference key
            value: Preference value
        """
        await self._conn.execute_write(
            queries.CREATE_PREFERENCE,
            {
                "user_id": user_id,
                "uuid": str(uuid7()),
                "key": key,
                "value": value,
            },
        )
        logger.debug("Stored preference", user_id=user_id, key=key)

    async def get_user_preferences(self, user_id: str) -> dict[str, str]:
        """
        Get all preferences for a user.

        Args:
            user_id: User ID

        Returns:
            Dict mapping preference keys to values
        """
        result = await self._conn.execute_query(
            queries.GET_USER_PREFERENCES,
            {"user_id": user_id},
        )

        return {r["key"]: r["value"] for r in result}

    async def mark_user_interested_in(self, user_id: str, entity_uuid: str) -> None:
        """
        Record that user is interested in an entity.

        Args:
            user_id: User ID
            entity_uuid: Entity UUID
        """
        await self._conn.execute_write(
            queries.UPSERT_USER_CONTEXT,
            {"user_id": user_id},
        )
        await self._conn.execute_write(
            queries.MARK_USER_INTERESTED_IN,
            {"user_id": user_id, "entity_uuid": entity_uuid},
        )

    # =========================================================================
    # Reasoning Trace Operations
    # =========================================================================

    async def create_reasoning_trace(
        self,
        query: str,
        conclusion: str,
        strategy: str,
        confidence: float,
        entity_uuids: Optional[list[str]] = None,
        axiom_uuids: Optional[list[str]] = None,
    ) -> str:
        """
        Create a reasoning trace.

        Args:
            query: The original query
            conclusion: The conclusion reached
            strategy: The reasoning strategy used
            confidence: Confidence in the conclusion
            entity_uuids: UUIDs of entities used
            axiom_uuids: UUIDs of axioms applied

        Returns:
            The trace UUID
        """
        trace_uuid = str(uuid7())

        await self._conn.execute_write(
            queries.CREATE_REASONING_TRACE,
            {
                "uuid": trace_uuid,
                "query": query,
                "conclusion": conclusion,
                "strategy": strategy,
                "confidence": confidence,
            },
        )

        # Link to entities
        for entity_uuid in entity_uuids or []:
            await self._conn.execute_write(
                queries.LINK_TRACE_TO_ENTITY,
                {"trace_uuid": trace_uuid, "entity_uuid": entity_uuid},
            )

        # Link to axioms
        for axiom_uuid in axiom_uuids or []:
            await self._conn.execute_write(
                queries.LINK_TRACE_TO_AXIOM,
                {"trace_uuid": trace_uuid, "axiom_uuid": axiom_uuid},
            )

        logger.debug("Created reasoning trace", uuid=trace_uuid, strategy=strategy)

        return trace_uuid

    async def get_recent_traces(
        self,
        hours: int = 24,
        limit: int = 10,
    ) -> list[StoredReasoningTrace]:
        """
        Get recent reasoning traces.

        Args:
            hours: How far back to look
            limit: Maximum results

        Returns:
            List of recent traces
        """
        result = await self._conn.execute_query(
            queries.GET_RECENT_TRACES,
            {"hours": hours, "limit": limit},
        )

        return [self._parse_reasoning_trace(r["rt"]) for r in result]

    async def get_trace_details(self, uuid: str) -> Optional[dict[str, Any]]:
        """
        Get full details of a reasoning trace.

        Args:
            uuid: Trace UUID

        Returns:
            Trace with linked entities, axioms, and facts
        """
        result = await self._conn.execute_query(
            queries.GET_TRACE_DETAILS,
            {"uuid": uuid},
        )

        if not result:
            return None

        return result[0]

    # =========================================================================
    # Axiom Operations
    # =========================================================================

    async def upsert_axiom(
        self,
        uuid: str,
        trait_bit: int,
        name: str,
        statement: str,
        axiom_type: str,
        prop: str,
        confidence: float = 1.0,
    ) -> None:
        """
        Upsert an axiom.

        Args:
            uuid: Axiom UUID
            trait_bit: Associated trait bit position
            name: Axiom name
            statement: The axiom statement
            axiom_type: Type (necessary, typical, possible)
            prop: Property the axiom derives
            confidence: Confidence level
        """
        await self._conn.execute_write(
            queries.UPSERT_AXIOM,
            {
                "uuid": uuid,
                "trait_bit": trait_bit,
                "name": name,
                "statement": statement,
                "axiom_type": axiom_type,
                "property": prop,
                "confidence": confidence,
            },
        )

    async def get_axioms_for_trait(self, bit_position: int) -> list[dict[str, Any]]:
        """
        Get all axioms for a trait.

        Args:
            bit_position: Trait bit position (1-32)

        Returns:
            List of axiom data
        """
        result = await self._conn.execute_query(
            queries.GET_AXIOMS_FOR_TRAIT,
            {"bit_position": bit_position},
        )

        return [r["a"] for r in result]

    async def get_all_axioms(self) -> list[dict[str, Any]]:
        """
        Get all axioms.

        Returns:
            List of all axiom data
        """
        result = await self._conn.execute_query(queries.GET_ALL_AXIOMS)
        return [r["a"] for r in result]

    # =========================================================================
    # Namespace Operations
    # =========================================================================

    async def create_namespace(
        self,
        code: str,
        name: str,
        description: Optional[str] = None,
    ) -> StoredNamespace:
        """
        Create a namespace, auto-creating parent namespaces if needed.

        Hierarchical namespaces use colon separators (e.g., "SE:aerospace:propulsion").
        Parent namespaces are created automatically if they don't exist.

        Args:
            code: Unique namespace code (e.g., "SE", "SE:aerospace")
            name: Human-readable name
            description: Optional description

        Returns:
            The created namespace
        """
        # Parse hierarchy from code
        parts = code.split(":")
        parent_code = None

        # Create parent namespaces if needed
        for i in range(len(parts) - 1):
            ancestor_code = ":".join(parts[: i + 1])
            existing = await self.get_namespace(ancestor_code)
            if not existing:
                # Create parent with generated name
                await self._conn.execute_write(
                    queries.CREATE_NAMESPACE,
                    {
                        "code": ancestor_code,
                        "name": parts[i].upper(),
                        "description": None,
                        "parent_code": parent_code,
                    },
                )
                logger.debug("Auto-created parent namespace", code=ancestor_code)
            parent_code = ancestor_code

        # Create the actual namespace
        result = await self._conn.execute_write(
            queries.CREATE_NAMESPACE,
            {
                "code": code,
                "name": name,
                "description": description,
                "parent_code": parent_code,
            },
        )

        ns_data = result[0]["n"] if result else {}
        logger.info("Created namespace", code=code, parent=parent_code)

        return StoredNamespace(
            uuid=ns_data.get("uuid", ""),
            code=code,
            name=name,
            description=description,
            is_root=parent_code is None,
            created_at=ns_data.get("created_at", datetime.utcnow()),
        )

    async def get_namespace(self, code: str) -> Optional[StoredNamespace]:
        """
        Get a namespace by code.

        Args:
            code: Namespace code (e.g., "SE:aerospace")

        Returns:
            Namespace if found, None otherwise
        """
        result = await self._conn.execute_query(
            queries.FIND_NAMESPACE_BY_CODE,
            {"code": code},
        )

        if not result:
            return None

        return self._parse_namespace(result[0]["n"])

    async def list_namespaces(
        self,
        parent_code: Optional[str] = None,
        include_descendants: bool = False,
    ) -> list[StoredNamespace]:
        """
        List namespaces.

        Args:
            parent_code: List children of this namespace (None for root namespaces)
            include_descendants: If True, include entire subtree

        Returns:
            List of namespaces
        """
        if parent_code is None:
            # List root namespaces
            result = await self._conn.execute_query(queries.LIST_ROOT_NAMESPACES)
        elif include_descendants:
            # List entire subtree
            result = await self._conn.execute_query(
                queries.LIST_NAMESPACE_DESCENDANTS,
                {"code": parent_code},
            )
        else:
            # List direct children only
            result = await self._conn.execute_query(
                queries.LIST_NAMESPACE_CHILDREN,
                {"parent_code": parent_code},
            )

        return [self._parse_namespace(r["n"]) for r in result]

    async def delete_namespace(
        self,
        code: str,
        cascade: bool = False,
    ) -> bool:
        """
        Delete a namespace.

        Args:
            code: Namespace code
            cascade: If True, also delete child namespaces

        Returns:
            True if deleted
        """
        if cascade:
            await self._conn.execute_write(
                queries.DELETE_NAMESPACE_CASCADE,
                {"code": code},
            )
        else:
            await self._conn.execute_write(
                queries.DELETE_NAMESPACE,
                {"code": code},
            )

        logger.info("Deleted namespace", code=code, cascade=cascade)
        return True

    async def assign_entity_to_namespace(
        self,
        entity_uuid: str,
        namespace_code: str,
        primary: bool = True,
    ) -> None:
        """
        Assign an entity to a namespace.

        Args:
            entity_uuid: Entity UUID
            namespace_code: Namespace code
            primary: Whether this is the entity's primary namespace
        """
        await self._conn.execute_write(
            queries.ASSIGN_ENTITY_TO_NAMESPACE,
            {
                "entity_uuid": entity_uuid,
                "namespace_code": namespace_code,
                "primary": primary,
            },
        )
        logger.debug(
            "Assigned entity to namespace",
            entity_uuid=entity_uuid,
            namespace=namespace_code,
            primary=primary,
        )

    async def remove_entity_from_namespace(
        self,
        entity_uuid: str,
        namespace_code: str,
    ) -> None:
        """
        Remove an entity from a namespace.

        Args:
            entity_uuid: Entity UUID
            namespace_code: Namespace code
        """
        await self._conn.execute_write(
            queries.REMOVE_ENTITY_FROM_NAMESPACE,
            {
                "entity_uuid": entity_uuid,
                "namespace_code": namespace_code,
            },
        )

    async def get_entity_namespaces(self, entity_uuid: str) -> list[StoredNamespace]:
        """
        Get all namespaces an entity belongs to.

        Args:
            entity_uuid: Entity UUID

        Returns:
            List of namespaces (primary first)
        """
        result = await self._conn.execute_query(
            queries.GET_ENTITY_NAMESPACES,
            {"entity_uuid": entity_uuid},
        )

        return [self._parse_namespace(r["n"]) for r in result]

    async def list_entities_in_namespace(
        self,
        namespace_code: str,
        name_contains: Optional[str] = None,
        hex_pattern: Optional[str] = None,
        limit: int = 100,
    ) -> list[StoredEntity]:
        """
        List entities in a namespace (including descendants).

        Args:
            namespace_code: Namespace code
            name_contains: Optional name filter
            hex_pattern: Optional hex prefix filter
            limit: Maximum results

        Returns:
            List of entities in the namespace subtree
        """
        if name_contains or hex_pattern:
            result = await self._conn.execute_query(
                queries.LIST_ENTITIES_IN_NAMESPACE_FILTERED,
                {
                    "namespace_code": namespace_code,
                    "name_filter": name_contains,
                    "hex_filter": hex_pattern.upper() if hex_pattern else None,
                    "limit": limit,
                },
            )
        else:
            result = await self._conn.execute_query(
                queries.LIST_ENTITIES_IN_NAMESPACE,
                {"namespace_code": namespace_code, "limit": limit},
            )

        return [self._parse_entity(r["e"]) for r in result]

    async def count_entities_in_namespace(self, namespace_code: str) -> int:
        """
        Count entities in a namespace (including descendants).

        Args:
            namespace_code: Namespace code

        Returns:
            Entity count
        """
        result = await self._conn.execute_query(
            queries.COUNT_ENTITIES_IN_NAMESPACE,
            {"namespace_code": namespace_code},
        )
        return result[0]["entity_count"] if result else 0

    async def get_namespace_context(
        self,
        namespace_code: str,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Get all entities and their facts under a namespace subtree.

        Returns entities with hex codes and all facts whose subject is
        an entity in the namespace (or descendants).

        Args:
            namespace_code: Root namespace code
            user_id: Optional user ID to scope facts

        Returns:
            Dict with 'entities' and 'facts' lists
        """
        if user_id:
            result = await self._conn.execute_query(
                queries.GET_NAMESPACE_CONTEXT_USER,
                {"namespace_code": namespace_code, "user_id": user_id},
            )
        else:
            result = await self._conn.execute_query(
                queries.GET_NAMESPACE_CONTEXT,
                {"namespace_code": namespace_code},
            )

        entities = []
        all_facts = []
        seen_fact_uuids: set[str] = set()

        for row in result:
            entity = self._parse_entity(row["e"])
            entities.append(entity)

            for f_data in row.get("facts", []):
                if f_data and f_data.get("uuid"):
                    if f_data["uuid"] not in seen_fact_uuids:
                        seen_fact_uuids.add(f_data["uuid"])
                        all_facts.append(self._parse_fact(f_data))

        return {
            "entities": entities,
            "facts": all_facts,
        }

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_statistics(self) -> dict[str, Any]:
        """
        Get knowledge graph statistics.

        Returns:
            Statistics about nodes and relationships
        """
        node_stats = await self._conn.execute_query(queries.GET_GRAPH_STATISTICS)
        rel_stats = await self._conn.execute_query(queries.GET_RELATIONSHIP_STATISTICS)

        return {
            "nodes": {r["label"]: r["count"] for r in node_stats},
            "relationships": {r["relationship_type"]: r["count"] for r in rel_stats},
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_entity(self, data: dict[str, Any]) -> StoredEntity:
        """Parse Neo4j node data into StoredEntity."""
        return StoredEntity(
            uuid=data["uuid"],
            name=data["name"],
            hex_code=data["hex_code"],
            binary_code=data.get("binary_code", ""),
            description=data.get("description"),
            source=data.get("source", "unknown"),
            created_at=data.get("created_at", datetime.utcnow()),
            updated_at=data.get("updated_at", datetime.utcnow()),
        )

    def _parse_fact(self, data: dict[str, Any]) -> StoredFact:
        """Parse Neo4j node data into StoredFact."""
        return StoredFact(
            uuid=data["uuid"],
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "unknown"),
            created_at=data.get("created_at", datetime.utcnow()),
            category=data.get("category", "associative"),
            is_custom_predicate=data.get("is_custom_predicate", False),
            bound=data.get("bound", False),
            subject_entity_uuid=data.get("subject_entity_uuid"),
            object_entity_uuid=data.get("object_entity_uuid"),
            updated_at=data.get("updated_at"),
        )

    def _parse_reasoning_trace(self, data: dict[str, Any]) -> StoredReasoningTrace:
        """Parse Neo4j node data into StoredReasoningTrace."""
        return StoredReasoningTrace(
            uuid=data["uuid"],
            query=data["query"],
            conclusion=data["conclusion"],
            strategy=data["strategy"],
            confidence=data.get("confidence", 0.0),
            created_at=data.get("created_at", datetime.utcnow()),
        )

    def _parse_namespace(self, data: dict[str, Any]) -> StoredNamespace:
        """Parse Neo4j node data into StoredNamespace."""
        return StoredNamespace(
            uuid=data.get("uuid", ""),
            code=data["code"],
            name=data["name"],
            description=data.get("description"),
            is_root=data.get("is_root", False),
            created_at=data.get("created_at", datetime.utcnow()),
            entity_count=data.get("entity_count", 0),
        )

    def _hex_to_binary(self, hex_code: str) -> str:
        """Convert 8-char hex code to 32-bit binary string."""
        return bin(int(hex_code, 16))[2:].zfill(32)
