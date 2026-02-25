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


@dataclass
class StoredReasoningTrace:
    """Reasoning trace as stored in the knowledge graph."""

    uuid: str
    query: str
    conclusion: str
    strategy: str
    confidence: float
    created_at: datetime


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
    ) -> StoredEntity:
        """
        Upsert an entity from classification result.

        Args:
            classification: Classification result or Entity
            source: Source of the entity ("uht_factory", "user_defined", "inferred")
            description: Optional description override

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
                    },
                )

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
        source: str = "user",
        user_id: Optional[str] = None,
        trace_uuid: Optional[str] = None,
    ) -> StoredFact:
        """
        Store a fact in the knowledge graph.

        Args:
            subject: Subject of the fact
            predicate: Predicate/relationship
            obj: Object of the fact
            confidence: Confidence level (0-1)
            source: Source of the fact
            user_id: Optional user ID to link fact to
            trace_uuid: Optional reasoning trace UUID

        Returns:
            The stored fact
        """
        fact_uuid = str(uuid7())

        params = {
            "uuid": fact_uuid,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "source": source,
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

        logger.debug("Stored fact", uuid=fact_uuid, subject=subject, predicate=predicate)

        return StoredFact(
            uuid=fact_uuid,
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source=source,
            created_at=fact_data.get("created_at", datetime.utcnow()),
        )

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

    def _hex_to_binary(self, hex_code: str) -> str:
        """Convert 8-char hex code to 32-bit binary string."""
        return bin(int(hex_code, 16))[2:].zfill(32)
