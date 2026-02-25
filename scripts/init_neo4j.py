#!/usr/bin/env python3
"""Initialize Neo4j database with schema and seed data."""

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uht_substrate.config.logging import configure_logging, get_logger
from uht_substrate.config.settings import get_settings
from uht_substrate.graph.connection import Neo4jConnection
from uht_substrate.priors.heuristics import HeuristicRepository
from uht_substrate.priors.ontology import OntologyRepository
from uht_substrate.priors.trait_axioms import TraitAxiomRepository

configure_logging()
logger = get_logger(__name__)


async def seed_traits(conn: Neo4jConnection, axiom_repo: TraitAxiomRepository) -> None:
    """Seed trait nodes from axiom definitions."""
    logger.info("Seeding trait nodes")

    for trait in axiom_repo.get_all_traits():
        await conn.execute_write(
            """
            MERGE (t:Trait {bit_position: $bit})
            ON CREATE SET
                t.name = $name,
                t.layer = $layer,
                t.description = $description
            ON MATCH SET
                t.name = $name,
                t.layer = $layer,
                t.description = $description
            """,
            {
                "bit": trait.bit_position,
                "name": trait.name,
                "layer": trait.layer,
                "description": trait.description,
            },
        )

    logger.info("Seeded traits", count=len(axiom_repo.get_all_traits()))


async def seed_axioms(conn: Neo4jConnection, axiom_repo: TraitAxiomRepository) -> None:
    """Seed axiom nodes from trait axioms."""
    logger.info("Seeding axiom nodes")

    count = 0
    for trait in axiom_repo.get_all_traits():
        for axiom in trait.axioms:
            await conn.execute_write(
                """
                MERGE (a:Axiom {uuid: $uuid})
                ON CREATE SET
                    a.trait_bit = $trait_bit,
                    a.name = $name,
                    a.statement = $statement,
                    a.axiom_type = $axiom_type,
                    a.property = $property,
                    a.confidence = $confidence
                ON MATCH SET
                    a.name = $name,
                    a.statement = $statement,
                    a.axiom_type = $axiom_type,
                    a.property = $property,
                    a.confidence = $confidence
                """,
                {
                    "uuid": axiom.uuid,
                    "trait_bit": axiom.trait_bit,
                    "name": axiom.name,
                    "statement": axiom.statement,
                    "axiom_type": axiom.axiom_type,
                    "property": axiom.property,
                    "confidence": axiom.confidence,
                },
            )
            count += 1

    logger.info("Seeded axioms", count=count)


async def seed_heuristics(conn: Neo4jConnection, heuristic_repo: HeuristicRepository) -> None:
    """Seed heuristic nodes."""
    logger.info("Seeding heuristic nodes")

    for h in heuristic_repo.get_all():
        await conn.execute_write(
            """
            MERGE (h:Heuristic {uuid: $uuid})
            ON CREATE SET
                h.name = $name,
                h.description = $description,
                h.priority = $priority,
                h.applicability = $applicability
            ON MATCH SET
                h.name = $name,
                h.description = $description,
                h.priority = $priority,
                h.applicability = $applicability
            """,
            {
                "uuid": h.uuid,
                "name": h.name,
                "description": h.description,
                "priority": h.priority,
                "applicability": h.applicability,
            },
        )

    logger.info("Seeded heuristics", count=len(heuristic_repo.get_all()))


async def seed_ontology(conn: Neo4jConnection, ontology_repo: OntologyRepository) -> None:
    """Seed ontological commitment nodes."""
    logger.info("Seeding ontological commitment nodes")

    for c in ontology_repo.get_all():
        await conn.execute_write(
            """
            MERGE (oc:OntologicalCommitment {uuid: $uuid})
            ON CREATE SET
                oc.name = $name,
                oc.statement = $statement,
                oc.category = $category,
                oc.confidence = $confidence
            ON MATCH SET
                oc.name = $name,
                oc.statement = $statement,
                oc.category = $category,
                oc.confidence = $confidence
            """,
            {
                "uuid": c.uuid,
                "name": c.name,
                "statement": c.statement,
                "category": c.category,
                "confidence": c.confidence,
            },
        )

    logger.info("Seeded ontological commitments", count=len(ontology_repo.get_all()))


async def main() -> None:
    """Initialize the database."""
    settings = get_settings()
    logger.info("Initializing Neo4j database", uri=settings.neo4j_uri)

    # Connect to Neo4j
    conn = Neo4jConnection(settings)
    await conn.connect()

    try:
        # Initialize schema
        await conn.initialize_schema()

        # Load priors
        data_path = Path(__file__).parent.parent / "data" / "priors"
        axiom_repo = TraitAxiomRepository(data_path)
        ontology_repo = OntologyRepository(data_path)
        heuristic_repo = HeuristicRepository(data_path)

        # Seed data
        await seed_traits(conn, axiom_repo)
        await seed_axioms(conn, axiom_repo)
        await seed_heuristics(conn, heuristic_repo)
        await seed_ontology(conn, ontology_repo)

        logger.info("Database initialization complete")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
