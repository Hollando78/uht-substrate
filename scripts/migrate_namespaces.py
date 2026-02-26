#!/usr/bin/env python3
"""
Migration script to add namespace support to existing UHT Substrate graph.

Steps:
1. Ensure "global" namespace node exists
2. Add BELONGS_TO relationships from all existing entities to global
3. Validate migration

Usage:
    python scripts/migrate_namespaces.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uht_substrate.config.logging import get_logger
from uht_substrate.config.settings import Settings
from uht_substrate.graph.connection import Neo4jConnection

logger = get_logger(__name__)


async def migrate():
    """Run the namespace migration."""
    settings = Settings()
    conn = Neo4jConnection(settings)

    try:
        await conn.connect()
        logger.info("Connected to Neo4j")

        # Step 1: Ensure global namespace exists
        logger.info("Step 1: Creating global namespace if not exists...")
        result = await conn.execute_write(
            """
            MERGE (n:Namespace {code: 'global'})
            ON CREATE SET
                n.uuid = randomUUID(),
                n.name = 'Global',
                n.description = 'Default namespace for all entities',
                n.created_at = datetime(),
                n.is_root = true
            RETURN n.uuid as uuid, n.code as code
            """
        )
        if result:
            logger.info(
                "Global namespace ready",
                uuid=result[0]["uuid"],
                code=result[0]["code"],
            )

        # Step 2: Count entities without namespace
        unlinked = await conn.execute_query(
            """
            MATCH (e:Entity)
            WHERE NOT EXISTS { (e)-[:BELONGS_TO]->(:Namespace) }
            RETURN count(e) as count
            """
        )
        unlinked_count = unlinked[0]["count"] if unlinked else 0
        logger.info(f"Found {unlinked_count} entities without namespace")

        # Step 3: Link unlinked entities to global namespace
        if unlinked_count > 0:
            logger.info("Step 2: Linking entities to global namespace...")
            result = await conn.execute_write(
                """
                MATCH (e:Entity)
                WHERE NOT EXISTS { (e)-[:BELONGS_TO]->(:Namespace) }
                MATCH (g:Namespace {code: 'global'})
                CREATE (e)-[r:BELONGS_TO {
                    primary: true,
                    assigned_at: datetime(),
                    migrated: true
                }]->(g)
                RETURN count(e) as migrated_count
                """
            )
            migrated = result[0]["migrated_count"] if result else 0
            logger.info(f"Migrated {migrated} entities to global namespace")

        # Step 4: Validate
        logger.info("Step 3: Validating migration...")
        validation = await conn.execute_query(
            """
            MATCH (e:Entity)
            WHERE NOT EXISTS { (e)-[:BELONGS_TO]->(:Namespace) }
            RETURN count(e) as orphaned_count
            """
        )
        orphaned = validation[0]["orphaned_count"] if validation else 0

        if orphaned > 0:
            logger.warning(f"WARNING: {orphaned} entities still without namespace!")
        else:
            logger.info("All entities have namespace assignments")

        # Step 5: Summary
        summary = await conn.execute_query(
            """
            MATCH (n:Namespace)
            OPTIONAL MATCH (e:Entity)-[:BELONGS_TO]->(n)
            RETURN n.code as namespace, count(e) as entity_count
            ORDER BY n.code
            """
        )
        logger.info("Namespace summary:")
        for row in summary:
            logger.info(f"  {row['namespace']}: {row['entity_count']} entities")

        logger.info("Migration complete!")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
