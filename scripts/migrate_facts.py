#!/usr/bin/env python3
"""
Migration script to add predicate taxonomy fields to existing Fact nodes.

Steps:
1. Find all Fact nodes missing the 'category' field
2. Categorize each fact's predicate using PredicateTaxonomy
3. Set category, is_custom_predicate, bound=false, source
4. Attempt entity binding on all unbound facts
5. Report summary

Usage:
    python scripts/migrate_facts.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uht_substrate.config.logging import get_logger
from uht_substrate.config.settings import Settings
from uht_substrate.graph.connection import Neo4jConnection
from uht_substrate.graph.schema import PredicateTaxonomy

logger = get_logger(__name__)


async def migrate():
    """Run the fact taxonomy migration."""
    settings = Settings()
    conn = Neo4jConnection(settings)

    try:
        await conn.connect()
        logger.info("Connected to Neo4j")

        # Step 1: Find facts without category
        logger.info("Step 1: Finding facts without category...")
        unmigrated = await conn.execute_query(
            """
            MATCH (f:Fact)
            WHERE f.category IS NULL
            RETURN f
            ORDER BY f.created_at ASC
            """
        )
        logger.info(f"Found {len(unmigrated)} facts without category")

        if not unmigrated:
            logger.info("No facts to migrate")
        else:
            # Step 2: Categorize and update each fact
            logger.info("Step 2: Categorizing facts...")
            category_counts: dict[str, int] = {}
            custom_count = 0

            for row in unmigrated:
                f = row["f"]
                predicate = f.get("predicate", "RELATED_TO")
                category, is_custom = PredicateTaxonomy.categorize(predicate)

                # Determine source: check if existing source looks computed
                existing_source = f.get("source", "")
                if existing_source in ("computed", "system"):
                    source = "computed"
                elif existing_source in ("inferred",):
                    source = "inferred"
                else:
                    source = "asserted"

                await conn.execute_write(
                    """
                    MATCH (f:Fact {uuid: $uuid})
                    SET f.category = $category,
                        f.is_custom_predicate = $is_custom,
                        f.bound = COALESCE(f.bound, false),
                        f.source = $source,
                        f.updated_at = datetime()
                    RETURN f
                    """,
                    {
                        "uuid": f["uuid"],
                        "category": category,
                        "is_custom": is_custom,
                        "source": source,
                    },
                )

                category_counts[category] = category_counts.get(category, 0) + 1
                if is_custom:
                    custom_count += 1

            logger.info(f"Categorized {len(unmigrated)} facts:")
            for cat, count in sorted(category_counts.items()):
                logger.info(f"  {cat}: {count}")
            logger.info(f"  custom predicates: {custom_count}")

        # Step 3: Attempt entity binding for all unbound facts
        logger.info("Step 3: Attempting entity binding...")
        unbound_facts = await conn.execute_query(
            """
            MATCH (f:Fact)
            WHERE f.bound = false OR f.bound IS NULL
            RETURN f
            ORDER BY f.created_at ASC
            """
        )
        logger.info(f"Found {len(unbound_facts)} unbound facts")

        newly_bound = 0
        for row in unbound_facts:
            f = row["f"]
            fact_uuid = f["uuid"]
            subject = f["subject"]
            obj = f["object"]

            # Try to bind subject
            subj_result = await conn.execute_write(
                """
                MATCH (f:Fact {uuid: $fact_uuid})
                MATCH (e:Entity)
                WHERE toLower(e.name) = toLower($subject)
                MERGE (f)-[:FACT_ABOUT]->(e)
                RETURN e.uuid AS entity_uuid
                """,
                {"fact_uuid": fact_uuid, "subject": subject},
            )
            subj_uuid = subj_result[0]["entity_uuid"] if subj_result else None

            # Try to bind object
            obj_result = await conn.execute_write(
                """
                MATCH (f:Fact {uuid: $fact_uuid})
                MATCH (e:Entity)
                WHERE toLower(e.name) = toLower($object)
                MERGE (f)-[:FACT_REFERENCES]->(e)
                RETURN e.uuid AS entity_uuid
                """,
                {"fact_uuid": fact_uuid, "object": obj},
            )
            obj_uuid = obj_result[0]["entity_uuid"] if obj_result else None

            # If both bound, mark fact and create entity relationship
            if subj_uuid and obj_uuid:
                await conn.execute_write(
                    """
                    MATCH (f:Fact {uuid: $uuid})
                    SET f.bound = true,
                        f.subject_entity_uuid = $subj_uuid,
                        f.object_entity_uuid = $obj_uuid,
                        f.updated_at = datetime()
                    """,
                    {
                        "uuid": fact_uuid,
                        "subj_uuid": subj_uuid,
                        "obj_uuid": obj_uuid,
                    },
                )

                await conn.execute_write(
                    """
                    MATCH (e1:Entity {uuid: $source_uuid})
                    MATCH (e2:Entity {uuid: $target_uuid})
                    MERGE (e1)-[r:RELATED_TO {predicate: $predicate}]->(e2)
                    SET r.fact_uuid = $fact_uuid,
                        r.category = $category,
                        r.source = $source,
                        r.confidence = $confidence,
                        r.created_at = datetime()
                    """,
                    {
                        "source_uuid": subj_uuid,
                        "target_uuid": obj_uuid,
                        "predicate": f.get("predicate", "RELATED_TO"),
                        "fact_uuid": fact_uuid,
                        "category": f.get("category", "associative"),
                        "source": f.get("source", "asserted"),
                        "confidence": f.get("confidence", 1.0),
                    },
                )
                newly_bound += 1

        logger.info(f"Bound {newly_bound} facts to entities")

        # Step 4: Validate and summarize
        logger.info("Step 4: Validating migration...")
        summary = await conn.execute_query(
            """
            MATCH (f:Fact)
            RETURN f.category AS category,
                   f.source AS source,
                   f.bound AS bound,
                   count(f) AS count
            ORDER BY category, source
            """
        )

        logger.info("Migration summary:")
        total = 0
        for row in summary:
            total += row["count"]
            logger.info(
                f"  category={row['category']}, source={row['source']}, "
                f"bound={row['bound']}: {row['count']}"
            )
        logger.info(f"Total facts: {total}")

        # Check for any remaining unmigrated facts
        remaining = await conn.execute_query(
            """
            MATCH (f:Fact)
            WHERE f.category IS NULL
            RETURN count(f) AS count
            """
        )
        remaining_count = remaining[0]["count"] if remaining else 0
        if remaining_count > 0:
            logger.warning(f"WARNING: {remaining_count} facts still without category!")
        else:
            logger.info("All facts have category assignments")

        logger.info("Migration complete!")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
