"""Neo4j database connection management."""

from typing import Any, Optional

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import ServiceUnavailable

from ..config.logging import get_logger
from ..config.settings import Settings, get_settings
from .schema import get_schema_statements

logger = get_logger(__name__)


class Neo4jConnection:
    """Manages async Neo4j database connections."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Neo4j connection.

        Args:
            settings: Application settings (uses defaults if not provided)
        """
        self._settings = settings or get_settings()
        self._driver: Optional[AsyncDriver] = None

    async def connect(self) -> None:
        """Establish connection to Neo4j database."""
        if self._driver is not None:
            return

        logger.info(
            "Connecting to Neo4j",
            uri=self._settings.neo4j_uri,
            database=self._settings.neo4j_database,
        )

        self._driver = AsyncGraphDatabase.driver(
            self._settings.neo4j_uri,
            auth=(self._settings.neo4j_user, self._settings.neo4j_password),
        )

        # Verify connectivity
        try:
            await self._driver.verify_connectivity()
            logger.info("Neo4j connection established")
        except ServiceUnavailable as e:
            logger.error("Failed to connect to Neo4j", error=str(e))
            raise

    async def close(self) -> None:
        """Close the database connection."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    async def __aenter__(self) -> "Neo4jConnection":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.close()

    @property
    def driver(self) -> AsyncDriver:
        """Get the Neo4j driver, raising if not connected."""
        if self._driver is None:
            raise RuntimeError("Not connected to Neo4j. Call connect() first.")
        return self._driver

    def session(self, **kwargs: Any) -> AsyncSession:
        """
        Get a new database session.

        Args:
            **kwargs: Additional session configuration

        Returns:
            AsyncSession for database operations
        """
        return self.driver.session(
            database=self._settings.neo4j_database,
            **kwargs,
        )

    async def execute_query(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Execute a Cypher query and return results.

        Args:
            query: Cypher query string
            parameters: Query parameters
            **kwargs: Additional session configuration

        Returns:
            List of result records as dictionaries
        """
        async with self.session(**kwargs) as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def execute_write(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a write query in a transaction.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dictionaries
        """
        async with self.session() as session:

            async def work(tx: Any) -> list[dict[str, Any]]:
                result = await tx.run(query, parameters or {})
                return await result.data()

            return await session.execute_write(work)

    async def initialize_schema(self) -> None:
        """Initialize database schema with constraints and indexes."""
        logger.info("Initializing Neo4j schema")

        statements = get_schema_statements()
        for statement in statements:
            try:
                await self.execute_query(statement)
                logger.debug("Executed schema statement", statement=statement[:50])
            except Exception as e:
                # Some constraints may already exist, which is fine
                if "already exists" not in str(e).lower():
                    logger.warning(
                        "Schema statement failed",
                        statement=statement[:50],
                        error=str(e),
                    )

        logger.info("Schema initialization complete")

    async def health_check(self) -> bool:
        """
        Check if database is healthy.

        Returns:
            True if database is responsive
        """
        try:
            result = await self.execute_query("RETURN 1 as health")
            return len(result) > 0 and result[0].get("health") == 1
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False
