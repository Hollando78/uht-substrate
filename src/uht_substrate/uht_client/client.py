"""Async HTTP client for UHT Factory API."""

from typing import Optional

import httpx

from ..config.logging import get_logger
from ..config.settings import Settings, get_settings
from .cache import ResponseCache
from .models import (
    ClassificationResult,
    DisambiguationResult,
    Entity,
    NeighborhoodResult,
    PreprocessingResult,
    SemanticSearchResult,
    SemanticTriangle,
    SimilarityResult,
    TraitDefinition,
)

logger = get_logger(__name__)


class UHTClientError(Exception):
    """Base exception for UHT client errors."""

    pass


class UHTClient:
    """Async client for UHT Factory API."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        cache_ttl: int = 3600,
        api_key: Optional[str] = None,
    ):
        """
        Initialize UHT Factory API client.

        Args:
            settings: Application settings (uses defaults if not provided)
            cache_ttl: Cache time-to-live in seconds
            api_key: Optional API key for authenticated endpoints
        """
        self._settings = settings or get_settings()
        self._api_key = api_key

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key

        self._client = httpx.AsyncClient(
            base_url=self._settings.api_base_url,
            timeout=float(self._settings.api_timeout),
            headers=headers,
        )
        self._cache = ResponseCache(ttl=cache_ttl)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "UHTClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.close()

    # =========================================================================
    # Classification Endpoints
    # =========================================================================

    async def classify(
        self,
        entity: str,
        context: Optional[str] = None,
        force_refresh: bool = False,
        namespace: Optional[str] = None,
    ) -> ClassificationResult:
        """
        Classify an entity using 32 parallel trait evaluators.

        POST /classify/

        Args:
            entity: Entity name to classify
            context: Optional context to guide classification
            force_refresh: Skip cache and force fresh classification
            namespace: Optional namespace for cache key differentiation

        Returns:
            Classification result with hex code and trait assessments
        """
        # Cache key includes entity, context, and namespace
        # Same entity in different namespaces/contexts may have different classifications
        cache_key = f"classify:{entity}:{context or ''}:{namespace or 'global'}"

        if not force_refresh:
            cached = self._cache.get(cache_key)
            if cached:
                logger.debug("Cache hit for classification", entity=entity)
                return cached

        entity_obj: dict[str, object] = {"name": entity}
        if context:
            entity_obj["description"] = context
        payload: dict[str, object] = {"entity": entity_obj}
        if force_refresh:
            payload["use_cache"] = False

        logger.info(
            "Classifying entity",
            entity=entity,
            has_context=bool(context),
            use_cache=not force_refresh,
        )
        response = await self._client.post("/classify/", json=payload)
        response.raise_for_status()

        data = response.json()
        # API returns nested structure under "entity"
        entity_data = data.get("entity", data)
        result = ClassificationResult.model_validate(entity_data)
        self._cache.set(cache_key, result)

        return result

    async def classify_batch(
        self,
        entities: list[str],
        context: Optional[str] = None,
    ) -> list[ClassificationResult]:
        """
        Classify multiple entities in batch.

        POST /classify/batch

        Args:
            entities: List of entity names to classify
            context: Optional shared context

        Returns:
            List of classification results
        """
        payload: dict[str, object] = {"entities": entities}
        if context:
            payload["context"] = context

        logger.info("Batch classifying entities", count=len(entities))
        response = await self._client.post("/classify/batch", json=payload)
        response.raise_for_status()

        data = response.json()
        return [ClassificationResult.model_validate(r) for r in data.get("results", [])]

    async def explain_classification(
        self,
        entity_name: str,
        uht_code: str,
    ) -> dict[str, str]:
        """
        Get explanation for why entity has specific classification.

        POST /classify/explain

        Args:
            entity_name: Entity name
            uht_code: The hex code to explain

        Returns:
            Dict mapping trait names to explanations
        """
        response = await self._client.post(
            "/classify/explain",
            params={"entity_name": entity_name, "uht_code": uht_code},
        )
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Entity Endpoints
    # =========================================================================

    async def search_entities(
        self,
        query: Optional[str] = None,
        uht_pattern: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        """
        Search for entities.

        GET /entities/

        Args:
            query: Name search query
            uht_pattern: Hex pattern with wildcards (e.g., "FF??00??")
            limit: Maximum results (1-50000)
            offset: Pagination offset

        Returns:
            List of matching entities
        """
        params: dict[str, object] = {"limit": limit, "offset": offset}
        if query:
            params["name_contains"] = query
        if uht_pattern:
            params["uht_pattern"] = uht_pattern

        response = await self._client.get("/entities/", params=params)
        response.raise_for_status()

        data = response.json()
        # API returns {"entities": [...], "total": N, ...}
        results = data if isinstance(data, list) else data.get("entities", data.get("results", []))
        return [Entity.model_validate(e) for e in results]

    async def get_entity(self, uuid: str) -> Entity:
        """
        Get entity details by UUID.

        GET /entities/{uuid}

        Args:
            uuid: Entity UUID

        Returns:
            Entity details
        """
        cache_key = f"entity:{uuid}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        response = await self._client.get(f"/entities/{uuid}")
        response.raise_for_status()

        entity = Entity.model_validate(response.json())
        self._cache.set(cache_key, entity)

        return entity

    async def find_similar(
        self,
        uuid: str,
        threshold: int = 28,
        limit: int = 10,
    ) -> list[SimilarityResult]:
        """
        Find entities similar to a given entity.

        GET /entities/{uuid}/similar

        Args:
            uuid: Entity UUID
            threshold: Minimum shared traits (20-32)
            limit: Maximum results

        Returns:
            List of similar entities with scores
        """
        params = {"threshold": threshold, "limit": limit}
        response = await self._client.get(f"/entities/{uuid}/similar", params=params)
        response.raise_for_status()

        data = response.json()
        # API returns {"source_entity": {...}, "similar_entities": [...], "threshold": N}
        results = data if isinstance(data, list) else data.get("similar_entities", data.get("results", []))
        return [SimilarityResult.model_validate(r) for r in results]

    async def search_by_pattern(
        self,
        pattern: str,
        tolerance: int = 0,
        limit: int = 100,
    ) -> list[Entity]:
        """
        Search entities by binary pattern with wildcards.

        GET /entities/search/pattern

        Args:
            pattern: 32-char pattern with 0/1/X (X = wildcard)
            tolerance: Number of mismatches allowed (0-8)
            limit: Maximum results (1-500)

        Returns:
            List of matching entities
        """
        params = {"pattern": pattern, "tolerance": tolerance, "limit": limit}
        response = await self._client.get("/entities/search/pattern", params=params)
        response.raise_for_status()

        data = response.json()
        # API returns {"entities": [...], "total": N, ...}
        results = data if isinstance(data, list) else data.get("entities", data.get("results", []))
        return [Entity.model_validate(e) for e in results]

    # =========================================================================
    # Trait Endpoints
    # =========================================================================

    async def get_traits(self) -> tuple[list[TraitDefinition], str]:
        """
        Get all 32 canonical trait definitions.

        GET /traits/

        Returns:
            Tuple of (trait definitions list, version string)
        """
        cache_key = "traits:all"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        response = await self._client.get("/traits/")
        response.raise_for_status()

        data = response.json()
        version = data.get("version", "")

        # API returns traits nested under "layers" dict
        # Flatten into a single list
        traits_data = []
        if "layers" in data:
            for layer_name, layer_traits in data["layers"].items():
                traits_data.extend(layer_traits)
        elif isinstance(data, list):
            traits_data = data
        else:
            traits_data = data.get("traits", [])

        traits = [TraitDefinition.model_validate(t) for t in traits_data]

        result = (traits, version)
        # Cache for 24 hours since traits don't change often
        self._cache.set(cache_key, result, ttl=86400)

        return result

    async def get_trait(self, bit: int) -> TraitDefinition:
        """
        Get a single trait definition.

        GET /traits/{bit}

        Args:
            bit: Bit position (1-32)

        Returns:
            Trait definition
        """
        response = await self._client.get(f"/traits/{bit}")
        response.raise_for_status()
        return TraitDefinition.model_validate(response.json())

    async def get_trait_prompts(
        self,
        entity_name: str = "{{entity_name}}",
        entity_description: str = "{{entity_description}}",
    ) -> dict[int, str]:
        """
        Get classifier prompts for all traits.

        GET /traits/prompts

        Args:
            entity_name: Entity name placeholder
            entity_description: Description placeholder

        Returns:
            Dict mapping bit position to classifier prompt
        """
        params = {"entity_name": entity_name, "entity_description": entity_description}
        response = await self._client.get("/traits/prompts", params=params)
        response.raise_for_status()
        return response.json()

    async def get_trait_cooccurrence(self) -> dict[str, object]:
        """
        Get trait co-occurrence statistics.

        GET /traits/statistics/cooccurrence

        Returns:
            Co-occurrence matrix and statistics
        """
        response = await self._client.get("/traits/statistics/cooccurrence")
        response.raise_for_status()
        return response.json()

    async def get_trait_statistics(self) -> dict[str, object]:
        """
        Get full trait statistics.

        GET /traits/statistics/full

        Returns:
            Comprehensive trait analytics
        """
        response = await self._client.get("/traits/statistics/full")
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Preprocessing Endpoints
    # =========================================================================

    async def get_semantic_triangle(self, text: str) -> SemanticTriangle:
        """
        Get semantic triangle decomposition.

        POST /preprocess/triangle

        Args:
            text: Entity name or phrase

        Returns:
            Semantic triangle (symbol, referent, reference)
        """
        response = await self._client.post("/preprocess/triangle", json={"entity_name": text})
        response.raise_for_status()
        return SemanticTriangle.model_validate(response.json())

    async def preprocess(self, entity_name: str) -> PreprocessingResult:
        """
        Preprocess an entity before classification.

        POST /preprocess/preprocess

        Args:
            entity_name: Entity to preprocess

        Returns:
            Preprocessing result with normalization and checks
        """
        response = await self._client.post(
            "/preprocess/preprocess",
            params={"entity_name": entity_name},
        )
        response.raise_for_status()
        return PreprocessingResult.model_validate(response.json())

    async def check_duplicate(
        self,
        entity_name: str,
        threshold: float = 0.8,
    ) -> list[Entity]:
        """
        Check for potential duplicates of an entity.

        POST /preprocess/duplicate-check

        Args:
            entity_name: Entity to check
            threshold: Similarity threshold (0-1)

        Returns:
            List of potential duplicate entities
        """
        response = await self._client.post(
            "/preprocess/duplicate-check",
            params={"entity_name": entity_name, "threshold": threshold},
        )
        response.raise_for_status()

        data = response.json()
        duplicates = data if isinstance(data, list) else data.get("duplicates", [])
        return [Entity.model_validate(e) for e in duplicates]

    # =========================================================================
    # Graph Endpoints
    # =========================================================================

    async def get_neighborhood(
        self,
        uuid: str,
        metric: str = "embedding",
        k: int = 15,
        min_similarity: float = 0.3,
    ) -> NeighborhoodResult:
        """
        Get semantic neighborhood of an entity.

        GET /graph/neighborhood/{uuid}

        Args:
            uuid: Entity UUID
            metric: Similarity metric (embedding/hamming/hybrid)
            k: Number of neighbors (5-50)
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            Neighborhood graph with nodes and edges
        """
        params = {
            "metric": metric,
            "k": k,
            "min_similarity": min_similarity,
            "include_traits": True,
        }
        response = await self._client.get(f"/graph/neighborhood/{uuid}", params=params)
        response.raise_for_status()
        return NeighborhoodResult.model_validate(response.json())

    async def expand_graph(
        self,
        entity_uuids: list[str],
        depth: int = 1,
    ) -> NeighborhoodResult:
        """
        Expand graph from multiple entities.

        POST /graph/expand

        Args:
            entity_uuids: Starting entity UUIDs
            depth: Expansion depth

        Returns:
            Expanded neighborhood graph
        """
        response = await self._client.post(
            "/graph/expand",
            json={"entity_uuids": entity_uuids, "depth": depth},
        )
        response.raise_for_status()
        return NeighborhoodResult.model_validate(response.json())

    # =========================================================================
    # Embedding/Semantic Search Endpoints
    # =========================================================================

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[SemanticSearchResult]:
        """
        Semantic similarity search using embeddings.

        POST /embeddings/search

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of semantically similar entities with similarity scores
        """
        response = await self._client.post(
            "/embeddings/search",
            json={"query": query, "limit": limit},
        )
        response.raise_for_status()

        data = response.json()
        results = data if isinstance(data, list) else data.get("results", [])
        return [SemanticSearchResult.model_validate(e) for e in results]

    # =========================================================================
    # Dictionary/Disambiguation Endpoints
    # =========================================================================

    async def disambiguate(
        self,
        lemma: str,
        lang: str = "en",
    ) -> DisambiguationResult:
        """
        Disambiguate a polysemous word.

        GET /dictionary/disambiguate/{lemma}

        Args:
            lemma: Word to disambiguate
            lang: Language code (en/fr/de)

        Returns:
            Disambiguation result with all senses
        """
        response = await self._client.get(
            f"/dictionary/disambiguate/{lemma}",
            params={"lang": lang},
        )
        response.raise_for_status()
        return DisambiguationResult.model_validate(response.json())

    async def search_dictionary(
        self,
        query: str,
        lang: str = "en",
        limit: int = 20,
    ) -> list[dict[str, object]]:
        """
        Search dictionary for words.

        GET /dictionary/search

        Args:
            query: Search query
            lang: Language code
            limit: Maximum results

        Returns:
            List of matching words
        """
        params = {"q": query, "lang": lang, "limit": limit}
        response = await self._client.get("/dictionary/search", params=params)
        response.raise_for_status()

        data = response.json()
        return data if isinstance(data, list) else data.get("results", [])

    # =========================================================================
    # Hex Calculator Endpoints
    # =========================================================================

    async def analyze_hex(self, hex_code: str) -> dict[str, object]:
        """
        Analyze a hex code for its traits.

        POST /hex-calc/analyze

        Args:
            hex_code: 8-character hex code

        Returns:
            Analysis including layers, present traits, etc.
        """
        response = await self._client.post("/hex-calc/analyze", json={"hex_code": hex_code})
        response.raise_for_status()
        return response.json()

    async def name_hex(self, hex_code: str) -> dict[str, object]:
        """
        Generate a name/description for a hex code.

        POST /hex-calc/name

        Args:
            hex_code: 8-character hex code

        Returns:
            Generated name and description
        """
        response = await self._client.post("/hex-calc/name", json={"hex_code": hex_code})
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Health/System Endpoints
    # =========================================================================

    async def health_check(self) -> dict[str, object]:
        """
        Check API health.

        GET /health

        Returns:
            Health status
        """
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()
