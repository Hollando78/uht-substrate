# UHT Substrate

Semantic reasoning engine built on the [Universal Hex Taxonomy](https://universalhex.org). Classifies any concept into a 32-bit trait vector (8-char hex code), then uses that classification for comparison, search, fact storage, and knowledge graph operations.

Exposes three interfaces:
- **MCP server** — for AI agents (Claude, etc.)
- **REST API** — for apps, scripts, ChatGPT actions
- **CLI** — for terminal workflows (`npm install -g uht-substrate`)

Live at [substrate.universalhex.org](https://substrate.universalhex.org)

## Quick Start

```bash
# Start Neo4j
docker-compose up -d

# Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Initialize database
python scripts/init_neo4j.py

# Run server (serves MCP, REST API, and landing page on :8765)
uht-substrate
```

## Configuration

Copy `.env.example` to `.env` and configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `UHT_API_BASE_URL` | UHT Factory API URL | `https://factory.universalhex.org/api/v1` |
| `UHT_NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `UHT_NEO4J_PASSWORD` | Neo4j password | `uhtsubstrate123` |
| `UHT_SERVER_PORT` | Server port | `8765` |

## What is UHT?

Every entity is evaluated against **32 binary traits** across four semantic layers, producing an 8-character hex code — a universal coordinate for meaning.

| Layer | Bits | Examples |
|-------|------|----------|
| **Physical** | 1–8 | Physical Object, Synthetic, Biological, Powered |
| **Functional** | 9–16 | Intentionally Designed, State-Transforming, System-Integrated |
| **Abstract** | 17–24 | Symbolic, Rule-Governed, Compositional, Temporal |
| **Social** | 25–32 | Social Construct, Regulated, Economically Significant |

Two entities with similar hex codes share similar properties. Jaccard similarity over the 32-bit vectors measures meaningful overlap.

## MCP Tools

For AI agent integrations (Claude Desktop, etc.):

| Tool | Description |
|------|-------------|
| `classify_entity` | Classify any entity — returns hex code + 32 traits |
| `compare_entities` | Compare two entities (Jaccard similarity, shared/unique traits) |
| `batch_compare` | Compare one entity against many, ranked by similarity |
| `search_entities` | Semantic search across the entity corpus |
| `disambiguate_term` | Get word senses for polysemous terms |
| `infer_properties` | Infer properties from classification |
| `explore_neighborhood` | Find related entities in the knowledge graph |
| `find_similar_entities` | Find similar entities by trait overlap |
| `semantic_triangle` | Ogden-Richards semantic triangle decomposition |
| `map_properties_to_traits` | Map natural language properties to UHT trait bits |
| `store_fact` / `upsert_fact` | Store or upsert facts (subject-predicate-object) |
| `store_facts_bulk` | Bulk store facts from a list |
| `query_facts` | Query facts with filters (subject, predicate, category, namespace) |
| `get_user_context` | Get a user's stored facts |
| `get_namespace_context` | Get all entities and facts under a namespace |
| `create_namespace` / `list_namespaces` | Namespace management |
| `get_info` / `get_traits` / `get_patterns` | System reference data |

## REST API

All capabilities are also available via REST at `/api/`. Interactive docs at `/api/docs`.

```bash
# Classify
curl -X POST https://substrate.universalhex.org/api/classify \
  -H "Content-Type: application/json" \
  -d '{"entity": "hammer"}'

# Compare
curl -X POST https://substrate.universalhex.org/api/compare \
  -H "Content-Type: application/json" \
  -d '{"entity_a": "hammer", "entity_b": "screwdriver"}'

# Search
curl -X POST https://substrate.universalhex.org/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "hand tools", "limit": 10}'
```

See the full endpoint list at [`/api/docs`](https://substrate.universalhex.org/api/docs).

## CLI

Install the CLI for terminal workflows:

```bash
npm install -g uht-substrate
```

```bash
uht-substrate classify hammer --format pretty
uht-substrate compare hammer screwdriver
uht-substrate search "hand tools" --limit 10
uht-substrate facts store "spark plug" PART_OF "engine" --namespace SE:automotive
```

See [`cli/README.md`](cli/README.md) for full CLI documentation.

## Project Structure

```
src/uht_substrate/
  server.py          # MCP server, REST API, landing page
  tools/
    classify.py      # Classification engine
    context.py       # User context and fact management
    explore.py       # Semantic search and exploration
  uht_client/
    client.py        # HTTP client for UHT Factory API
  landing.html       # Interactive landing page
cli/                 # npm CLI package (TypeScript)
scripts/             # Database init and migration scripts
```

## License

MIT
