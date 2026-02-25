# UHT Substrate Agent

An MCP server that uses the Universal Hex Taxonomy (UHT) as ground truth for understanding and reasoning about any concept or entity.

## Features

- **Classification**: Classify entities using UHT's 32-bit trait system
- **Reasoning**: Apply trait axioms and heuristics to derive conclusions
- **Knowledge Graph**: Persistent Neo4j storage for entities, facts, and traces
- **Semantic Search**: Find similar entities and explore relationships
- **User Context**: Store and retrieve facts and preferences

## Quick Start

```bash
# Start Neo4j
docker-compose up -d

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -e .

# Initialize database
python scripts/init_neo4j.py

# Run server
uht-substrate
```

## Configuration

Copy `.env.example` to `.env` and configure:

- `UHT_NEO4J_URI`: Neo4j connection URI
- `UHT_NEO4J_PASSWORD`: Neo4j password
- `UHT_API_BASE_URL`: UHT Factory API URL

## MCP Tools

- `classify_entity` - Classify any entity
- `compare_entities` - Compare two entities
- `reason_about` - Answer questions using reasoning
- `explore_neighborhood` - Find related entities
- `disambiguate_term` - Resolve polysemous words
- `store_fact` / `get_user_context` - Manage user context

## License

MIT
