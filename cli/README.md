# uht-substrate

CLI for the [Universal Hex Taxonomy](https://factory.universalhex.org) Substrate API — classify entities, compare concepts, manage facts, and explore semantic relationships from your terminal.

## Install

```bash
npm install -g uht-substrate
```

Or run without installing:

```bash
npx uht-substrate <command>
```

## Setup

```bash
uht-substrate config set token <your-api-token>
```

The default API URL is `https://substrate.universalhex.org/api`. To override:

```bash
uht-substrate config set api-url https://your-server.com/api
```

Or use environment variables:

```bash
export UHT_API_URL=https://substrate.universalhex.org/api
export UHT_TOKEN=your-token
```

## Quick Start

```bash
# Classify an entity — get its 8-char hex code and 32 trait bits
uht-substrate classify hammer
uht-substrate classify "cognitive dissonance" -c "a psychology concept" --format pretty

# Reclassify with a rich description (skips cache)
uht-substrate classify "epistemic humility" \
  -c "the intellectual virtue of recognizing the limits of one's knowledge" \
  --force-refresh

# Compare two entities — Jaccard similarity, shared/unique traits
uht-substrate compare hammer screwdriver --format pretty
uht-substrate compare "Account (grounds)" democracy --format pretty

# Batch compare — rank candidates by similarity
uht-substrate batch-compare hammer wrench screwdriver pliers saw

# Semantic search across 16k+ entities
uht-substrate search "hand tools" --limit 10

# Disambiguate a polysemous term
uht-substrate disambiguate bank

# Infer properties from classification
uht-substrate infer hammer
```

## Commands

### Classification & Reasoning

| Command | Description |
|---------|-------------|
| `classify <entity>` | Classify an entity and get its hex code (`-c`, `-f`, `-n`) |
| `infer <entity>` | Infer properties from classification |
| `compare <a> <b>` | Compare two entities |
| `batch-compare <entity> <candidates...>` | Compare against multiple, ranked by Jaccard |
| `search <query>` | Semantic search across the entity corpus |
| `disambiguate <term>` | Get word senses for polysemous terms |
| `semantic-triangle <text>` | Ogden-Richards semantic triangle decomposition |
| `map-properties <props...>` | Map natural language properties to UHT trait bits |

### Entity Management

| Command | Description |
|---------|-------------|
| `entities list` | List entities (with uuid, description) |
| `entities get` | Get a single entity by `--name` or `--uuid` |
| `entities delete <name>` | Delete an entity from the local graph |
| `entities find-similar <entity>` | Find similar entities (experimental) |
| `entities explore <entity>` | Explore semantic neighborhood (experimental) |
| `entities search-traits` | Search entities by trait pattern |

#### Trait search example

Use `--<trait-name>` to require a trait, `--no-<trait-name>` to exclude:

```bash
uht-substrate entities search-traits --synthetic --powered --no-biological
```

### Fact Management

| Command | Description |
|---------|-------------|
| `facts store <subject> <predicate> <object>` | Store a fact |
| `facts store-bulk --file <path>` | Store multiple facts from a JSON file |
| `facts upsert <subject> <predicate> <object>` | Create or update a fact |
| `facts query` | Query facts with filters |
| `facts update <fact-id>` | Update an existing fact |
| `facts delete <fact-id>` | Delete a fact |
| `facts user-context` | Get a user's stored facts |
| `facts namespace-context <namespace>` | Get all entities and facts under a namespace |

```bash
# Store a typed fact
uht-substrate facts store "spark plug" PART_OF "engine" --namespace SE:automotive

# Query facts
uht-substrate facts query --subject "spark plug" --category compositional

# Bulk store from JSON
echo '[{"subject":"bolt","predicate":"PART_OF","object_value":"frame"}]' | uht-substrate facts store-bulk -f -
```

### Namespace Management

| Command | Description |
|---------|-------------|
| `namespaces create <code> <name>` | Create a namespace |
| `namespaces list` | List namespaces |
| `namespaces assign <entity> <namespace>` | Assign an entity to a namespace |

```bash
uht-substrate namespaces create SE:automotive "Automotive Engineering"
uht-substrate namespaces assign "spark plug" SE:automotive
uht-substrate namespaces list --parent SE --descendants
```

### Reference

| Command | Description |
|---------|-------------|
| `info` | System information and overview |
| `traits` | All 32 trait definitions (with version) |
| `trait-prompts` | Classifier prompts sent to the LLM (edge cases, examples) |
| `patterns` | Reasoning patterns for tool orchestration |

### Configuration

| Command | Description |
|---------|-------------|
| `config set <key> <value>` | Set a config value (`api-url`, `token`, `format`) |
| `config show` | Show current configuration |

## Pipeline Integration

### Semantic Impact Analysis

Analyse the semantic impact of requirement changes from an [Airgen](https://airgen.io) diff. For each changed requirement, the command classifies the text using UHT and detects **semantic drift** — cases where a textual edit shifts the requirement into a different meaning-space.

```bash
# Generate a diff from airgen
airgen diff --json > diff.json

# Analyse impact
uht-substrate impact --airgen-diff diff.json --format pretty
uht-substrate impact --airgen-diff diff.json --json
```

The input file must be the JSON output from `airgen diff --json`, structured as `{ summary, added, removed, modified }` where each requirement has `ref` and `text` fields, and modified entries have `old_text` and `new_text`.

**Pretty output:**

```
Summary: 1 added, 1 removed, 2 modified (1 with semantic drift)

Semantic Drift:
  REQ-005  40802B01 → 40846B01
    + System-Integrated
    + Signalling

Added:
  REQ-030  00802940  Intentionally Designed, Rule-Governed, Normative

Removed:
  REQ-010  40C06900  Synthetic, Intentionally Designed, Outputs Effect
```

**JSON output:**

```json
{
  "summary": { "added": 1, "removed": 1, "modified": 2, "semantic_drift": 1 },
  "drift": [{
    "ref": "REQ-005",
    "old_hex": "40802B01",
    "new_hex": "40846B01",
    "flipped_traits": [
      { "bit": 14, "name": "System-Integrated", "direction": "added" }
    ]
  }],
  "added": [{ "ref": "REQ-030", "hex": "00802940", "text": "...", "top_traits": ["..."] }],
  "removed": [{ "ref": "REQ-010", "hex": "40C06900", "text": "...", "top_traits": ["..."] }]
}
```

## Global Options

| Option | Description |
|--------|-------------|
| `--api-url <url>` | API base URL |
| `--token <token>` | Bearer token for authentication |
| `--format <fmt>` | Output format: `json` (default) or `pretty` |
| `--verbose` | Show request/response details |
| `--version` | Show version |

## Output Formats

**JSON** (default) — machine-readable, pipe to `jq`:

```bash
uht-substrate classify hammer | jq '.hex_code'
```

**Pretty** — colored terminal output:

```bash
uht-substrate classify hammer --format pretty
```

## What is UHT?

The Universal Hex Taxonomy encodes any concept as a 32-bit classification (8-char hex code) across four layers:

- **Physical** (bits 1–8): Material properties, boundaries, energy
- **Functional** (bits 9–16): Capabilities, interfaces, state
- **Abstract** (bits 17–24): Symbolic, temporal, rule-governed
- **Social** (bits 25–32): Cultural, economic, institutional

Two entities with similar hex codes share similar properties. Jaccard similarity measures meaningful overlap between classifications.

## License

MIT
