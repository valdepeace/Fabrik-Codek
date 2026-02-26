---
name: fabrik-codek
description: "Personal cognitive architecture that learns how you work. Builds a knowledge graph from your sessions, profiles your expertise, adapts retrieval per task, and self-corrects via outcome feedback. Three-tier hybrid RAG (vector + graph + full-text). 100% local with any Ollama model."
version: 1.7.0
homepage: https://github.com/ikchain/Fabrik-Codek
user-invocable: true
metadata:
  clawdbot:
    requires:
      bins: [fabrik]
      anyBins: [python3, python]
    homepage: https://github.com/ikchain/Fabrik-Codek
    os: [macos, linux]
    emoji: "🧠"
    configPaths:
      - "~/.claude/projects/*"
      - "./data/embeddings/"
      - "./data/graphdb/"
      - "./data/profile/"
    install:
      - kind: pip
        package: fabrik-codek
        bins: [fabrik]
---

# Fabrik-Codek

> A 7B model that knows you is worth more than a 400B that doesn't.

Fabrik-Codek is a **personal cognitive architecture** that runs locally with any Ollama model. It doesn't just retrieve documents — it builds a knowledge graph from how you work, measures your expertise per topic, routes tasks to the right model with the right retrieval strategy, observes whether its responses actually helped, and refines itself over time.

## How It Works

1. **You work** — Fabrik-Codek captures code changes, session transcripts, decisions, and learnings in a local datalake
2. **Knowledge extraction** — An 11-step pipeline extracts entities and relationships into a knowledge graph alongside a vector DB
3. **Personal profiling** — Analyzes your datalake to learn your domain, stack, patterns, and tooling preferences
4. **Competence scoring** — Measures how deep your knowledge is per topic (Expert / Competent / Novice / Unknown)
5. **Adaptive routing** — Classifies each query by task type and topic, selects the right model, adapts retrieval depth, and builds a 3-layer system prompt
6. **Outcome tracking** — Infers whether responses were useful from conversational patterns (zero friction, no manual feedback)
7. **Self-correction** — Adjusts retrieval parameters for underperforming task/topic combinations

Every interaction feeds back into the system. No data leaves your machine at any point.

## Setup

Configure as an MCP server in your `openclaw.json` or `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "fabrik-codek": {
      "command": "fabrik",
      "args": ["mcp"]
    }
  }
}
```

For network access (SSE transport):

```json
{
  "mcpServers": {
    "fabrik-codek": {
      "command": "fabrik",
      "args": ["mcp", "--transport", "sse", "--port", "8421"]
    }
  }
}
```

### First Run

After installing, initialize and build the knowledge base:

```bash
fabrik init                              # Set up config, download models
fabrik graph build --include-transcripts  # Build knowledge graph from sessions
fabrik rag index                         # Index datalake into vector DB
fabrik profile build                     # Build your personal profile
fabrik competence build                  # Build competence map
```

## Available MCP Tools

### fabrik_ask

Ask a question to the local LLM with optional context from the knowledge base. The Task Router automatically classifies your query, selects the right model based on your competence, adapts retrieval strategy, and builds a personalized system prompt.

- `use_rag=true` — vector search context
- `use_graph=true` — hybrid context (vector + graph + full-text)

Example: *"How should I handle database connection pooling?"*

### fabrik_search

Semantic vector search across your accumulated knowledge. Returns the most relevant documents, patterns, and examples by meaning — not just keywords.

Example: *"Find examples of retry logic with exponential backoff"*

### fabrik_graph_search

Traverse the knowledge graph to find entities (technologies, patterns, strategies) and their relationships. Useful for understanding how concepts connect in your experience.

- `depth` — how many hops to traverse (default: 2)

Example: *"What technologies are related to FastAPI in my knowledge graph?"*

### fabrik_fulltext_search

Full-text keyword search via Meilisearch. Use this for exact keyword or phrase matching when you know the specific terms. Optional — the system works without Meilisearch installed.

Example: *"Search for 'EXPLAIN ANALYZE' in my knowledge base"*

### fabrik_graph_stats

Knowledge graph statistics: entity count, edge count, connected components, type breakdown, and relation types.

### fabrik_status

System health check: Ollama availability, RAG engine, knowledge graph, full-text search, and datalake status.

## Available MCP Resources

| URI | Description |
|-----|-------------|
| `fabrik://status` | System component status |
| `fabrik://graph/stats` | Knowledge graph statistics |
| `fabrik://config` | Current configuration (sanitized) |

## When to Use Each Tool

| Scenario | Tool | Why |
|----------|------|-----|
| Coding question needing context | `fabrik_ask` with `use_graph=true` | Gets hybrid retrieval + personalized prompt |
| Find similar patterns or examples | `fabrik_search` | Semantic similarity across all knowledge |
| Understand how concepts relate | `fabrik_graph_search` | Graph traversal shows entity relationships |
| Find exact terms or phrases | `fabrik_fulltext_search` | BM25 keyword matching |
| Check if knowledge base is healthy | `fabrik_status` | Component health check |
| Understand knowledge distribution | `fabrik_graph_stats` | Entity/edge counts and types |

## The Cognitive Loop

The system gets smarter the more you use it:

```
You work → Flywheel captures it → Pipeline extracts knowledge
    ↑                                        ↓
Strategy Optimizer ← Outcome Tracker ← LLM responds with context
    ↓                                        ↑
    └──── adjusts retrieval ──→ Task Router ─┘
                                    ↓
                  Profile + Competence + task-specific prompt
```

- **Personal Profile** learns your domain, stack, and preferences from your datalake
- **Competence Model** scores expertise per topic using 4 signals (entry count, graph density, recency, outcome rate)
- **Task Router** classifies queries into 7 task types, detects topic, selects model, adapts retrieval
- **Outcome Tracker** infers response quality from conversational patterns (topic change = accepted, reformulation = rejected)
- **Strategy Optimizer** adjusts retrieval parameters for weak spots
- **Graph Temporal Decay** fades stale knowledge, reinforces recent activity
- **Semantic Drift Detection** alerts when an entity's context shifts between graph builds

## Requirements

- [Fabrik-Codek](https://github.com/ikchain/Fabrik-Codek) installed (`pip install fabrik-codek`)
- [Ollama](https://ollama.ai/) running locally with any model (e.g., `ollama pull qwen2.5-coder:7b`)
- Optional: [Meilisearch](https://meilisearch.com/) for full-text search (system works without it)

## Security & Privacy

- **100% local**: All data stays on your machine. No external API calls, no telemetry, no cloud dependencies
- **No credentials required**: Connects only to your local Ollama instance (`localhost:11434`)
- **No external endpoints**: This skill does not contact any external services
- **Data paths**: Reads transcript files from `~/.claude/projects/*/` (local JSON already on disk). Writes to `./data/embeddings/` (vector DB), `./data/graphdb/` (knowledge graph), and `./data/profile/` (personal profile). All paths are declared in the skill metadata
- **Session reading**: `fabrik learn` is opt-in — triggered manually by the user, not automatic background surveillance. Transcripts may contain sensitive data; review before indexing
- **Network exposure**: Default transport is `stdio` (no network). SSE transport binds to `127.0.0.1` by default. If you change the bind address, ensure proper firewall/ACL rules
- **Open source**: Fully auditable at [github.com/ikchain/Fabrik-Codek](https://github.com/ikchain/Fabrik-Codek) (MIT license)
