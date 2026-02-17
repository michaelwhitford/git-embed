# git-embed

Semantic similarity search for git repositories. A self-contained git extension that embeds file content and provides vector similarity search — no external APIs, no config, no dependencies beyond the binary.

## Install

```bash
# From source
cargo build --release
cp target/release/git-embed /usr/local/bin/
```

When `git-embed` is on PATH, git discovers it automatically:
```bash
git embed search "market regime detection"
```

## Commands

```bash
git embed                          # update index for changed files
git embed update                   # explicit form of the above
git embed search "query"           # find similar content
git embed similar <file>           # find files similar to this one
git embed status                   # indexed/total, model, health
git embed gc                       # prune unreferenced embeddings
git embed clear                    # delete the entire embedding index
git embed install                  # install git hooks for auto-update
git embed uninstall                # remove git-embed hooks
```

## Automatic Updates

```bash
git embed install
```

Installs `post-commit`, `post-merge`, and `post-checkout` hooks (following the git-lfs pattern). After install, embeddings update automatically in the background after every commit, merge, or branch switch.

- Hooks are thin shell shims that delegate to `git-embed`
- If `git-embed` isn't on PATH, the hook exits silently (no blocked commits)
- Updates run in the background (`&`) so commits stay fast
- Appends to existing hooks without clobbering them
- `git embed uninstall` cleanly removes only the git-embed sections

## How It Works

**Content-addressed caching via blob SHA.** Git already content-addresses every file. git-embed maintains a mapping from `blob-sha → embedding-vector`. Same content = same SHA = never recompute. File renames without content changes = free.

**Storage: custom git ref.** The index lives at `refs/embed/v1/index` — a blob in git's object store. Not in the working tree, not in commit history. Distributes naturally:
```bash
git push origin refs/embed/v1/index
git fetch origin refs/embed/v1/index:refs/embed/v1/index
```

**Self-contained model.** Uses `nomic-embed-text-v1.5` (768-dim, Apache 2.0) via ONNX Runtime. Zero config, zero API keys. Every node produces identical embeddings. Model downloaded on first run to `~/.git-embed/models/nomic-embed-text-v1.5/`.

**Matryoshka truncation.** Vectors are stored at full 768 dimensions. At query time, truncate + renormalize for speed/precision tradeoff:
- 768 dims → full fidelity
- 256 dims → ~97% quality, ~3× faster search
- 64 dims → ~85% quality, ~8× faster search

## Performance

Single static binary (~21 MB). ~3,200 lines of Rust. 87 tests.

### Indexing (96-file Clojure project, CPU inference)

| Batch Size | Peak RSS | Wall Time | Per-doc |
|-----------|----------|-----------|---------|
| 32 (auto, ≥4 GB system) | 3.5 GB | 29s | 301 ms |
| 4 (auto, ~1.5 GB system) | 1.5 GB | 33s | 342 ms |
| 1 (auto, ~1 GB system) | 890 MB | 39s | 406 ms |

Batch size is **auto-detected from available system memory** — no configuration needed. On a 1 GB container it runs at batch-size 1 and stays under 900 MB. On a workstation it uses full batch-size 32 for maximum throughput. Override with `-b`/`--batch-size` if needed.

### Search & Similarity

| Operation | Time | Peak RSS |
|-----------|------|----------|
| `search` (model load + query embed + scan) | 307 ms | 773 MB |
| `similar` (no model needed — index only) | 6 ms | 13 MB |

Search is dominated by model load (287 ms) and query embedding (11 ms). The actual vector scan over 94 embeddings takes <1 ms.

### Micro-benchmarks (Criterion)

| Operation | Result |
|-----------|--------|
| Cosine similarity (768-dim pair) | 515 ns |
| Brute-force scan (10K embeddings @ 768d) | 5.6 ms |
| Brute-force scan (50K embeddings @ 768d) | 33 ms |
| Matryoshka scan (10K @ 64d vs 768d) | 716 µs vs 5.6 ms (8× faster) |
| Serialize 10K embeddings | 4.6 ms (~7 GiB/s) |
| Deserialize 10K embeddings | 5.7 ms (~5 GiB/s) |
| Model load | 287 ms |
| Embed query (short text) | 11 ms |
| Embed document (~200 tokens) | 72 ms |
| Batch 32 × medium docs | 1.4s (23 docs/sec) |

### Memory Profile

| Metric | Value |
|--------|-------|
| Per-embedding footprint | ~3.1 KiB (768 × f32 + SHA + overhead) |
| Index 10K files (on wire) | ~31 MiB |
| Search working set (10K files) | ~34 MiB |
| Model weights (FP32 ONNX) | 522 MiB |
| Model baseline RSS | ~750 MiB |

### Tuning Flags

```bash
git embed update -j 4              # limit to 4 inference threads
git embed update -b 1              # force batch-size 1 (low memory)
git embed update -b 16             # explicit batch size
git embed search -d 256 "query"    # Matryoshka: faster search, slightly lower quality
git embed update --time-stats      # per-phase timing breakdown
git embed update --memory-stats    # peak heap / allocation count
```

## Architecture

```
git-embed search "query"
    │
    ├── tokenize(query, prefix="search_query: ")
    ├── infer(ONNX model, tokens) → 768-dim vector
    ├── load index from refs/embed/v1/index
    ├── for each (sha, vec) in index:
    │     cosine_similarity(query_vec, vec[:dims])
    └── return top-k results with paths
```

### Index Format (binary, big-endian, Java DataOutputStream compatible)

```
[version:i32][model:java-utf(u16-len + bytes)][dims:i32][count:i32]
[sha:java-utf, float32×dims]...
```

### Model

**nomic-embed-text-v1.5** — 137M parameters, 768 dimensions, Apache 2.0 license.

Trained with Matryoshka Representation Learning (MRL) — important information is front-loaded into earlier dimensions, enabling meaningful truncation at query time.

Task prefixes (applied automatically):
- `search_document: ` — when indexing file content
- `search_query: ` — when searching

### Memory-Aware Batching

During indexing, documents are grouped into batches for ONNX inference. Each batch item allocates ~90 MiB of activation memory for transformer forward passes. git-embed detects available system memory at startup and computes the optimal batch size:

```
usable = available_memory - 800 MiB (model) - 256 MiB (headroom)
batch_size = clamp(usable / 90 MiB, 1, 32)
```

Within each batch, chunks are sorted by token count and packed using a token-budget algorithm to minimize padding waste. Long documents (>512 tokens) are automatically split at line boundaries and reassembled via weighted-average pooling.

### Version = Model

git-embed 1.x ships nomic-embed-text-v1.5, stores refs at `refs/embed/v1/index`. Upgrading the model = new ref namespace (`v2/`). Old and new coexist.

## Development

```bash
# Build
cargo build --release

# Run tests (87 tests)
cargo test

# Run benchmarks
cargo bench --bench search
cargo bench --bench index_serde
cargo bench --bench inference
cargo bench --bench memory           # standalone binary, not criterion

# Run from source
cargo run -- search "query"
cargo run -- update -v --time-stats
```

## License

MIT
