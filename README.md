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
- 256 dims → ~97% quality
- 64 dims → ~85% quality

## Performance

Single static binary (~21MB). Instant startup (~7ms). On an 87-file repo:

| Metric | Value |
|--------|-------|
| Index time | ~27s (CPU ONNX inference) |
| Startup | ~7ms |
| Search | <100ms |
| Memory | ~500MB peak |

Long documents are automatically chunked at 512 tokens (the model's training context) and batched for throughput. Token-budget batching prevents OOM from padding waste.

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

### Version = Model

git-embed 1.x ships nomic-embed-text-v1.5, stores refs at `refs/embed/v1/index`. Upgrading the model = new ref namespace (`v2/`). Old and new coexist.

## Development

```bash
# Build
cargo build --release

# Run tests
cargo test

# Run from source
cargo run -- search "query"
cargo run -- update -v
```

## License

MIT
