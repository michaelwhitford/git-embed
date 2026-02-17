//! Benchmarks for ONNX inference and tokenization.
//!
//! These benchmarks require the model to be downloaded (~500MB).
//! Run `git-embed update` once in any repo to trigger the download,
//! or the benchmarks will download it on first run.
//!
//! NOTE: These are inherently slow (ML inference on CPU). Criterion
//! will adjust sample sizes automatically. Consider running with:
//!
//!   cargo bench --bench inference -- --sample-size 10
//!
//! For quick iteration during optimization.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::PathBuf;
use std::time::Duration;

use git_embed::model::{EmbedModel, MODEL_NAME};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".git-embed")
        .join("models")
        .join(MODEL_NAME)
}

fn model_available() -> bool {
    let dir = model_dir();
    dir.join("model.onnx").exists() && dir.join("tokenizer.json").exists()
}

/// Load model or skip benchmarks if not downloaded.
fn load_model() -> Option<EmbedModel> {
    if !model_available() {
        eprintln!(
            "⚠ Model not found at {:?}. Run `git-embed update` first to download.",
            model_dir()
        );
        eprintln!("  Skipping inference benchmarks.");
        return None;
    }
    EmbedModel::load().ok()
}

/// Sample texts of various lengths for benchmarking.
fn sample_short() -> &'static str {
    "Implement a binary search tree in Rust with insert, delete, and search operations."
}

fn sample_medium() -> String {
    // ~200 tokens: a realistic function-sized code chunk
    r#"
use std::collections::HashMap;

/// A simple in-memory cache with TTL-based expiration.
pub struct Cache<V> {
    entries: HashMap<String, (V, std::time::Instant)>,
    ttl: std::time::Duration,
}

impl<V: Clone> Cache<V> {
    pub fn new(ttl: std::time::Duration) -> Self {
        Self {
            entries: HashMap::new(),
            ttl,
        }
    }

    pub fn get(&self, key: &str) -> Option<V> {
        self.entries.get(key).and_then(|(v, inserted)| {
            if inserted.elapsed() < self.ttl {
                Some(v.clone())
            } else {
                None
            }
        })
    }

    pub fn set(&mut self, key: String, value: V) {
        self.entries.insert(key, (value, std::time::Instant::now()));
    }

    pub fn evict_expired(&mut self) {
        self.entries.retain(|_, (_, inserted)| inserted.elapsed() < self.ttl);
    }
}
"#.to_string()
}

fn sample_long() -> String {
    // ~800 tokens: will trigger chunking (> CHUNK_THRESHOLD of 512)
    let medium = sample_medium();
    format!("{}\n{}\n{}\n{}", medium, medium, medium, medium)
}

fn sample_very_long() -> String {
    // ~3200 tokens: multiple chunks, tests batching within a single doc
    let medium = sample_medium();
    let mut s = String::new();
    for _ in 0..16 {
        s.push_str(&medium);
        s.push('\n');
    }
    s
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

fn bench_model_load(c: &mut Criterion) {
    if !model_available() {
        eprintln!("  Skipping model_load benchmark (model not downloaded)");
        return;
    }

    let mut group = c.benchmark_group("model_load");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("load_from_disk", |bench| {
        bench.iter(|| {
            let model = EmbedModel::load_from(&model_dir()).unwrap();
            criterion::black_box(model);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Single inference at various lengths
// ---------------------------------------------------------------------------

fn bench_embed_single(c: &mut Criterion) {
    let mut model = match load_model() {
        Some(m) => m,
        None => return,
    };

    let mut group = c.benchmark_group("embed_single");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    // Query embedding (typically short)
    group.bench_function("query_short", |bench| {
        bench.iter(|| {
            let v = model.embed_query(sample_short()).unwrap();
            criterion::black_box(v);
        });
    });

    // Document embedding at various sizes
    let medium = sample_medium();
    group.bench_function("document_medium", |bench| {
        bench.iter(|| {
            let v = model.embed_document(&medium).unwrap();
            criterion::black_box(v);
        });
    });

    let long = sample_long();
    group.bench_function("document_long_chunked", |bench| {
        bench.iter(|| {
            let v = model.embed_document(&long).unwrap();
            criterion::black_box(v);
        });
    });

    let very_long = sample_very_long();
    group.bench_function("document_very_long_chunked", |bench| {
        bench.iter(|| {
            let v = model.embed_document(&very_long).unwrap();
            criterion::black_box(v);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Batch inference
// ---------------------------------------------------------------------------

fn bench_embed_batch(c: &mut Criterion) {
    let mut model = match load_model() {
        Some(m) => m,
        None => return,
    };

    let mut group = c.benchmark_group("embed_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let medium = sample_medium();

    // Batch of identical short texts (best case for padding)
    for &batch_size in &[1, 4, 8, 16, 32] {
        let texts: Vec<&str> = vec![sample_short(); batch_size];
        group.bench_with_input(
            BenchmarkId::new("short_uniform", batch_size),
            &texts,
            |bench, texts| {
                bench.iter(|| {
                    let v = model.embed_documents(texts).unwrap();
                    criterion::black_box(v);
                });
            },
        );
    }

    // Batch of mixed lengths (realistic: mix of short, medium, long)
    let long = sample_long();
    let mixed: Vec<&str> = vec![
        sample_short(),
        &medium,
        sample_short(),
        &medium,
        &long,
        sample_short(),
        &medium,
        &long,
    ];

    group.bench_function("mixed_lengths_8", |bench| {
        bench.iter(|| {
            let v = model.embed_documents(&mixed).unwrap();
            criterion::black_box(v);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Throughput: documents per second
// ---------------------------------------------------------------------------

fn bench_throughput(c: &mut Criterion) {
    let mut model = match load_model() {
        Some(m) => m,
        None => return,
    };

    let mut group = c.benchmark_group("throughput");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let medium = sample_medium();
    let batch: Vec<&str> = vec![medium.as_str(); 32];

    group.bench_function("32_medium_docs", |bench| {
        bench.iter(|| {
            let v = model.embed_documents(&batch).unwrap();
            criterion::black_box(v);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_model_load,
    bench_embed_single,
    bench_embed_batch,
    bench_throughput,
);
criterion_main!(benches);
