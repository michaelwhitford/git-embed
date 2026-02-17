//! Benchmarks for index serialization and deserialization.
//!
//! Tests performance of serialize_index / deserialize_index at various
//! index sizes (100, 1K, 10K, 50K embeddings) with 768-dim vectors.
//!
//! These are called on every single command (update, search, gc, status),
//! so their performance directly impacts user-perceived latency.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use std::collections::HashMap;

use git_embed::index::{deserialize_index, empty_index, serialize_index, EmbedIndex};

/// Generate a random 40-char hex SHA.
fn random_sha(rng: &mut impl Rng) -> String {
    (0..40)
        .map(|_| {
            let idx = rng.gen_range(0..16);
            char::from_digit(idx, 16).unwrap()
        })
        .collect()
}

/// Generate a random 768-dim embedding vector with values in [-1, 1].
fn random_embedding(rng: &mut impl Rng, dims: usize) -> Vec<f32> {
    (0..dims).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect()
}

/// Build a synthetic index with `n` embeddings of `dims` dimensions.
fn build_index(n: usize, dims: usize) -> EmbedIndex {
    let mut rng = rand::thread_rng();
    let mut idx = empty_index();
    idx.dims = dims as i32;
    idx.embeddings = HashMap::with_capacity(n);
    for _ in 0..n {
        idx.embeddings
            .insert(random_sha(&mut rng), random_embedding(&mut rng, dims));
    }
    idx
}

fn bench_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialize_index");

    for &n in &[100, 1_000, 10_000, 50_000] {
        let idx = build_index(n, 768);
        let bytes = n * (2 + 40 + 768 * 4); // approximate wire size
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &idx, |b, idx| {
            b.iter(|| {
                let data = serialize_index(idx).unwrap();
                criterion::black_box(data);
            });
        });
    }

    group.finish();
}

fn bench_deserialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("deserialize_index");

    for &n in &[100, 1_000, 10_000, 50_000] {
        let idx = build_index(n, 768);
        let data = serialize_index(&idx).unwrap();
        let bytes = data.len();
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &data, |b, data| {
            b.iter(|| {
                let idx = deserialize_index(data).unwrap();
                criterion::black_box(idx);
            });
        });
    }

    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_roundtrip");

    for &n in &[100, 1_000, 10_000] {
        let idx = build_index(n, 768);
        group.bench_with_input(BenchmarkId::from_parameter(n), &idx, |b, idx| {
            b.iter(|| {
                let data = serialize_index(idx).unwrap();
                let idx2 = deserialize_index(&data).unwrap();
                criterion::black_box(idx2);
            });
        });
    }

    group.finish();
}

fn bench_serialize_size(c: &mut Criterion) {
    // Measure how serialized size scales — useful for understanding I/O costs.
    let mut group = c.benchmark_group("serialize_output_size");

    for &n in &[100, 1_000, 10_000, 50_000] {
        let idx = build_index(n, 768);
        group.bench_with_input(BenchmarkId::from_parameter(n), &idx, |b, idx| {
            b.iter(|| {
                let data = serialize_index(idx).unwrap();
                criterion::black_box(data.len());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_serialize,
    bench_deserialize,
    bench_roundtrip,
    bench_serialize_size,
);
criterion_main!(benches);
