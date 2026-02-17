//! Benchmarks for similarity search operations.
//!
//! Tests the hot inner loops: cosine_similarity, truncate_and_normalize,
//! and full brute-force search at various index sizes and dimension counts.
//!
//! These represent the core query-time performance — everything the user
//! waits on after the index is loaded.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use std::collections::HashMap;

use git_embed::index::{empty_index, EmbedIndex};
use git_embed::search::{self, cosine_similarity, truncate_and_normalize};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn random_sha(rng: &mut impl Rng) -> String {
    (0..40)
        .map(|_| char::from_digit(rng.gen_range(0..16), 16).unwrap())
        .collect()
}

fn random_vec(rng: &mut impl Rng, dims: usize) -> Vec<f32> {
    (0..dims).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect()
}

fn build_index(n: usize, dims: usize) -> EmbedIndex {
    let mut rng = rand::thread_rng();
    let mut idx = empty_index();
    idx.dims = dims as i32;
    idx.embeddings = HashMap::with_capacity(n);
    for _ in 0..n {
        idx.embeddings
            .insert(random_sha(&mut rng), random_vec(&mut rng, dims));
    }
    idx
}

// ---------------------------------------------------------------------------
// Micro: cosine_similarity
// ---------------------------------------------------------------------------

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");
    let mut rng = rand::thread_rng();

    for &dims in &[64, 128, 256, 384, 512, 768] {
        let a = random_vec(&mut rng, dims);
        let b = random_vec(&mut rng, dims);

        group.throughput(Throughput::Elements(dims as u64));
        group.bench_with_input(
            BenchmarkId::new("dims", dims),
            &(a.clone(), b.clone()),
            |bench, (a, b)| {
                bench.iter(|| criterion::black_box(cosine_similarity(a, b, dims)));
            },
        );
    }

    group.finish();
}

fn bench_cosine_matryoshka(c: &mut Criterion) {
    // Full 768-dim vectors but queried at various truncation levels.
    // This is the real Matryoshka use case.
    let mut group = c.benchmark_group("cosine_matryoshka_truncation");
    let mut rng = rand::thread_rng();

    let a = random_vec(&mut rng, 768);
    let b = random_vec(&mut rng, 768);

    for &dims in &[64, 128, 256, 384, 512, 768] {
        group.bench_with_input(
            BenchmarkId::new("query_dims", dims),
            &dims,
            |bench, &dims| {
                bench.iter(|| criterion::black_box(cosine_similarity(&a, &b, dims)));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Micro: truncate_and_normalize
// ---------------------------------------------------------------------------

fn bench_truncate_and_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("truncate_and_normalize");
    let mut rng = rand::thread_rng();

    let v = random_vec(&mut rng, 768);

    for &dims in &[64, 128, 256, 768] {
        group.bench_with_input(
            BenchmarkId::new("dims", dims),
            &dims,
            |bench, &dims| {
                bench.iter(|| criterion::black_box(truncate_and_normalize(&v, dims)));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Search: brute-force scan at various index sizes
// ---------------------------------------------------------------------------

fn bench_search_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_brute_force");
    let mut rng = rand::thread_rng();

    for &n in &[100, 1_000, 10_000, 50_000] {
        let idx = build_index(n, 768);
        let query = random_vec(&mut rng, 768);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("embeddings", n),
            &(idx, query.clone()),
            |bench, (idx, query)| {
                bench.iter(|| {
                    // Simulate the optimized search: score with refs, partial sort, materialize top-k
                    let scored: Vec<(f32, &str, &str)> = idx
                        .embeddings
                        .iter()
                        .map(|(sha, vec)| (cosine_similarity(query, vec, 768), sha.as_str(), sha.as_str()))
                        .collect();
                    let results = search::top_k_results(scored, 10);
                    criterion::black_box(results);
                });
            },
        );
    }

    group.finish();
}

fn bench_search_scan_by_dims(c: &mut Criterion) {
    // Fixed 10K index, vary query dimensions (Matryoshka tradeoff)
    let mut group = c.benchmark_group("search_scan_by_dims");
    let mut rng = rand::thread_rng();

    let idx = build_index(10_000, 768);
    let query = random_vec(&mut rng, 768);

    for &dims in &[64, 128, 256, 768] {
        group.bench_with_input(
            BenchmarkId::new("dims", dims),
            &dims,
            |bench, &dims| {
                bench.iter(|| {
                    let scored: Vec<(f32, &str, &str)> = idx
                        .embeddings
                        .iter()
                        .map(|(sha, vec)| (cosine_similarity(&query, vec, dims), sha.as_str(), sha.as_str()))
                        .collect();
                    let results = search::top_k_results(scored, 10);
                    criterion::black_box(results);
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Search: sort vs partial sort
// ---------------------------------------------------------------------------

fn bench_search_sort_strategy(c: &mut Criterion) {
    // Compare full sort + truncate vs select_nth_unstable for top-k
    let mut group = c.benchmark_group("search_sort_strategy");
    let mut rng = rand::thread_rng();

    let n = 10_000;
    let idx = build_index(n, 768);
    let query = random_vec(&mut rng, 768);

    // Pre-compute scores to isolate sorting cost
    let scores: Vec<(f32, usize)> = idx
        .embeddings
        .iter()
        .enumerate()
        .map(|(i, (_, vec))| (cosine_similarity(&query, vec, 768), i))
        .collect();

    group.bench_function("full_sort_truncate", |bench| {
        bench.iter(|| {
            let mut s = scores.clone();
            s.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            s.truncate(10);
            criterion::black_box(s);
        });
    });

    group.bench_function("partial_sort_select_nth", |bench| {
        bench.iter(|| {
            let mut s = scores.clone();
            if s.len() > 10 {
                s.select_nth_unstable_by(9, |a, b| {
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                });
                s.truncate(10);
                s.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            }
            criterion::black_box(s);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Memory: estimate allocation pressure
// ---------------------------------------------------------------------------

fn bench_clone_overhead(c: &mut Criterion) {
    // In the current search code, paths are cloned per result.
    // Measure the clone cost at scale.
    let mut group = c.benchmark_group("search_clone_overhead");
    let mut rng = rand::thread_rng();

    let n = 10_000;
    let paths: Vec<String> = (0..n)
        .map(|i| format!("src/some/nested/path/file_{}.rs", i))
        .collect();
    let shas: Vec<String> = (0..n).map(|_| random_sha(&mut rng)).collect();

    group.bench_function("clone_all_paths", |bench| {
        bench.iter(|| {
            let results: Vec<(String, String)> = shas
                .iter()
                .zip(paths.iter())
                .map(|(sha, path)| (sha.clone(), path.clone()))
                .collect();
            criterion::black_box(results);
        });
    });

    group.bench_function("reference_paths", |bench| {
        bench.iter(|| {
            let results: Vec<(&str, &str)> = shas
                .iter()
                .zip(paths.iter())
                .map(|(sha, path)| (sha.as_str(), path.as_str()))
                .collect();
            criterion::black_box(results);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_cosine_matryoshka,
    bench_truncate_and_normalize,
    bench_search_scan,
    bench_search_scan_by_dims,
    bench_search_sort_strategy,
    bench_clone_overhead,
);
criterion_main!(benches);
