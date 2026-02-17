//! Memory profiling benchmarks using dhat.
//!
//! Unlike criterion benchmarks that measure time, these measure **heap allocations**:
//! total bytes allocated, number of allocations, and peak heap usage.
//!
//! dhat operates as a global allocator that instruments every alloc/dealloc,
//! so we can't use criterion here (it would measure its own overhead too).
//! Instead, this is a standalone binary that runs each scenario in isolation
//! and prints a summary table.
//!
//! Run with: cargo bench --bench memory
//! Or for full dhat profiling: cargo bench --bench memory -- --dhat

use std::collections::HashMap;

// When --dhat flag is passed, use the dhat allocator for detailed profiling.
// Otherwise we just do manual tracking via a counting allocator.

// ---------------------------------------------------------------------------
// Counting allocator — lightweight allocation tracker
// ---------------------------------------------------------------------------

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);
static CURRENT_BYTES: AtomicU64 = AtomicU64::new(0);
static TRACKING: AtomicBool = AtomicBool::new(false);

struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if TRACKING.load(Ordering::Relaxed) && !ptr.is_null() {
            let size = layout.size() as u64;
            ALLOC_BYTES.fetch_add(size, Ordering::Relaxed);
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            let current = CURRENT_BYTES.fetch_add(size, Ordering::Relaxed) + size;
            // Update peak (relaxed CAS loop)
            let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
            while current > peak {
                match PEAK_BYTES.compare_exchange_weak(
                    peak,
                    current,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACKING.load(Ordering::Relaxed) {
            CURRENT_BYTES.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn reset_tracking() {
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    PEAK_BYTES.store(0, Ordering::Relaxed);
    CURRENT_BYTES.store(0, Ordering::Relaxed);
    TRACKING.store(true, Ordering::Relaxed);
}

fn stop_tracking() -> MemStats {
    TRACKING.store(false, Ordering::Relaxed);
    MemStats {
        total_bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        alloc_count: ALLOC_COUNT.load(Ordering::Relaxed),
        peak_bytes: PEAK_BYTES.load(Ordering::Relaxed),
    }
}

#[derive(Debug, Clone)]
struct MemStats {
    total_bytes: u64,
    alloc_count: u64,
    peak_bytes: u64,
}

// ---------------------------------------------------------------------------
// Helpers (same as other benchmarks)
// ---------------------------------------------------------------------------

use git_embed::index::{deserialize_index, empty_index, serialize_index, EmbedIndex};
use git_embed::search::{self, cosine_similarity};

fn random_sha_seeded(seed: u64) -> String {
    // Simple deterministic "random" SHA for reproducibility
    let mut state = seed;
    (0..40)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = ((state >> 33) % 16) as u32;
            char::from_digit(idx, 16).unwrap()
        })
        .collect()
}

fn random_f32_seeded(seed: u64) -> f32 {
    let state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn random_vec_seeded(seed: u64, dims: usize) -> Vec<f32> {
    (0..dims)
        .map(|i| random_f32_seeded(seed.wrapping_add(i as u64)))
        .collect()
}

fn build_index(n: usize, dims: usize) -> EmbedIndex {
    let mut idx = empty_index();
    idx.dims = dims as i32;
    idx.embeddings = HashMap::with_capacity(n);
    for i in 0..n {
        idx.embeddings
            .insert(random_sha_seeded(i as u64), random_vec_seeded(i as u64 + 1_000_000, dims));
    }
    idx
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

fn fmt_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn fmt_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

/// Measure memory for deserializing an index of N embeddings.
fn measure_deserialize(n: usize, dims: usize) -> MemStats {
    // Build and serialize BEFORE we start tracking
    let idx = build_index(n, dims);
    let data = serialize_index(&idx).unwrap();
    drop(idx); // free the build-time allocations

    // Now measure only deserialization
    reset_tracking();
    let deserialized = deserialize_index(&data).unwrap();
    std::hint::black_box(&deserialized);
    let stats = stop_tracking();

    drop(deserialized);
    stats
}

/// Measure memory for serializing an index of N embeddings.
fn measure_serialize(n: usize, dims: usize) -> MemStats {
    let idx = build_index(n, dims);

    reset_tracking();
    let data = serialize_index(&idx).unwrap();
    std::hint::black_box(&data);
    let stats = stop_tracking();

    drop(data);
    drop(idx);
    stats
}

/// Measure memory for the scoring + top-k phase of search.
fn measure_search_scoring(n: usize, dims: usize, top: usize) -> MemStats {
    let idx = build_index(n, dims);
    let query = random_vec_seeded(999_999, dims);

    // Simulate sha_paths map (built before tracking in real code too,
    // but we want to measure it since it's part of every search)
    let paths: Vec<(String, String)> = (0..n)
        .map(|i| {
            (
                random_sha_seeded(i as u64),
                format!("src/path/to/file_{}.rs", i),
            )
        })
        .collect();
    let sha_paths: HashMap<String, Vec<String>> = {
        let mut map = HashMap::new();
        for (sha, path) in &paths {
            map.entry(sha.clone()).or_insert_with(Vec::new).push(path.clone());
        }
        map
    };

    reset_tracking();

    // Score all candidates (the hot path)
    let scored: Vec<(f32, &str, &str)> = idx
        .embeddings
        .iter()
        .flat_map(|(sha, vec)| {
            let score = cosine_similarity(&query, vec, dims);
            sha_paths
                .get(sha.as_str())
                .into_iter()
                .flat_map(move |p| p.iter().map(move |path| (score, sha.as_str(), path.as_str())))
        })
        .collect();

    let results = search::top_k_results(scored, top);
    std::hint::black_box(&results);

    let stats = stop_tracking();

    drop(results);
    stats
}

/// Measure memory for building the blob_sha_to_paths map.
fn measure_sha_paths_map(n: usize) -> MemStats {
    // Simulate the entries that would come from git::walk_tree
    let entries: Vec<(String, String)> = (0..n)
        .map(|i| {
            (
                random_sha_seeded(i as u64),
                format!("src/module/subdir/file_{}.rs", i),
            )
        })
        .collect();

    reset_tracking();

    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    for (sha, path) in &entries {
        map.entry(sha.clone()).or_insert_with(Vec::new).push(path.clone());
    }

    std::hint::black_box(&map);
    let stats = stop_tracking();

    drop(map);
    stats
}

/// Measure memory footprint of the EmbedIndex HashMap itself.
fn measure_index_footprint(n: usize, dims: usize) -> MemStats {
    reset_tracking();
    let idx = build_index(n, dims);
    std::hint::black_box(&idx);
    let stats = stop_tracking();
    drop(idx);
    stats
}

/// Measure per-embedding byte cost (total heap / count).
fn per_embedding_cost(n: usize, dims: usize) -> (u64, u64) {
    reset_tracking();
    let idx = build_index(n, dims);
    std::hint::black_box(&idx);
    let stats = stop_tracking();
    drop(idx);
    (stats.peak_bytes, stats.peak_bytes / n as u64)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn print_header(title: &str) {
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  {}", title);
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(
        "{:<40} {:>14} {:>14} {:>14}",
        "Scenario", "Total Alloc", "Alloc Count", "Peak Heap"
    );
    println!("{}", "─".repeat(82));
}

fn print_row(label: &str, stats: &MemStats) {
    println!(
        "{:<40} {:>14} {:>14} {:>14}",
        label,
        fmt_bytes(stats.total_bytes),
        fmt_count(stats.alloc_count),
        fmt_bytes(stats.peak_bytes),
    );
}

fn main() {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║              git-embed Memory Profile (heap allocations)             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");

    // ---- Index in-memory footprint ----
    print_header("Index In-Memory Footprint (HashMap<String, Vec<f32>>)");
    for &n in &[100, 1_000, 10_000, 50_000] {
        let stats = measure_index_footprint(n, 768);
        print_row(&format!("{:>6} embeddings × 768d", n), &stats);
    }

    // Per-embedding cost analysis
    println!();
    println!("  Per-embedding heap cost at 768 dims:");
    for &n in &[100, 1_000, 10_000, 50_000] {
        let (total, per) = per_embedding_cost(n, 768);
        println!(
            "    {:>6} entries: {} total, ~{} per embedding",
            n,
            fmt_bytes(total),
            fmt_bytes(per)
        );
    }

    // ---- Deserialization ----
    print_header("Deserialization (wire bytes → HashMap)");
    for &n in &[100, 1_000, 10_000, 50_000] {
        let stats = measure_deserialize(n, 768);
        print_row(&format!("deserialize {:>6} × 768d", n), &stats);
    }

    // ---- Serialization ----
    print_header("Serialization (HashMap → wire bytes)");
    for &n in &[100, 1_000, 10_000, 50_000] {
        let stats = measure_serialize(n, 768);
        print_row(&format!("serialize {:>6} × 768d", n), &stats);
    }

    // ---- Matryoshka dimension comparison ----
    print_header("Matryoshka: Index Footprint at Different Dimensions");
    for &dims in &[64, 128, 256, 384, 512, 768] {
        let stats = measure_index_footprint(10_000, dims);
        print_row(&format!("10K embeddings × {}d", dims), &stats);
    }

    // ---- Search scoring ----
    print_header("Search Scoring + Top-K (score all candidates, select top-k)");
    for &n in &[100, 1_000, 10_000, 50_000] {
        let stats = measure_search_scoring(n, 768, 10);
        print_row(&format!("score {:>6} → top 10", n), &stats);
    }

    // Different top-k values at 10K
    println!();
    println!("  Top-K variation (10K embeddings, 768d):");
    println!(
        "  {:<36} {:>14} {:>14} {:>14}",
        "", "Total Alloc", "Alloc Count", "Peak Heap"
    );
    println!("  {}", "─".repeat(78));
    for &top in &[5, 10, 25, 50, 100] {
        let stats = measure_search_scoring(10_000, 768, top);
        println!(
            "  {:<36} {:>14} {:>14} {:>14}",
            format!("score 10K → top {}", top),
            fmt_bytes(stats.total_bytes),
            fmt_count(stats.alloc_count),
            fmt_bytes(stats.peak_bytes),
        );
    }

    // ---- SHA-to-paths map ----
    print_header("blob_sha_to_paths Map (rebuilt every search)");
    for &n in &[100, 1_000, 10_000, 50_000] {
        let stats = measure_sha_paths_map(n);
        print_row(&format!("{:>6} tree entries", n), &stats);
    }

    // ---- Summary ----
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Summary: Typical Search Memory Profile (10K files, 768d, top 10)");
    println!("═══════════════════════════════════════════════════════════════════════");

    let idx_stats = measure_index_footprint(10_000, 768);
    let deser_stats = measure_deserialize(10_000, 768);
    let sha_stats = measure_sha_paths_map(10_000);
    let search_stats = measure_search_scoring(10_000, 768, 10);

    println!(
        "  Index footprint (resident):     {:>14}  (peak heap)",
        fmt_bytes(idx_stats.peak_bytes)
    );
    println!(
        "  Deserialization:                {:>14}  (total alloc'd, {} allocs)",
        fmt_bytes(deser_stats.total_bytes),
        fmt_count(deser_stats.alloc_count)
    );
    println!(
        "  SHA→paths map:                  {:>14}  (peak heap, rebuilt each search)",
        fmt_bytes(sha_stats.peak_bytes)
    );
    println!(
        "  Search scoring + top-k:         {:>14}  (total alloc'd, {} allocs)",
        fmt_bytes(search_stats.total_bytes),
        fmt_count(search_stats.alloc_count)
    );
    println!(
        "  Estimated total per-search:     {:>14}  (index + sha_map + scoring)",
        fmt_bytes(idx_stats.peak_bytes + sha_stats.peak_bytes + search_stats.peak_bytes)
    );

    println!();
    println!("Note: ONNX model session (~100s MB) is not measured here — it requires");
    println!("the model files to be present. Run `cargo bench --bench inference` for");
    println!("model load timing. Use `dhat` profiler for detailed heap analysis:");
    println!("  DHAT_LOG=dhat-heap.json cargo bench --bench memory");
    println!();
}
