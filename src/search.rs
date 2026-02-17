use std::collections::HashMap;

use anyhow::Result;
use git2::Repository;

use crate::git;
use crate::index::EmbedIndex;

// ---- Types ----

#[derive(Debug)]
pub struct SearchResult {
    pub score: f32,
    pub path: String,
    pub blob_sha: String,
}

pub struct SearchOpts {
    pub dims: usize,
    pub top: usize,
    pub path: Option<String>,
}

impl Default for SearchOpts {
    fn default() -> Self {
        Self {
            dims: 768,
            top: 10,
            path: None,
        }
    }
}

// ---- Similarity math ----

/// Cosine similarity between two vectors, truncated to `dims` dimensions.
pub fn cosine_similarity(a: &[f32], b: &[f32], dims: usize) -> f32 {
    let n = dims.min(a.len()).min(b.len());

    let mut dot: f32 = 0.0;
    let mut na: f32 = 0.0;
    let mut nb: f32 = 0.0;

    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }

    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Truncate a vector to `dims` dimensions and L2-normalize it.
pub fn truncate_and_normalize(v: &[f32], dims: usize) -> Vec<f32> {
    let n = dims.min(v.len());
    let slice = &v[..n];

    let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm == 0.0 {
        slice.to_vec()
    } else {
        slice.iter().map(|x| x / norm).collect()
    }
}

// ---- Search ----

/// Comparator for descending f32 scores.
fn cmp_score_desc(a: &f32, b: &f32) -> std::cmp::Ordering {
    b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
}

/// Build a blob SHA → set-of-paths map from the repository HEAD tree.
fn blob_sha_to_paths(repo: &Repository) -> HashMap<String, Vec<String>> {
    let entries = git::walk_tree(repo).unwrap_or_default();
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    for entry in entries {
        map.entry(entry.blob_sha)
            .or_default()
            .push(entry.path);
    }
    map
}

/// Select the top-k results from scored candidates using partial sort.
///
/// Uses `select_nth_unstable_by` (O(n)) instead of full sort (O(n log n)),
/// then sorts only the top-k slice. Only materializes owned `SearchResult`s
/// for the final winners — all scoring is done with borrowed references.
pub fn top_k_results<'a>(
    mut scored: Vec<(f32, &'a str, &'a str)>, // (score, sha, path)
    top: usize,
) -> Vec<SearchResult> {
    if scored.is_empty() {
        return Vec::new();
    }

    let k = top.min(scored.len());

    if k < scored.len() {
        // Partial sort: partition so the top-k are in scored[..k] (unordered)
        scored.select_nth_unstable_by(k - 1, |a, b| cmp_score_desc(&a.0, &b.0));
        scored.truncate(k);
    }

    // Sort only the top-k for final ordering
    scored.sort_unstable_by(|a, b| cmp_score_desc(&a.0, &b.0));

    // Materialize owned results only for the winners
    scored
        .into_iter()
        .map(|(score, sha, path)| SearchResult {
            score,
            path: path.to_string(),
            blob_sha: sha.to_string(),
        })
        .collect()
}

/// Search the index for vectors most similar to `query_vec`.
///
/// Returns up to `opts.top` results sorted by descending cosine similarity.
/// If `opts.path` is set, only results whose path starts with that prefix are included.
pub fn search(
    repo: &Repository,
    index: &EmbedIndex,
    query_vec: &[f32],
    opts: &SearchOpts,
) -> Vec<SearchResult> {
    let sha_paths = blob_sha_to_paths(repo);

    // Score all candidates with borrowed references — no allocations per candidate
    let scored: Vec<(f32, &str, &str)> = index
        .embeddings
        .iter()
        .flat_map(|(sha, vec)| {
            let score = cosine_similarity(query_vec, vec, opts.dims);
            sha_paths
                .get(sha.as_str())
                .into_iter()
                .flat_map(move |paths| {
                    paths.iter().map(move |path| (score, sha.as_str(), path.as_str()))
                })
        })
        .filter(|(_, _, path)| match &opts.path {
            Some(prefix) => path.starts_with(prefix.as_str()),
            None => true,
        })
        .collect();

    top_k_results(scored, opts.top)
}

/// Find files similar to the file at `file_path`.
///
/// Looks up the blob SHA for `file_path` in the HEAD tree, retrieves its
/// embedding from the index, and scores all other indexed blobs against it.
/// Returns up to `opts.top` results sorted by descending cosine similarity.
pub fn similar(
    repo: &Repository,
    index: &EmbedIndex,
    file_path: &str,
    opts: &SearchOpts,
) -> Result<Vec<SearchResult>> {
    // Find the blob SHA for the target file in the current tree
    let entries = git::walk_tree(repo)?;
    let target = entries
        .iter()
        .find(|e| e.path == file_path)
        .ok_or_else(|| anyhow::anyhow!("File not found in tree: {}", file_path))?;

    let target_sha = &target.blob_sha;

    // Get the embedding for the target file
    let target_vec = index
        .embeddings
        .get(target_sha)
        .ok_or_else(|| anyhow::anyhow!("File not indexed: {}", file_path))?;

    let sha_paths = blob_sha_to_paths(repo);

    // Score all candidates with borrowed references — no allocations per candidate
    let scored: Vec<(f32, &str, &str)> = index
        .embeddings
        .iter()
        .filter(|(sha, _)| *sha != target_sha)
        .flat_map(|(sha, vec)| {
            let score = cosine_similarity(target_vec, vec, opts.dims);
            sha_paths
                .get(sha.as_str())
                .into_iter()
                .flat_map(move |paths| {
                    paths.iter().map(move |path| (score, sha.as_str(), path.as_str()))
                })
        })
        .filter(|(_, _, path)| match &opts.path {
            Some(prefix) => path.starts_with(prefix.as_str()),
            None => true,
        })
        .collect();

    Ok(top_k_results(scored, opts.top))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- cosine_similarity ---

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v, 3);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b, 2);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b, 2);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 0.0];
        let sim = cosine_similarity(&a, &b, 2);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_truncates_to_dims() {
        let a = vec![1.0, 0.0, 999.0];
        let b = vec![1.0, 0.0, -999.0];
        // With dims=2 the third element is ignored, so they're identical
        let sim = cosine_similarity(&a, &b, 2);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_dims_zero() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let sim = cosine_similarity(&a, &b, 0);
        assert_eq!(sim, 0.0); // no elements → zero denom
    }

    #[test]
    fn test_cosine_similarity_mismatched_lengths() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0];
        let sim = cosine_similarity(&a, &b, 3);
        // min(3, 3, 2) = 2, so only first 2 elements compared
        assert!((sim - 1.0).abs() < 1e-6);
    }

    // --- truncate_and_normalize ---

    #[test]
    fn test_truncate_and_normalize() {
        let v = vec![3.0, 4.0, 100.0];
        let normed = truncate_and_normalize(&v, 2);
        assert_eq!(normed.len(), 2);
        // 3/5, 4/5
        assert!((normed[0] - 0.6).abs() < 1e-6);
        assert!((normed[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_truncate_and_normalize_zero() {
        let v = vec![0.0, 0.0];
        let normed = truncate_and_normalize(&v, 2);
        assert_eq!(normed, vec![0.0, 0.0]);
    }

    #[test]
    fn test_truncate_and_normalize_dims_larger_than_vec() {
        let v = vec![3.0, 4.0];
        let normed = truncate_and_normalize(&v, 100);
        assert_eq!(normed.len(), 2); // clamped to v.len()
        assert!((normed[0] - 0.6).abs() < 1e-6);
    }

    // --- Integration: search and similar against temp repo ---

    fn temp_repo_with_index(
        files: &[(&str, &str)],
        embeddings: &[(&str, &str, Vec<f32>)], // (sha_hint, path, vec)
    ) -> (tempfile::TempDir, Repository, EmbedIndex) {
        let dir = tempfile::tempdir().unwrap();
        let repo = Repository::init(dir.path()).unwrap();

        let tree_oid = {
            let mut tb = repo.treebuilder(None).unwrap();
            for &(path, content) in files {
                let oid = repo.blob(content.as_bytes()).unwrap();
                tb.insert(path, oid, 0o100644).unwrap();
            }
            tb.write().unwrap()
        };

        {
            let tree = repo.find_tree(tree_oid).unwrap();
            let sig = git2::Signature::now("test", "test@test.com").unwrap();
            repo.commit(Some("HEAD"), &sig, &sig, "init", &tree, &[]).unwrap();
        }

        // Build index: look up actual blob SHAs from the tree
        let entries = git::walk_tree(&repo).unwrap();
        let mut idx = crate::index::empty_index();
        idx.dims = 3; // small dims for testing

        for &(sha_hint, path, ref vec) in embeddings {
            let actual_sha = entries
                .iter()
                .find(|e| e.path == path)
                .map(|e| e.blob_sha.clone())
                .unwrap_or_else(|| sha_hint.to_string());
            idx.embeddings.insert(actual_sha, vec.clone());
        }

        (dir, repo, idx)
    }

    #[test]
    fn test_search_returns_sorted_results() {
        let (_dir, repo, idx) = temp_repo_with_index(
            &[
                ("a.txt", "alpha"),
                ("b.txt", "beta"),
                ("c.txt", "gamma"),
            ],
            &[
                ("", "a.txt", vec![1.0, 0.0, 0.0]),
                ("", "b.txt", vec![0.9, 0.1, 0.0]),
                ("", "c.txt", vec![0.0, 1.0, 0.0]),
            ],
        );

        let query = vec![1.0, 0.0, 0.0];
        let opts = SearchOpts { dims: 3, top: 10, path: None };
        let results = search(&repo, &idx, &query, &opts);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].path, "a.txt"); // most similar
        assert!((results[0].score - 1.0).abs() < 1e-6);
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);
    }

    #[test]
    fn test_search_respects_top() {
        let (_dir, repo, idx) = temp_repo_with_index(
            &[
                ("a.txt", "alpha"),
                ("b.txt", "beta"),
                ("c.txt", "gamma"),
            ],
            &[
                ("", "a.txt", vec![1.0, 0.0, 0.0]),
                ("", "b.txt", vec![0.9, 0.1, 0.0]),
                ("", "c.txt", vec![0.0, 1.0, 0.0]),
            ],
        );

        let query = vec![1.0, 0.0, 0.0];
        let opts = SearchOpts { dims: 3, top: 1, path: None };
        let results = search(&repo, &idx, &query, &opts);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "a.txt");
    }

    #[test]
    fn test_search_path_filter() {
        let (_dir, repo, idx) = temp_repo_with_index(
            &[
                ("a.txt", "alpha"),
                ("b.txt", "beta"),
            ],
            &[
                ("", "a.txt", vec![1.0, 0.0, 0.0]),
                ("", "b.txt", vec![1.0, 0.0, 0.0]),
            ],
        );

        let query = vec![1.0, 0.0, 0.0];
        let opts = SearchOpts { dims: 3, top: 10, path: Some("a".to_string()) };
        let results = search(&repo, &idx, &query, &opts);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "a.txt");
    }

    #[test]
    fn test_search_empty_index() {
        let dir = tempfile::tempdir().unwrap();
        let repo = Repository::init(dir.path()).unwrap();
        let idx = crate::index::empty_index();

        let query = vec![1.0, 0.0, 0.0];
        let opts = SearchOpts::default();
        let results = search(&repo, &idx, &query, &opts);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_stale_embedding_excluded() {
        // Index has an embedding for a SHA not in the tree
        let (_dir, repo, mut idx) = temp_repo_with_index(
            &[("a.txt", "alpha")],
            &[("", "a.txt", vec![1.0, 0.0, 0.0])],
        );
        idx.embeddings.insert("deadbeef00000000000000000000000000000000".to_string(), vec![1.0, 0.0, 0.0]);

        let query = vec![1.0, 0.0, 0.0];
        let opts = SearchOpts { dims: 3, top: 10, path: None };
        let results = search(&repo, &idx, &query, &opts);
        // Only the real file should appear, not the stale SHA
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "a.txt");
    }

    // --- similar ---

    #[test]
    fn test_similar_excludes_self() {
        let (_dir, repo, idx) = temp_repo_with_index(
            &[
                ("a.txt", "alpha"),
                ("b.txt", "beta"),
            ],
            &[
                ("", "a.txt", vec![1.0, 0.0, 0.0]),
                ("", "b.txt", vec![0.9, 0.1, 0.0]),
            ],
        );

        let opts = SearchOpts { dims: 3, top: 10, path: None };
        let results = similar(&repo, &idx, "a.txt", &opts).unwrap();
        // Should not include a.txt itself
        assert!(results.iter().all(|r| r.path != "a.txt"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "b.txt");
    }

    #[test]
    fn test_similar_file_not_found() {
        let (_dir, repo, idx) = temp_repo_with_index(
            &[("a.txt", "alpha")],
            &[("", "a.txt", vec![1.0, 0.0, 0.0])],
        );

        let opts = SearchOpts::default();
        let result = similar(&repo, &idx, "nonexistent.txt", &opts);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_similar_file_not_indexed() {
        let (_dir, repo, idx) = temp_repo_with_index(
            &[
                ("a.txt", "alpha"),
                ("b.txt", "beta"),
            ],
            &[
                ("", "a.txt", vec![1.0, 0.0, 0.0]),
                // b.txt exists in tree but NOT in index
            ],
        );

        let opts = SearchOpts::default();
        let result = similar(&repo, &idx, "b.txt", &opts);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not indexed"));
    }
}
