//! Git operations — repository introspection, blob reading, ref management.
//!
//! Uses git2 (libgit2) for all git operations.

use std::collections::HashMap;
use std::env;

use anyhow::{Context, Result};
use git2::{ObjectType, Repository, TreeWalkMode, TreeWalkResult};

/// The ref path where the embedding index is stored.
const EMBED_REF: &str = "refs/embed/v1/index";

/// A blob entry from the repository tree.
pub struct TreeEntry {
    pub path: String,
    pub blob_sha: String,
}

/// Known text file extensions (lowercase, with leading dot).
const TEXT_EXTENSIONS: &[&str] = &[
    ".md", ".txt", ".clj", ".cljs", ".cljc", ".edn", ".bb",
    ".py", ".rb", ".rs", ".go", ".java", ".kt", ".scala",
    ".js", ".ts", ".jsx", ".tsx", ".css", ".html", ".xml",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".zsh", ".fish",
    ".sql", ".graphql", ".proto",
    ".org", ".rst", ".adoc", ".tex",
    ".el", ".lisp", ".scm", ".rkt", ".hs", ".ml", ".ex", ".exs",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".swift", ".r", ".lua",
    ".vim", ".nix", ".dhall",
    ".dockerfile", ".tf", ".hcl",
    ".gitignore", ".gitattributes", ".editorconfig",
];

/// Find the git repository from the given path or current directory.
///
/// Checks `GIT_EMBED_CWD` env var first, then falls back to the given path
/// or the current working directory. Uses `Repository::discover()` to walk
/// up and find the `.git` directory.
pub fn find_repo(path: Option<&str>) -> Result<Repository> {
    let start = match path {
        Some(p) => p.to_string(),
        None => env::var("GIT_EMBED_CWD")
            .unwrap_or_else(|_| env::current_dir()
                .expect("failed to get current directory")
                .to_string_lossy()
                .into_owned()),
    };
    Repository::discover(&start)
        .with_context(|| format!("failed to find git repository from '{}'", start))
}

/// Walk the HEAD tree recursively, returning all blob entries.
///
/// Returns an empty `Vec` if HEAD does not exist (empty repository).
pub fn walk_tree(repo: &Repository) -> Result<Vec<TreeEntry>> {
    let head = match repo.head() {
        Ok(h) => h,
        Err(e) if e.code() == git2::ErrorCode::UnbornBranch
              || e.code() == git2::ErrorCode::NotFound => {
            return Ok(Vec::new());
        }
        Err(e) => return Err(e).context("failed to resolve HEAD"),
    };

    let commit = head.peel_to_commit()
        .context("failed to peel HEAD to commit")?;
    let tree = commit.tree()
        .context("failed to get tree from HEAD commit")?;

    let mut entries = Vec::new();

    tree.walk(TreeWalkMode::PreOrder, |dir, entry| {
        if entry.kind() == Some(ObjectType::Blob) {
            let name = entry.name().unwrap_or("");
            let full_path = if dir.is_empty() {
                name.to_string()
            } else {
                format!("{}{}", dir, name)
            };
            entries.push(TreeEntry {
                path: full_path,
                blob_sha: entry.id().to_string(),
            });
        }
        TreeWalkResult::Ok
    }).context("failed to walk tree")?;

    Ok(entries)
}

/// Heuristic: is this file path likely a text file?
///
/// Checks the file extension against a known set of text extensions.
/// Files with no extension are assumed to be text. Comparison is
/// case-insensitive.
pub fn text_blob(path: &str) -> bool {
    match path.rfind('.') {
        Some(dot_idx) => {
            let ext = path[dot_idx..].to_lowercase();
            TEXT_EXTENSIONS.contains(&ext.as_str())
        }
        // No extension → assume text
        None => true,
    }
}

/// Read a blob by its SHA hex string as a UTF-8 string.
pub fn read_blob(repo: &Repository, sha: &str) -> Result<String> {
    let oid = git2::Oid::from_str(sha)
        .with_context(|| format!("invalid SHA: '{}'", sha))?;
    let blob = repo.find_blob(oid)
        .with_context(|| format!("blob not found: {}", sha))?;
    let content = std::str::from_utf8(blob.content())
        .with_context(|| format!("blob {} is not valid UTF-8", sha))?;
    Ok(content.to_string())
}

/// Read the embed index ref blob bytes.
///
/// Returns `None` if the ref does not exist.
pub fn read_ref(repo: &Repository) -> Result<Option<Vec<u8>>> {
    let reference = match repo.find_reference(EMBED_REF) {
        Ok(r) => r,
        Err(e) if e.code() == git2::ErrorCode::NotFound => return Ok(None),
        Err(e) => return Err(e).context("failed to read embed ref"),
    };

    let oid = reference
        .target()
        .context("embed ref is not a direct reference")?;

    let blob = repo.find_blob(oid)
        .context("failed to read blob for embed ref")?;

    Ok(Some(blob.content().to_vec()))
}

/// Write data as a blob and update the embed ref to point to it.
///
/// Force-updates the ref regardless of its current value.
pub fn write_ref(repo: &Repository, data: &[u8]) -> Result<()> {
    let oid = repo.blob(data)
        .context("failed to write blob")?;

    repo.reference(
        EMBED_REF,
        oid,
        true, // force
        &format!("git-embed: update index → {}", oid),
    ).context("failed to update embed ref")?;

    Ok(())
}

/// Delete the embed ref.
///
/// Returns `true` if the ref was deleted, `false` if it did not exist.
pub fn delete_ref(repo: &Repository) -> Result<bool> {
    let mut reference = match repo.find_reference(EMBED_REF) {
        Ok(r) => r,
        Err(e) if e.code() == git2::ErrorCode::NotFound => return Ok(false),
        Err(e) => return Err(e).context("failed to look up embed ref"),
    };

    reference.delete()
        .context("failed to delete embed ref")?;

    Ok(true)
}

/// Build a map of blob SHA → list of file paths from the current HEAD tree.
pub fn blob_sha_to_paths(repo: &Repository) -> Result<HashMap<String, Vec<String>>> {
    let entries = walk_tree(repo)?;
    let mut map: HashMap<String, Vec<String>> = HashMap::new();

    for entry in entries {
        map.entry(entry.blob_sha)
            .or_default()
            .push(entry.path);
    }

    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- text_blob ---

    #[test]
    fn text_blob_recognizes_extensions() {
        assert!(text_blob("src/main.rs"));
        assert!(text_blob("README.md"));
        assert!(text_blob("config.yaml"));
        assert!(text_blob("Dockerfile.dockerfile"));
        assert!(text_blob(".gitignore"));
    }

    #[test]
    fn text_blob_case_insensitive() {
        assert!(text_blob("FILE.RS"));
        assert!(text_blob("notes.TXT"));
        assert!(text_blob("App.Jsx"));
    }

    #[test]
    fn text_blob_no_extension_is_text() {
        assert!(text_blob("Makefile"));
        assert!(text_blob("LICENSE"));
    }

    #[test]
    fn text_blob_rejects_binary() {
        assert!(!text_blob("image.png"));
        assert!(!text_blob("archive.zip"));
        assert!(!text_blob("model.onnx"));
    }

    #[test]
    fn text_blob_multi_dot_uses_last_extension() {
        // archive.tar.gz → checks ".gz" which is not in list
        assert!(!text_blob("archive.tar.gz"));
        // src.bak/main.rs → checks ".rs" which is in list
        assert!(text_blob("src.bak/main.rs"));
    }

    // --- Helper: create a temp repo with a commit ---

    fn temp_repo_with_commit(files: &[(&str, &str)]) -> (tempfile::TempDir, Repository) {
        let dir = tempfile::tempdir().unwrap();
        let repo = Repository::init(dir.path()).unwrap();

        // Write files to disk and use the index to build the tree
        // (handles nested paths correctly)
        for &(path, content) in files {
            let full = dir.path().join(path);
            if let Some(parent) = full.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&full, content).unwrap();
        }

        let tree_oid = {
            let mut index = repo.index().unwrap();
            for &(path, _) in files {
                index.add_path(std::path::Path::new(path)).unwrap();
            }
            index.write().unwrap();
            index.write_tree().unwrap()
        };

        {
            let tree = repo.find_tree(tree_oid).unwrap();
            let sig = git2::Signature::now("test", "test@test.com").unwrap();
            repo.commit(Some("HEAD"), &sig, &sig, "init", &tree, &[]).unwrap();
        }

        (dir, repo)
    }

    // --- find_repo ---

    #[test]
    fn test_find_repo_with_explicit_path() {
        let (dir, _repo) = temp_repo_with_commit(&[("README.md", "hello")]);
        let found = find_repo(Some(dir.path().to_str().unwrap())).unwrap();
        assert!(found.path().exists());
    }

    #[test]
    fn test_find_repo_invalid_path() {
        let result = find_repo(Some("/nonexistent/path/xyz"));
        assert!(result.is_err());
    }

    // --- walk_tree ---

    #[test]
    fn test_walk_tree_empty_repo() {
        let dir = tempfile::tempdir().unwrap();
        let repo = Repository::init(dir.path()).unwrap();
        let entries = walk_tree(&repo).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_walk_tree_returns_blobs() {
        let (_dir, repo) = temp_repo_with_commit(&[
            ("README.md", "hello world"),
            ("main.rs", "fn main() {}"),
        ]);
        let entries = walk_tree(&repo).unwrap();
        assert_eq!(entries.len(), 2);
        let paths: Vec<&str> = entries.iter().map(|e| e.path.as_str()).collect();
        assert!(paths.contains(&"README.md"));
        assert!(paths.contains(&"main.rs"));
    }

    #[test]
    fn test_walk_tree_includes_nested_files() {
        let (_dir, repo) = temp_repo_with_commit(&[
            ("top.txt", "top level"),
            ("src/lib.rs", "pub fn x() {}"),
        ]);
        let entries = walk_tree(&repo).unwrap();
        let paths: Vec<&str> = entries.iter().map(|e| e.path.as_str()).collect();
        assert!(paths.contains(&"top.txt"));
        // walk_tree uses git2's tree walk which gives "dir/" prefix + name
        assert!(
            paths.contains(&"src/lib.rs"),
            "expected 'src/lib.rs' in {:?}",
            paths
        );
    }

    #[test]
    fn test_walk_tree_sha_is_hex() {
        let (_dir, repo) = temp_repo_with_commit(&[("a.txt", "content")]);
        let entries = walk_tree(&repo).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].blob_sha.len(), 40);
        assert!(entries[0].blob_sha.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // --- read_blob ---

    #[test]
    fn test_read_blob_returns_content() {
        let (_dir, repo) = temp_repo_with_commit(&[("hello.txt", "hello world")]);
        let entries = walk_tree(&repo).unwrap();
        let content = read_blob(&repo, &entries[0].blob_sha).unwrap();
        assert_eq!(content, "hello world");
    }

    #[test]
    fn test_read_blob_invalid_sha() {
        let (_dir, repo) = temp_repo_with_commit(&[("a.txt", "x")]);
        let result = read_blob(&repo, "not-a-sha");
        assert!(result.is_err());
    }

    #[test]
    fn test_read_blob_nonexistent_sha() {
        let (_dir, repo) = temp_repo_with_commit(&[("a.txt", "x")]);
        let result = read_blob(&repo, "0000000000000000000000000000000000000000");
        assert!(result.is_err());
    }

    // --- read_ref / write_ref / delete_ref ---

    #[test]
    fn test_ref_roundtrip() {
        let (_dir, repo) = temp_repo_with_commit(&[("a.txt", "x")]);

        // Initially no ref
        assert!(read_ref(&repo).unwrap().is_none());

        // Write
        let data = b"test index data";
        write_ref(&repo, data).unwrap();

        // Read back
        let loaded = read_ref(&repo).unwrap().unwrap();
        assert_eq!(loaded, data);

        // Delete
        assert!(delete_ref(&repo).unwrap());

        // Gone
        assert!(read_ref(&repo).unwrap().is_none());
    }

    #[test]
    fn test_delete_ref_nonexistent() {
        let (_dir, repo) = temp_repo_with_commit(&[("a.txt", "x")]);
        assert!(!delete_ref(&repo).unwrap());
    }

    #[test]
    fn test_write_ref_overwrites() {
        let (_dir, repo) = temp_repo_with_commit(&[("a.txt", "x")]);
        write_ref(&repo, b"first").unwrap();
        write_ref(&repo, b"second").unwrap();
        let loaded = read_ref(&repo).unwrap().unwrap();
        assert_eq!(loaded, b"second");
    }

    // --- blob_sha_to_paths ---

    #[test]
    fn test_blob_sha_to_paths_maps_correctly() {
        let (_dir, repo) = temp_repo_with_commit(&[
            ("a.txt", "content A"),
            ("b.txt", "content B"),
        ]);
        let map = blob_sha_to_paths(&repo).unwrap();
        assert_eq!(map.len(), 2);
        // Each SHA should map to exactly one path
        for paths in map.values() {
            assert_eq!(paths.len(), 1);
        }
    }

    #[test]
    fn test_blob_sha_to_paths_duplicate_content() {
        // Two files with identical content share a blob SHA
        let (_dir, repo) = temp_repo_with_commit(&[
            ("a.txt", "same content"),
            ("b.txt", "same content"),
        ]);
        let map = blob_sha_to_paths(&repo).unwrap();
        // One SHA → two paths
        assert_eq!(map.len(), 1);
        let paths = map.values().next().unwrap();
        assert_eq!(paths.len(), 2);
    }
}
