use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::process;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};

use git_embed::git;
use git_embed::index;
use git_embed::model::EmbedModel;
use git_embed::search::{self, SearchOpts};

// ---------------------------------------------------------------------------
// Memory-tracking allocator (near-zero overhead when enabled)
// ---------------------------------------------------------------------------

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
            // Update peak via relaxed CAS loop
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

/// Semantic similarity search for git repositories.
#[derive(Parser)]
#[command(name = "git-embed", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Truncation dimensions (64, 128, 256, 384, 512, 768)
    #[arg(short, long, default_value_t = 768, global = true)]
    dims: usize,

    /// Number of results
    #[arg(short = 'n', long, default_value_t = 10, global = true)]
    top: usize,

    /// Restrict search to path prefix
    #[arg(short, long, global = true)]
    path: Option<String>,

    /// Verbose output during indexing
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Show peak memory usage after command completes
    #[arg(long, global = true)]
    memory_stats: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Index all unindexed text files in HEAD
    Update,
    /// Semantic search across indexed content
    Search {
        /// Query text
        query: Vec<String>,
    },
    /// Find files similar to a given file
    Similar {
        /// File path
        file: String,
    },
    /// Show index status
    Status,
    /// Remove embeddings for deleted files
    Gc,
    /// Delete the entire embedding index
    Clear,
    /// Install git hooks for automatic index updates
    Install,
    /// Remove git-embed hooks
    Uninstall,
}

fn main() {
    let cli = Cli::parse();

    let show_mem = cli.memory_stats;
    if show_mem {
        TRACKING.store(true, Ordering::Relaxed);
    }

    if let Err(e) = run(cli) {
        eprintln!("Error: {:#}", e);
        if show_mem {
            print_memory_stats();
        }
        process::exit(1);
    }

    if show_mem {
        print_memory_stats();
    }
}

fn fmt_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn print_memory_stats() {
    // Stop tracking before we print (avoids measuring our own output allocs)
    TRACKING.store(false, Ordering::Relaxed);

    let total = ALLOC_BYTES.load(Ordering::Relaxed);
    let count = ALLOC_COUNT.load(Ordering::Relaxed);
    let peak = PEAK_BYTES.load(Ordering::Relaxed);

    eprintln!();
    eprintln!("Memory: {} peak, {} total across {} allocations",
        fmt_bytes(peak),
        fmt_bytes(total),
        count,
    );
}

fn run(cli: Cli) -> Result<()> {
    let repo = git::find_repo(None)?;

    match cli.command {
        None | Some(Command::Update) => cmd_update(&repo, &cli),
        Some(Command::Search { ref query }) => cmd_search(&repo, &cli, query),
        Some(Command::Similar { ref file }) => cmd_similar(&repo, &cli, file),
        Some(Command::Status) => cmd_status(&repo),
        Some(Command::Gc) => cmd_gc(&repo),
        Some(Command::Clear) => cmd_clear(&repo),
        Some(Command::Install) => cmd_install(&repo),
        Some(Command::Uninstall) => cmd_uninstall(&repo),
    }
}

fn cmd_update(repo: &git2::Repository, cli: &Cli) -> Result<()> {
    let mut idx = index::load_index(repo)?;

    let blobs = git::walk_tree(repo)?;
    let text_blobs: Vec<_> = blobs.iter().filter(|e| git::text_blob(&e.path)).collect();

    // --- Prune stale embeddings (inline gc) ---
    let live_shas: std::collections::HashSet<&str> =
        blobs.iter().map(|e| e.blob_sha.as_str()).collect();
    let before = idx.embeddings.len();
    idx.embeddings.retain(|sha, _| live_shas.contains(sha.as_str()));
    let pruned = before - idx.embeddings.len();

    // --- Find new blobs to index ---
    let new_blobs: Vec<_> = text_blobs
        .iter()
        .filter(|e| !idx.embeddings.contains_key(&e.blob_sha))
        .collect();

    let total = new_blobs.len();
    if total == 0 && pruned == 0 {
        println!("Index is current, nothing to update.");
        return Ok(());
    }

    if pruned > 0 {
        eprintln!("Pruned {} stale embeddings", pruned);
    }

    if total == 0 {
        // Only pruned, no new — save and done
        index::save_index(repo, &idx)?;
        println!("Index updated: {} embeddings", idx.embeddings.len());
        return Ok(());
    }

    eprintln!("Indexing {} new blobs", total);
    let mut model = EmbedModel::load()?;

    // Read contents, filter skippable
    let mut contents: Vec<(String, String, String)> = Vec::new(); // (sha, path, content)
    for entry in &new_blobs {
        match git::read_blob(repo, &entry.blob_sha) {
            Ok(content) => {
                if content.is_empty() || content.len() > 102_400 {
                    if cli.verbose {
                        eprintln!("  ✗ {} (skipped: size)", entry.path);
                    }
                    continue;
                }
                contents.push((entry.blob_sha.clone(), entry.path.clone(), content));
            }
            Err(e) => {
                if cli.verbose {
                    eprintln!("  ✗ {}: {}", entry.path, e);
                }
            }
        }
    }

    let skipped = total - contents.len();
    if skipped > 0 {
        eprintln!("Skipping {} blobs (empty or >100KB)", skipped);
    }

    // Process in batches of 32
    let batch_size = 32;
    let total_items = contents.len();

    let pb = if !cli.verbose {
        let pb = ProgressBar::new(total_items as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  Indexing: {pos}/{len} ({percent}%)")
                .unwrap(),
        );
        Some(pb)
    } else {
        None
    };

    for chunk in contents.chunks(batch_size) {
        let texts: Vec<&str> = chunk.iter().map(|(_, _, c)| c.as_str()).collect();
        match model.embed_documents(&texts) {
            Ok(embs) => {
                for (i, (sha, path, _)) in chunk.iter().enumerate() {
                    if cli.verbose {
                        println!("  ✓ {} ({})", path, sha);
                    }
                    idx.embeddings.insert(sha.clone(), embs[i].clone());
                }
            }
            Err(e) => {
                eprintln!("Batch failed ({}), falling back to sequential", e);
                for (sha, path, content) in chunk {
                    match model.embed_document(content) {
                        Ok(emb) => {
                            if cli.verbose {
                                println!("  ✓ {} ({})", path, sha);
                            }
                            idx.embeddings.insert(sha.clone(), emb);
                        }
                        Err(e) => {
                            if cli.verbose {
                                eprintln!("  ✗ {}: {}", path, e);
                            }
                        }
                    }
                }
            }
        }
        if let Some(ref pb) = pb {
            pb.set_position(pb.position() + chunk.len() as u64);
        }
    }

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    index::save_index(repo, &idx)?;
    println!(
        "Index updated: {} embeddings (+{}, -{})",
        idx.embeddings.len(),
        contents.len(),
        pruned,
    );

    Ok(())
}

fn cmd_search(repo: &git2::Repository, cli: &Cli, query: &[String]) -> Result<()> {
    if query.is_empty() {
        eprintln!("Usage: git embed search <query>");
        process::exit(1);
    }

    let mut model = EmbedModel::load()?;
    let idx = index::load_index(repo)?;

    let query_text = query.join(" ");
    let query_vec = model.embed_query(&query_text)?;

    let opts = SearchOpts {
        dims: cli.dims,
        top: cli.top,
        path: cli.path.clone(),
    };

    let results = search::search(repo, &idx, &query_vec, &opts);
    for r in &results {
        println!("{:.4}  {}", r.score, r.path);
    }

    Ok(())
}

fn cmd_similar(repo: &git2::Repository, cli: &Cli, file: &str) -> Result<()> {
    let idx = index::load_index(repo)?;

    let opts = SearchOpts {
        dims: cli.dims,
        top: cli.top,
        path: cli.path.clone(),
    };

    let results = search::similar(repo, &idx, file, &opts)?;
    for r in &results {
        println!("{:.4}  {}", r.score, r.path);
    }

    Ok(())
}

fn hooks_installed(repo: &git2::Repository) -> Vec<&'static str> {
    let dir = hooks_dir(repo);
    HOOKS
        .iter()
        .filter(|&&(name, _)| {
            let path = dir.join(name);
            path.exists()
                && std::fs::read_to_string(&path)
                    .map(|c| c.contains(HOOK_MARKER))
                    .unwrap_or(false)
        })
        .map(|&(name, _)| name)
        .collect()
}

fn cmd_status(repo: &git2::Repository) -> Result<()> {
    let idx = index::load_index(repo)?;
    let blobs = git::walk_tree(repo)?;
    let total = blobs.len();
    let indexed = idx.embeddings.len();

    println!("Model:   {}", idx.model);
    println!("Indexed: {} / {} blobs", indexed, total);
    println!("Dims:    {}", idx.dims);
    println!("Ref:     refs/embed/v1/index");

    let installed = hooks_installed(repo);
    if installed.len() == HOOKS.len() {
        println!("Hooks:   installed ({})", installed.join(", "));
    } else if installed.is_empty() {
        println!("Hooks:   not installed (run `git embed install`)");
    } else {
        let missing: Vec<&str> = HOOKS
            .iter()
            .map(|&(name, _)| name)
            .filter(|name| !installed.contains(name))
            .collect();
        println!("Hooks:   partial (missing: {})", missing.join(", "));
    }

    Ok(())
}

fn cmd_gc(repo: &git2::Repository) -> Result<()> {
    let mut idx = index::load_index(repo)?;
    let blobs = git::walk_tree(repo)?;
    let live_shas: std::collections::HashSet<String> =
        blobs.iter().map(|e| e.blob_sha.clone()).collect();

    let before = idx.embeddings.len();
    idx.embeddings.retain(|sha, _| live_shas.contains(sha));
    let pruned = before - idx.embeddings.len();

    index::save_index(repo, &idx)?;
    println!("Pruned {} unreferenced embeddings", pruned);

    Ok(())
}

fn cmd_clear(repo: &git2::Repository) -> Result<()> {
    if index::clear_index(repo)? {
        println!("Index cleared.");
    } else {
        println!("No index found.");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Hook management (git-lfs pattern)
// ---------------------------------------------------------------------------

/// Marker comment used to identify git-embed hook content.
const HOOK_MARKER: &str = "# git-embed";

/// The hooks we install and the git-embed subcommand each invokes.
const HOOKS: &[(&str, &str)] = &[
    ("post-commit", "update"),
    ("post-merge", "update"),
    ("post-checkout", "update"),
];

/// Generate the shell snippet for a single hook.
fn hook_snippet(subcmd: &str) -> String {
    format!(
        r#"{marker}
command -v git-embed >/dev/null 2>&1 || exit 0
git embed {subcmd} >/dev/null 2>&1 &
"#,
        marker = HOOK_MARKER,
        subcmd = subcmd,
    )
}

fn hooks_dir(repo: &git2::Repository) -> std::path::PathBuf {
    repo.path().join("hooks")
}

fn cmd_install(repo: &git2::Repository) -> Result<()> {
    let dir = hooks_dir(repo);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create hooks dir: {}", dir.display()))?;

    for &(hook_name, subcmd) in HOOKS {
        let path = dir.join(hook_name);
        let snippet = hook_snippet(subcmd);

        if path.exists() {
            let existing = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;

            if existing.contains(HOOK_MARKER) {
                // Already installed — replace our section
                let updated = replace_hook_section(&existing, &snippet);
                std::fs::write(&path, updated)
                    .with_context(|| format!("failed to write {}", path.display()))?;
                println!("  Updated {}", hook_name);
            } else {
                // Existing hook from something else — append
                let mut content = existing;
                if !content.ends_with('\n') {
                    content.push('\n');
                }
                content.push('\n');
                content.push_str(&snippet);
                std::fs::write(&path, content)
                    .with_context(|| format!("failed to write {}", path.display()))?;
                println!("  Appended to existing {}", hook_name);
            }
        } else {
            // New hook file
            let content = format!("#!/bin/sh\n{snippet}");
            std::fs::write(&path, content)
                .with_context(|| format!("failed to write {}", path.display()))?;
            println!("  Created {}", hook_name);
        }

        make_executable(&path)?;
    }

    println!("Hooks installed. Embeddings will update automatically after commit/merge/checkout.");
    Ok(())
}

fn cmd_uninstall(repo: &git2::Repository) -> Result<()> {
    let dir = hooks_dir(repo);
    let mut removed = 0;

    for &(hook_name, _) in HOOKS {
        let path = dir.join(hook_name);
        if !path.exists() {
            continue;
        }

        let existing = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;

        if !existing.contains(HOOK_MARKER) {
            continue;
        }

        let cleaned = remove_hook_section(&existing);
        let trimmed = cleaned.trim();

        if trimmed.is_empty() || trimmed == "#!/bin/sh" {
            // Nothing left — remove the file entirely
            std::fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
            println!("  Removed {}", hook_name);
        } else {
            // Other hook content remains — keep the file
            std::fs::write(&path, cleaned)
                .with_context(|| format!("failed to write {}", path.display()))?;
            println!("  Cleaned git-embed section from {}", hook_name);
        }
        removed += 1;
    }

    if removed == 0 {
        println!("No git-embed hooks found.");
    } else {
        println!("Hooks removed.");
    }
    Ok(())
}

/// Replace the git-embed section in an existing hook file.
fn replace_hook_section(content: &str, new_snippet: &str) -> String {
    let mut result = String::new();
    let mut in_section = false;
    let mut replaced = false;

    for line in content.lines() {
        if line.starts_with(HOOK_MARKER) {
            if !replaced {
                result.push_str(new_snippet);
                replaced = true;
            }
            in_section = true;
            continue;
        }

        if in_section {
            // Skip lines until we hit an empty line or another section
            if line.is_empty() {
                in_section = false;
            }
            continue;
        }

        result.push_str(line);
        result.push('\n');
    }

    result
}

/// Remove the git-embed section from a hook file.
fn remove_hook_section(content: &str) -> String {
    let mut result = String::new();
    let mut in_section = false;

    for line in content.lines() {
        if line.starts_with(HOOK_MARKER) {
            in_section = true;
            continue;
        }

        if in_section {
            if line.is_empty() {
                in_section = false;
            }
            continue;
        }

        result.push_str(line);
        result.push('\n');
    }

    result
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<()> {
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Pure function tests: hook_snippet ---

    #[test]
    fn test_hook_snippet_contains_marker() {
        let snippet = hook_snippet("update");
        assert!(snippet.starts_with(HOOK_MARKER));
    }

    #[test]
    fn test_hook_snippet_contains_command() {
        let snippet = hook_snippet("update");
        assert!(snippet.contains("git embed update"));
    }

    #[test]
    fn test_hook_snippet_graceful_degradation() {
        let snippet = hook_snippet("update");
        assert!(snippet.contains("command -v git-embed"));
        assert!(snippet.contains("exit 0"));
    }

    #[test]
    fn test_hook_snippet_runs_in_background() {
        let snippet = hook_snippet("update");
        assert!(snippet.contains("&\n"));
    }

    // --- Pure function tests: replace_hook_section ---

    #[test]
    fn test_replace_hook_section_updates_existing() {
        let old_snippet = hook_snippet("update");
        let existing = format!("#!/bin/sh\n{old_snippet}");

        let new_snippet = hook_snippet("update");
        let result = replace_hook_section(&existing, &new_snippet);

        // Should contain exactly one marker
        assert_eq!(result.matches(HOOK_MARKER).count(), 1);
        assert!(result.starts_with("#!/bin/sh\n"));
    }

    #[test]
    fn test_replace_hook_section_preserves_other_content() {
        let other = "#!/bin/sh\necho 'other hook'\n";
        let old_snippet = hook_snippet("update");
        let existing = format!("{other}\n{old_snippet}");

        let new_snippet = hook_snippet("update");
        let result = replace_hook_section(&existing, &new_snippet);

        assert!(result.contains("echo 'other hook'"));
        assert_eq!(result.matches(HOOK_MARKER).count(), 1);
    }

    // --- Pure function tests: remove_hook_section ---

    #[test]
    fn test_remove_hook_section_removes_marker_and_commands() {
        let snippet = hook_snippet("update");
        let content = format!("#!/bin/sh\n{snippet}");

        let result = remove_hook_section(&content);

        assert!(!result.contains(HOOK_MARKER));
        assert!(!result.contains("git embed update"));
        assert!(result.contains("#!/bin/sh"));
    }

    #[test]
    fn test_remove_hook_section_preserves_other_content() {
        let snippet = hook_snippet("update");
        let content = format!("#!/bin/sh\necho 'before'\n\n{snippet}\necho 'after'\n");

        let result = remove_hook_section(&content);

        assert!(result.contains("echo 'before'"));
        assert!(result.contains("echo 'after'"));
        assert!(!result.contains(HOOK_MARKER));
    }

    #[test]
    fn test_remove_hook_section_no_marker_unchanged() {
        let content = "#!/bin/sh\necho 'hello'\n";
        let result = remove_hook_section(content);
        assert_eq!(result, content);
    }

    // --- Integration tests: install / uninstall with temp repo ---

    fn temp_repo() -> (tempfile::TempDir, git2::Repository) {
        let dir = tempfile::tempdir().unwrap();
        let repo = git2::Repository::init(dir.path()).unwrap();
        (dir, repo)
    }

    #[test]
    fn test_install_creates_hook_files() {
        let (_dir, repo) = temp_repo();

        cmd_install(&repo).unwrap();

        for &(hook_name, _) in HOOKS {
            let path = hooks_dir(&repo).join(hook_name);
            assert!(path.exists(), "{} should exist", hook_name);

            let content = std::fs::read_to_string(&path).unwrap();
            assert!(content.starts_with("#!/bin/sh\n"));
            assert!(content.contains(HOOK_MARKER));
            assert!(content.contains("git embed"));
        }
    }

    #[cfg(unix)]
    #[test]
    fn test_install_sets_executable() {
        use std::os::unix::fs::PermissionsExt;

        let (_dir, repo) = temp_repo();
        cmd_install(&repo).unwrap();

        for &(hook_name, _) in HOOKS {
            let path = hooks_dir(&repo).join(hook_name);
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert!(mode & 0o111 != 0, "{} should be executable", hook_name);
        }
    }

    #[test]
    fn test_install_idempotent() {
        let (_dir, repo) = temp_repo();

        cmd_install(&repo).unwrap();
        cmd_install(&repo).unwrap();

        for &(hook_name, _) in HOOKS {
            let path = hooks_dir(&repo).join(hook_name);
            let content = std::fs::read_to_string(&path).unwrap();
            // Marker should appear exactly once
            assert_eq!(
                content.matches(HOOK_MARKER).count(),
                1,
                "{} should have exactly one marker",
                hook_name
            );
        }
    }

    #[test]
    fn test_install_appends_to_existing_hook() {
        let (_dir, repo) = temp_repo();
        let dir = hooks_dir(&repo);
        std::fs::create_dir_all(&dir).unwrap();

        // Pre-existing hook from another tool
        let existing = "#!/bin/sh\necho 'pre-existing hook'\n";
        std::fs::write(dir.join("post-commit"), existing).unwrap();

        cmd_install(&repo).unwrap();

        let content = std::fs::read_to_string(dir.join("post-commit")).unwrap();
        assert!(content.contains("echo 'pre-existing hook'"), "should preserve existing");
        assert!(content.contains(HOOK_MARKER), "should add git-embed section");
    }

    #[test]
    fn test_uninstall_removes_hook_files() {
        let (_dir, repo) = temp_repo();

        cmd_install(&repo).unwrap();
        cmd_uninstall(&repo).unwrap();

        for &(hook_name, _) in HOOKS {
            let path = hooks_dir(&repo).join(hook_name);
            assert!(!path.exists(), "{} should be removed", hook_name);
        }
    }

    #[test]
    fn test_uninstall_preserves_other_hook_content() {
        let (_dir, repo) = temp_repo();
        let dir = hooks_dir(&repo);
        std::fs::create_dir_all(&dir).unwrap();

        // Pre-existing hook content
        let existing = "#!/bin/sh\necho 'keep me'\n";
        std::fs::write(dir.join("post-commit"), existing).unwrap();

        cmd_install(&repo).unwrap();
        cmd_uninstall(&repo).unwrap();

        let path = dir.join("post-commit");
        assert!(path.exists(), "file should still exist with other content");
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("echo 'keep me'"), "should preserve other content");
        assert!(!content.contains(HOOK_MARKER), "should remove git-embed section");
    }

    #[test]
    fn test_uninstall_no_hooks_is_noop() {
        let (_dir, repo) = temp_repo();
        // Should not error when no hooks exist
        cmd_uninstall(&repo).unwrap();
    }

    #[test]
    fn test_uninstall_ignores_non_embed_hooks() {
        let (_dir, repo) = temp_repo();
        let dir = hooks_dir(&repo);
        std::fs::create_dir_all(&dir).unwrap();

        let other = "#!/bin/sh\necho 'other tool'\n";
        std::fs::write(dir.join("post-commit"), other).unwrap();

        cmd_uninstall(&repo).unwrap();

        // Should not touch it — no marker present
        let content = std::fs::read_to_string(dir.join("post-commit")).unwrap();
        assert_eq!(content, other);
    }

    #[test]
    fn test_full_lifecycle() {
        let (_dir, repo) = temp_repo();

        // Install
        cmd_install(&repo).unwrap();
        for &(hook_name, _) in HOOKS {
            assert!(hooks_dir(&repo).join(hook_name).exists());
        }

        // Re-install (idempotent)
        cmd_install(&repo).unwrap();
        for &(hook_name, _) in HOOKS {
            let content = std::fs::read_to_string(hooks_dir(&repo).join(hook_name)).unwrap();
            assert_eq!(content.matches(HOOK_MARKER).count(), 1);
        }

        // Uninstall
        cmd_uninstall(&repo).unwrap();
        for &(hook_name, _) in HOOKS {
            assert!(!hooks_dir(&repo).join(hook_name).exists());
        }

        // Uninstall again (noop)
        cmd_uninstall(&repo).unwrap();
    }
}
