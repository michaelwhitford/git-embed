//! Embedding index — load, save, serialize, deserialize.
//!
//! The index is stored as a git blob at `refs/embed/v1/index` in a binary
//! format compatible with Java's `DataOutputStream`:
//!
//! ```text
//! [version: i32 BE] [model: java-utf] [dims: i32 BE] [count: i32 BE]
//! for each embedding:
//!   [sha: java-utf] [vector: f32 × dims, each BE IEEE 754]
//! ```
//!
//! "java-utf" means Java Modified UTF-8 (`writeUTF`/`readUTF`): a 2-byte
//! big-endian length prefix (byte count) followed by the UTF-8 bytes.
//! For the ASCII strings we store (SHA hashes, model names) this is simply
//! `[u16 len][ascii bytes]`.

use std::collections::HashMap;
use std::io::{Cursor, Read, Write};

use anyhow::{bail, Context, Result};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use git2::Repository;
use log::warn;

use crate::git;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// In-memory representation of the embedding index.
#[derive(Debug, Clone)]
pub struct EmbedIndex {
    pub version: i32,
    pub model: String,
    pub dims: i32,
    pub embeddings: HashMap<String, Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const DEFAULT_MODEL: &str = "nomic-embed-text-v1.5";
const DEFAULT_DIMS: i32 = 768;

/// Create an empty index with default settings.
pub fn empty_index() -> EmbedIndex {
    EmbedIndex {
        version: 1,
        model: DEFAULT_MODEL.to_string(),
        dims: DEFAULT_DIMS,
        embeddings: HashMap::new(),
    }
}

// ---------------------------------------------------------------------------
// Java Modified UTF-8 helpers
// ---------------------------------------------------------------------------

/// Read a Java Modified UTF-8 string (as written by `DataOutputStream.writeUTF`).
///
/// Format: 2-byte big-endian length prefix (byte count), then that many bytes
/// of UTF-8 data.  For pure-ASCII strings this is trivial.
fn read_java_utf(rdr: &mut impl Read) -> Result<String> {
    let len = rdr.read_u16::<BigEndian>()? as usize;
    let mut buf = vec![0u8; len];
    rdr.read_exact(&mut buf)?;
    String::from_utf8(buf).context("invalid UTF-8 in java-utf string")
}

/// Write a Java Modified UTF-8 string (compatible with `DataInputStream.readUTF`).
///
/// For the ASCII-only strings we handle, this is a 2-byte big-endian length
/// followed by the raw bytes.
fn write_java_utf(wtr: &mut impl Write, s: &str) -> Result<()> {
    let bytes = s.as_bytes();
    if bytes.len() > u16::MAX as usize {
        bail!(
            "string too long for java modified UTF-8 ({} bytes, max {})",
            bytes.len(),
            u16::MAX
        );
    }
    wtr.write_u16::<BigEndian>(bytes.len() as u16)?;
    wtr.write_all(bytes)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

/// Serialize an [`EmbedIndex`] to bytes in the Java-compatible binary format.
pub fn serialize_index(idx: &EmbedIndex) -> Result<Vec<u8>> {
    // Pre-allocate: header + per-entry (40-byte SHA + dims×4 bytes)
    let estimated = 4 + 2 + idx.model.len() + 4 + 4
        + idx.embeddings.len() * (2 + 40 + (idx.dims as usize) * 4);
    let mut buf = Vec::with_capacity(estimated);

    buf.write_i32::<BigEndian>(idx.version)?;
    write_java_utf(&mut buf, &idx.model)?;
    buf.write_i32::<BigEndian>(idx.dims)?;
    buf.write_i32::<BigEndian>(idx.embeddings.len() as i32)?;

    for (sha, vec) in &idx.embeddings {
        write_java_utf(&mut buf, sha)?;
        for &val in vec {
            buf.write_f32::<BigEndian>(val)?;
        }
    }

    Ok(buf)
}

/// Deserialize an [`EmbedIndex`] from bytes in the Java-compatible binary format.
pub fn deserialize_index(data: &[u8]) -> Result<EmbedIndex> {
    let mut rdr = Cursor::new(data);

    let version = rdr
        .read_i32::<BigEndian>()
        .context("failed to read version")?;
    let model = read_java_utf(&mut rdr).context("failed to read model")?;
    let dims = rdr
        .read_i32::<BigEndian>()
        .context("failed to read dims")?;
    let count = rdr
        .read_i32::<BigEndian>()
        .context("failed to read count")?;

    if dims <= 0 {
        bail!("invalid dims: {dims}");
    }
    if count < 0 {
        bail!("invalid count: {count}");
    }

    let mut embeddings = HashMap::with_capacity(count as usize);

    for i in 0..count {
        let sha = read_java_utf(&mut rdr)
            .with_context(|| format!("failed to read sha for embedding {i}"))?;
        let mut vec = Vec::with_capacity(dims as usize);
        for j in 0..dims {
            let val = rdr.read_f32::<BigEndian>().with_context(|| {
                format!("failed to read float [{j}/{dims}] for embedding {i} (sha={sha})")
            })?;
            vec.push(val);
        }
        embeddings.insert(sha, vec);
    }

    Ok(EmbedIndex {
        version,
        model,
        dims,
        embeddings,
    })
}

// ---------------------------------------------------------------------------
// Git-backed persistence
// ---------------------------------------------------------------------------

/// Load the embedding index from the git ref.
///
/// Returns an empty index if the ref does not exist. If the stored data is
/// corrupt, logs a warning and returns an empty index.
pub fn load_index(repo: &Repository) -> Result<EmbedIndex> {
    match git::read_ref(repo)? {
        Some(data) => match deserialize_index(&data) {
            Ok(idx) => Ok(idx),
            Err(e) => {
                warn!("Corrupt index, starting fresh: {e}");
                Ok(empty_index())
            }
        },
        None => Ok(empty_index()),
    }
}

/// Serialize the index and write it to the git ref.
pub fn save_index(repo: &Repository, idx: &EmbedIndex) -> Result<()> {
    let data = serialize_index(idx)?;
    git::write_ref(repo, &data)?;
    Ok(())
}

/// Delete the embedding index ref entirely. Returns `true` if the ref existed
/// and was deleted, `false` if it was already absent.
pub fn clear_index(repo: &Repository) -> Result<bool> {
    git::delete_ref(repo)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_index_defaults() {
        let idx = empty_index();
        assert_eq!(idx.version, 1);
        assert_eq!(idx.model, "nomic-embed-text-v1.5");
        assert_eq!(idx.dims, 768);
        assert!(idx.embeddings.is_empty());
    }

    #[test]
    fn test_roundtrip_empty() {
        let idx = empty_index();
        let data = serialize_index(&idx).unwrap();
        let idx2 = deserialize_index(&data).unwrap();
        assert_eq!(idx2.version, idx.version);
        assert_eq!(idx2.model, idx.model);
        assert_eq!(idx2.dims, idx.dims);
        assert!(idx2.embeddings.is_empty());
    }

    #[test]
    fn test_roundtrip_with_embeddings() {
        let mut idx = empty_index();
        idx.embeddings.insert(
            "abc123".to_string(),
            vec![1.0, 2.0, 3.0],
        );
        idx.embeddings.insert(
            "def456".to_string(),
            vec![4.0, 5.0, 6.0],
        );
        idx.dims = 3;

        let data = serialize_index(&idx).unwrap();
        let idx2 = deserialize_index(&data).unwrap();

        assert_eq!(idx2.version, 1);
        assert_eq!(idx2.dims, 3);
        assert_eq!(idx2.embeddings.len(), 2);
        assert_eq!(idx2.embeddings["abc123"], vec![1.0, 2.0, 3.0]);
        assert_eq!(idx2.embeddings["def456"], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_java_utf_roundtrip() {
        let test_str = "hello-world-sha256abc";
        let mut buf = Vec::new();
        write_java_utf(&mut buf, test_str).unwrap();

        // Verify format: 2-byte BE length + bytes
        assert_eq!(buf.len(), 2 + test_str.len());
        let expected_len = (test_str.len() as u16).to_be_bytes();
        assert_eq!(&buf[..2], &expected_len);

        let mut cursor = Cursor::new(&buf);
        let result = read_java_utf(&mut cursor).unwrap();
        assert_eq!(result, test_str);
    }

    #[test]
    fn test_java_utf_empty_string() {
        let mut buf = Vec::new();
        write_java_utf(&mut buf, "").unwrap();
        assert_eq!(buf, vec![0, 0]);

        let mut cursor = Cursor::new(&buf);
        let result = read_java_utf(&mut cursor).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_deserialize_corrupt_data() {
        // Too short to even contain a version
        let result = deserialize_index(&[0x00, 0x01]);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_negative_dims() {
        // version=1, model="" (0x0000), dims=-1, count=0
        let mut buf = Vec::new();
        buf.write_i32::<BigEndian>(1).unwrap();
        buf.write_u16::<BigEndian>(0).unwrap(); // empty model
        buf.write_i32::<BigEndian>(-1).unwrap(); // bad dims
        buf.write_i32::<BigEndian>(0).unwrap();

        let result = deserialize_index(&buf);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid dims"));
    }

    #[test]
    fn test_deserialize_zero_dims() {
        let mut buf = Vec::new();
        buf.write_i32::<BigEndian>(1).unwrap();
        buf.write_u16::<BigEndian>(0).unwrap();
        buf.write_i32::<BigEndian>(0).unwrap(); // dims=0
        buf.write_i32::<BigEndian>(0).unwrap();
        let result = deserialize_index(&buf);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid dims"));
    }

    #[test]
    fn test_deserialize_truncated_embedding() {
        // version=1, model="", dims=3, count=1, sha="aa", then only 2 floats (need 3)
        let mut buf = Vec::new();
        buf.write_i32::<BigEndian>(1).unwrap();
        buf.write_u16::<BigEndian>(0).unwrap();
        buf.write_i32::<BigEndian>(3).unwrap();
        buf.write_i32::<BigEndian>(1).unwrap();
        // sha
        buf.write_u16::<BigEndian>(2).unwrap();
        buf.extend_from_slice(b"aa");
        // only 2 floats instead of 3
        buf.write_f32::<BigEndian>(1.0).unwrap();
        buf.write_f32::<BigEndian>(2.0).unwrap();
        let result = deserialize_index(&buf);
        assert!(result.is_err());
    }

    // --- Git-backed persistence (load/save/clear) ---

    fn temp_repo_with_commit() -> (tempfile::TempDir, git2::Repository) {
        let dir = tempfile::tempdir().unwrap();
        let repo = git2::Repository::init(dir.path()).unwrap();
        let tree_oid = repo.treebuilder(None).unwrap().write().unwrap();
        {
            let tree = repo.find_tree(tree_oid).unwrap();
            let sig = git2::Signature::now("test", "test@test.com").unwrap();
            repo.commit(Some("HEAD"), &sig, &sig, "init", &tree, &[]).unwrap();
        }
        (dir, repo)
    }

    #[test]
    fn test_load_index_empty_repo() {
        let (_dir, repo) = temp_repo_with_commit();
        let idx = load_index(&repo).unwrap();
        assert!(idx.embeddings.is_empty());
        assert_eq!(idx.version, 1);
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let (_dir, repo) = temp_repo_with_commit();
        let mut idx = empty_index();
        // Must match idx.dims (768) in length
        let emb = vec![0.5f32; 768];
        idx.embeddings.insert("abc123".to_string(), emb.clone());

        save_index(&repo, &idx).unwrap();
        let loaded = load_index(&repo).unwrap();

        assert_eq!(loaded.embeddings.len(), 1);
        assert_eq!(loaded.embeddings["abc123"], emb);
        assert_eq!(loaded.model, DEFAULT_MODEL);
    }

    #[test]
    fn test_save_overwrites_previous() {
        let (_dir, repo) = temp_repo_with_commit();

        let mut idx1 = empty_index();
        idx1.embeddings.insert(
            "aaaa000000000000000000000000000000000001".to_string(),
            vec![1.0; 768],
        );
        save_index(&repo, &idx1).unwrap();

        let mut idx2 = empty_index();
        idx2.embeddings.insert(
            "bbbb000000000000000000000000000000000002".to_string(),
            vec![2.0; 768],
        );
        save_index(&repo, &idx2).unwrap();

        let loaded = load_index(&repo).unwrap();
        assert_eq!(loaded.embeddings.len(), 1);
        assert!(loaded.embeddings.contains_key("bbbb000000000000000000000000000000000002"));
    }

    #[test]
    fn test_clear_index_existing() {
        let (_dir, repo) = temp_repo_with_commit();
        let idx = empty_index();
        save_index(&repo, &idx).unwrap();

        assert!(clear_index(&repo).unwrap());
        let loaded = load_index(&repo).unwrap();
        assert!(loaded.embeddings.is_empty());
    }

    #[test]
    fn test_clear_index_nonexistent() {
        let (_dir, repo) = temp_repo_with_commit();
        assert!(!clear_index(&repo).unwrap());
    }

    #[test]
    fn test_load_index_corrupt_ref() {
        let (_dir, repo) = temp_repo_with_commit();
        // Write garbage data to the ref
        crate::git::write_ref(&repo, b"not valid index data").unwrap();
        // Should warn and return empty (not error)
        let idx = load_index(&repo).unwrap();
        assert!(idx.embeddings.is_empty());
    }

    /// Verify byte-level compatibility with Java's DataOutputStream format.
    #[test]
    fn test_binary_compatibility() {
        let mut idx = empty_index();
        idx.dims = 2;
        idx.embeddings
            .insert("aa".to_string(), vec![1.0_f32, -1.0_f32]);

        let data = serialize_index(&idx).unwrap();
        let mut cursor = Cursor::new(&data);

        // version
        assert_eq!(cursor.read_i32::<BigEndian>().unwrap(), 1);
        // model length + bytes
        let model_len = cursor.read_u16::<BigEndian>().unwrap();
        assert_eq!(model_len as usize, DEFAULT_MODEL.len());
        let mut model_bytes = vec![0u8; model_len as usize];
        cursor.read_exact(&mut model_bytes).unwrap();
        assert_eq!(String::from_utf8(model_bytes).unwrap(), DEFAULT_MODEL);
        // dims
        assert_eq!(cursor.read_i32::<BigEndian>().unwrap(), 2);
        // count
        assert_eq!(cursor.read_i32::<BigEndian>().unwrap(), 1);
        // sha
        let sha_len = cursor.read_u16::<BigEndian>().unwrap();
        assert_eq!(sha_len, 2);
        let mut sha_bytes = vec![0u8; 2];
        cursor.read_exact(&mut sha_bytes).unwrap();
        assert_eq!(String::from_utf8(sha_bytes).unwrap(), "aa");
        // floats
        assert_eq!(cursor.read_f32::<BigEndian>().unwrap(), 1.0_f32);
        assert_eq!(cursor.read_f32::<BigEndian>().unwrap(), -1.0_f32);
    }
}
