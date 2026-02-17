//! Model management — download, load, and run inference with nomic-embed-text-v1.5.
//!
//! Uses ONNX Runtime (via `ort`) for inference and HuggingFace `tokenizers`
//! crate directly (no JNI wrapper).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const MODEL_ID: &str = "nomic-ai/nomic-embed-text-v1.5";
pub const MODEL_NAME: &str = "nomic-embed-text-v1.5";
pub const MODEL_DIMS: usize = 768;
pub const MAX_SEQ_LENGTH: usize = 8192;
pub const CHUNK_THRESHOLD: usize = 512;

pub const PREFIX_DOCUMENT: &str = "search_document: ";
pub const PREFIX_QUERY: &str = "search_query: ";

const MAX_BATCH_TOKENS: usize = 16384;
const MAX_BATCH_SIZE: usize = 32;

/// Model files to download from HuggingFace.
const MODEL_FILES: &[(&str, &str)] = &[
    ("model.onnx", "onnx/model.onnx"),
    ("tokenizer.json", "tokenizer.json"),
    ("special_tokens_map.json", "special_tokens_map.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
];

// ---------------------------------------------------------------------------
// Model directory
// ---------------------------------------------------------------------------

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".git-embed")
        .join("models")
        .join(MODEL_NAME)
}

fn model_ready() -> bool {
    let dir = model_dir();
    dir.join("model.onnx").exists() && dir.join("tokenizer.json").exists()
}

/// Download model files if not present. Returns the model directory path.
pub fn ensure_model() -> Result<PathBuf> {
    let dir = model_dir();
    if model_ready() {
        return Ok(dir);
    }

    eprintln!("Downloading model {} to {}", MODEL_ID, dir.display());
    std::fs::create_dir_all(&dir)?;

    for &(local_name, repo_path) in MODEL_FILES {
        let dest = dir.join(local_name);
        if dest.exists() {
            continue;
        }
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            MODEL_ID, repo_path
        );
        eprintln!("  Downloading {} ...", local_name);

        let resp = ureq::get(&url).call()
            .with_context(|| format!("failed to download {}", url))?;

        let len: u64 = resp
            .header("Content-Length")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        let pb = ProgressBar::new(len);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("    [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏ "),
        );

        let mut reader = pb.wrap_read(resp.into_reader());
        let mut file = std::fs::File::create(&dest)?;
        std::io::copy(&mut reader, &mut file)?;
        pb.finish_and_clear();
    }

    Ok(dir)
}

// ---------------------------------------------------------------------------
// EmbedModel — holds session + tokenizer
// ---------------------------------------------------------------------------

pub struct EmbedModel {
    session: Session, // needs &mut for run()
    /// Tokenizer with truncation at CHUNK_THRESHOLD for inference.
    tokenizer: Tokenizer,
    /// Tokenizer without truncation for measuring true token counts.
    counting_tokenizer: Tokenizer,
}

impl EmbedModel {
    /// Load the ONNX model and tokenizers from the model directory.
    pub fn load() -> Result<Self> {
        let dir = ensure_model()?;
        Self::load_from(&dir)
    }

    pub fn load_from(dir: &Path) -> Result<Self> {
        let model_path = dir.join("model.onnx");
        let tokenizer_path = dir.join("tokenizer.json");

        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(cpus)?
            .with_inter_threads(1)?
            .commit_from_file(&model_path)
            .with_context(|| format!("failed to load ONNX model from {}", model_path.display()))?;

        // Inference tokenizer: truncation at CHUNK_THRESHOLD
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;
        tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: CHUNK_THRESHOLD,
            ..Default::default()
        })).map_err(|e| anyhow::anyhow!("failed to set truncation: {}", e))?;
        tokenizer.with_padding(None);

        // Counting tokenizer: no truncation
        let mut counting_tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load counting tokenizer: {}", e))?;
        let _ = counting_tokenizer.with_truncation(None);
        counting_tokenizer.with_padding(None);

        eprintln!(
            "Loaded ONNX model: {} ({} intra-op threads)",
            model_path.display(),
            cpus
        );

        Ok(Self {
            session,
            tokenizer,
            counting_tokenizer,
        })
    }

    // --- Token counting ---

    /// True token count (no truncation). Used for chunking decisions.
    fn token_count(&self, text: &str) -> usize {
        self.counting_tokenizer
            .encode(text, true)
            .map(|enc| enc.get_ids().len())
            .unwrap_or(0)
    }

    // --- Single inference ---

    /// Embed a single text (must fit within CHUNK_THRESHOLD after tokenization).
    fn embed_single(&mut self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {}", e))?;
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let type_ids = encoding.get_type_ids();
        let seq_len = ids.len();

        let ids_arr = Array2::from_shape_vec(
            (1, seq_len),
            ids.iter().map(|&x| x as i64).collect(),
        )?;
        let mask_arr = Array2::from_shape_vec(
            (1, seq_len),
            mask.iter().map(|&x| x as i64).collect(),
        )?;
        let types_arr = Array2::from_shape_vec(
            (1, seq_len),
            type_ids.iter().map(|&x| x as i64).collect(),
        )?;

        let ids_tensor = Tensor::from_array(ids_arr)?;
        let mask_tensor = Tensor::from_array(mask_arr)?;
        let types_tensor = Tensor::from_array(types_arr)?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
            "token_type_ids" => types_tensor,
        ])?;

        // Try to extract output — could be [1, dims] or [1, seq_len, dims]
        Self::extract_embedding(&outputs, mask, seq_len)
    }

    /// Extract embedding from ONNX output, handling both 2D and 3D shapes.
    fn extract_embedding(
        outputs: &ort::session::SessionOutputs,
        mask: &[u32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let tensor = &outputs[0];
        let shape = tensor.shape(); // returns &Shape which derefs to &[i64]

        if shape.len() == 3 {
            // [1, seq_len, dims] → mean pool
            let (_shape, data) = tensor.try_extract_tensor::<f32>()?;
            // data is flat: [1 * seq_len * dims], laid out as [batch][token][dim]
            let mut result = vec![0.0f32; MODEL_DIMS];
            let mask_sum: f32 = mask.iter().take(seq_len).map(|&m| m as f32).sum::<f32>().max(1e-9);

            for d in 0..MODEL_DIMS {
                let mut sum = 0.0f32;
                for t in 0..seq_len {
                    // flat index: batch=0, token=t, dim=d → 0*seq_len*dims + t*dims + d
                    sum += data[t * MODEL_DIMS + d] * mask[t] as f32;
                }
                result[d] = sum / mask_sum;
            }
            Ok(result)
        } else {
            // [1, dims] → direct
            let (_shape, data) = tensor.try_extract_tensor::<f32>()?;
            let mut result = vec![0.0f32; MODEL_DIMS];
            for d in 0..MODEL_DIMS {
                result[d] = data[d];
            }
            Ok(result)
        }
    }

    // --- Batched inference ---

    /// Embed a batch of pre-tokenized texts in one forward pass.
    fn embed_batch_raw(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let batch_size = texts.len();
        let encodings: Vec<_> = texts
            .iter()
            .map(|t| {
                self.tokenizer
                    .encode(*t, true)
                    .map_err(|e| anyhow::anyhow!("tokenization failed: {}", e))
            })
            .collect::<Result<Vec<_>>>()?;

        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        if max_len == 0 {
            return Ok(vec![vec![0.0; MODEL_DIMS]; batch_size]);
        }

        // Build padded arrays [batch, max_len]
        let mut ids_flat = vec![0i64; batch_size * max_len];
        let mut mask_flat = vec![0i64; batch_size * max_len];
        let mut types_flat = vec![0i64; batch_size * max_len];
        let mut seq_lens = Vec::with_capacity(batch_size);
        let mut masks: Vec<Vec<u32>> = Vec::with_capacity(batch_size);

        for (b, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let m = enc.get_attention_mask();
            let tids = enc.get_type_ids();
            let sl = ids.len();
            seq_lens.push(sl);
            masks.push(m.to_vec());

            for i in 0..sl {
                ids_flat[b * max_len + i] = ids[i] as i64;
                mask_flat[b * max_len + i] = m[i] as i64;
                types_flat[b * max_len + i] = tids[i] as i64;
            }
        }

        let ids_arr = Array2::from_shape_vec((batch_size, max_len), ids_flat)?;
        let mask_arr = Array2::from_shape_vec((batch_size, max_len), mask_flat)?;
        let types_arr = Array2::from_shape_vec((batch_size, max_len), types_flat)?;

        let ids_tensor = Tensor::from_array(ids_arr)?;
        let mask_tensor = Tensor::from_array(mask_arr)?;
        let types_tensor = Tensor::from_array(types_arr)?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
            "token_type_ids" => types_tensor,
        ])?;

        let tensor = &outputs[0];
        let shape = tensor.shape(); // &Shape, derefs to &[i64]

        if shape.len() == 3 {
            // [batch, seq_len, dims] → mean pool per example
            let (_shape, data) = tensor.try_extract_tensor::<f32>()?;
            // data is flat: batch * max_len * MODEL_DIMS
            let out_seq_len = shape[1] as usize;
            let mut results = Vec::with_capacity(batch_size);

            for b in 0..batch_size {
                let sl = seq_lens[b];
                let mask = &masks[b];
                let mask_sum: f32 = mask.iter().take(sl).map(|&m| m as f32).sum::<f32>().max(1e-9);
                let mut emb = vec![0.0f32; MODEL_DIMS];

                for d in 0..MODEL_DIMS {
                    let mut sum = 0.0f32;
                    for t in 0..sl {
                        // flat index: b * out_seq_len * MODEL_DIMS + t * MODEL_DIMS + d
                        sum += data[b * out_seq_len * MODEL_DIMS + t * MODEL_DIMS + d] * mask[t] as f32;
                    }
                    emb[d] = sum / mask_sum;
                }
                results.push(emb);
            }
            Ok(results)
        } else {
            // [batch, dims] — flat: batch * MODEL_DIMS
            let (_shape, data) = tensor.try_extract_tensor::<f32>()?;
            let mut results = Vec::with_capacity(batch_size);
            for b in 0..batch_size {
                let mut emb = vec![0.0f32; MODEL_DIMS];
                for d in 0..MODEL_DIMS {
                    emb[d] = data[b * MODEL_DIMS + d];
                }
                results.push(emb);
            }
            Ok(results)
        }
    }

    // --- Chunking ---

    /// Split text into line-boundary chunks that fit within token budget.
    fn split_text_chunks(&self, text: &str, prefix: &str, budget: usize) -> Vec<String> {
        let lines: Vec<&str> = text.lines().collect();
        let prefix_tokens = self.token_count(prefix);
        let chunk_budget = budget.saturating_sub(prefix_tokens).saturating_sub(2);
        if chunk_budget == 0 {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let mut current_lines: Vec<&str> = Vec::new();
        let mut current_tokens: usize = 0;

        for line in &lines {
            let line_tokens = self.token_count(line).saturating_sub(2).max(1);
            let new_total = current_tokens + line_tokens;

            if new_total <= chunk_budget {
                current_lines.push(line);
                current_tokens = new_total;
            } else if !current_lines.is_empty() {
                chunks.push(current_lines.join("\n"));
                current_lines = vec![line];
                current_tokens = line_tokens;
            } else {
                // Single oversized line — include it anyway
                chunks.push(line.to_string());
            }
        }

        if !current_lines.is_empty() {
            chunks.push(current_lines.join("\n"));
        }

        chunks
    }

    /// Weighted average of embeddings by token count.
    fn weighted_average(embs: &[(Vec<f32>, usize)]) -> Vec<f32> {
        let total_weight: f64 = embs.iter().map(|(_, w)| *w as f64).sum();
        let mut result = vec![0.0f32; MODEL_DIMS];
        if total_weight > 0.0 {
            for (emb, weight) in embs {
                let w = *weight as f64 / total_weight;
                for d in 0..MODEL_DIMS {
                    result[d] += (emb[d] as f64 * w) as f32;
                }
            }
        }
        result
    }

    /// Group items into batches respecting token budget and max batch size.
    fn make_batches(items: &[(usize, String, usize)]) -> Vec<Vec<usize>> {
        // items: (original_index, prefixed_text, token_count)
        let mut sorted: Vec<(usize, usize)> = items
            .iter()
            .enumerate()
            .map(|(i, (_, _, toks))| (i, *toks))
            .collect();
        sorted.sort_by_key(|&(_, toks)| toks);

        let mut batches = Vec::new();
        let mut current_batch: Vec<usize> = Vec::new();

        for (idx, toks) in sorted {
            let projected = (current_batch.len() + 1) * toks;
            if !current_batch.is_empty()
                && (current_batch.len() >= MAX_BATCH_SIZE || projected > MAX_BATCH_TOKENS)
            {
                batches.push(std::mem::take(&mut current_batch));
            }
            current_batch.push(idx);
        }
        if !current_batch.is_empty() {
            batches.push(current_batch);
        }
        batches
    }

    // --- Public API ---

    /// Embed a single text with prefix, handling chunking if needed.
    pub fn embed_text(&mut self, text: &str, prefix: &str) -> Result<Vec<f32>> {
        let prefixed = format!("{}{}", prefix, text);
        let n_tokens = self.token_count(&prefixed);

        if n_tokens <= CHUNK_THRESHOLD {
            return self.embed_single(&prefixed);
        }

        let chunks = self.split_text_chunks(text, prefix, CHUNK_THRESHOLD);
        // Collect all prefixed chunks with token counts
        let chunk_items: Vec<(usize, String, usize)> = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let pc = format!("{}{}", prefix, c);
                let toks = self.token_count(&pc);
                (i, pc, toks)
            })
            .collect();

        // Batch the chunks
        let batches = Self::make_batches(&chunk_items);
        let mut all_embs: Vec<Option<Vec<f32>>> = vec![None; chunks.len()];

        for batch_indices in &batches {
            let batch_texts: Vec<&str> = batch_indices
                .iter()
                .map(|&i| chunk_items[i].1.as_str())
                .collect();
            let batch_embs = self.embed_batch_raw(&batch_texts)?;
            for (bi, &idx) in batch_indices.iter().enumerate() {
                all_embs[idx] = Some(batch_embs[bi].clone());
            }
        }

        let weighted: Vec<(Vec<f32>, usize)> = chunk_items
            .iter()
            .map(|(i, _, toks)| {
                (all_embs[*i].clone().unwrap_or_else(|| vec![0.0; MODEL_DIMS]), *toks)
            })
            .collect();

        Ok(Self::weighted_average(&weighted))
    }

    /// Embed document content with search_document prefix.
    pub fn embed_document(&mut self, content: &str) -> Result<Vec<f32>> {
        self.embed_text(content, PREFIX_DOCUMENT)
    }

    /// Embed query text with search_query prefix.
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        self.embed_text(query, PREFIX_QUERY)
    }

    /// Embed multiple documents with batched inference.
    /// Returns embeddings in the same order as input texts.
    pub fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Build flat list of all chunks across all documents
        struct ChunkInfo {
            doc_idx: usize,
            prefixed: String,
            tokens: usize,
            single: bool,
        }

        let mut all_chunks: Vec<ChunkInfo> = Vec::new();

        for (doc_idx, &text) in texts.iter().enumerate() {
            let prefixed = format!("{}{}", PREFIX_DOCUMENT, text);
            let n_tokens = self.token_count(&prefixed);

            if n_tokens <= CHUNK_THRESHOLD {
                all_chunks.push(ChunkInfo {
                    doc_idx,
                    prefixed,
                    tokens: n_tokens,
                    single: true,
                });
            } else {
                let chunks = self.split_text_chunks(text, PREFIX_DOCUMENT, CHUNK_THRESHOLD);
                for c in chunks {
                    let pc = format!("{}{}", PREFIX_DOCUMENT, c);
                    let toks = self.token_count(&pc);
                    all_chunks.push(ChunkInfo {
                        doc_idx,
                        prefixed: pc,
                        tokens: toks,
                        single: false,
                    });
                }
            }
        }

        // Sort chunks by token count for efficient batching
        let mut indices: Vec<usize> = (0..all_chunks.len()).collect();
        indices.sort_by_key(|&i| all_chunks[i].tokens);

        // Build batches
        let mut batches: Vec<Vec<usize>> = Vec::new();
        let mut current_batch: Vec<usize> = Vec::new();

        for idx in indices {
            let toks = all_chunks[idx].tokens;
            let projected = (current_batch.len() + 1) * toks;
            if !current_batch.is_empty()
                && (current_batch.len() >= MAX_BATCH_SIZE || projected > MAX_BATCH_TOKENS)
            {
                batches.push(std::mem::take(&mut current_batch));
            }
            current_batch.push(idx);
        }
        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        // Run batched inference
        let mut chunk_embs: Vec<Option<Vec<f32>>> = vec![None; all_chunks.len()];
        for batch_indices in &batches {
            let batch_texts: Vec<&str> = batch_indices
                .iter()
                .map(|&i| all_chunks[i].prefixed.as_str())
                .collect();
            let embs = self.embed_batch_raw(&batch_texts)?;
            for (bi, &idx) in batch_indices.iter().enumerate() {
                chunk_embs[idx] = Some(embs[bi].clone());
            }
        }

        // Reassemble per-document embeddings
        let num_docs = texts.len();
        let mut doc_embs = vec![Vec::new(); num_docs];

        // Group chunks by doc_idx
        let mut doc_chunks: Vec<Vec<usize>> = vec![Vec::new(); num_docs];
        for (i, chunk) in all_chunks.iter().enumerate() {
            doc_chunks[chunk.doc_idx].push(i);
        }

        for doc_idx in 0..num_docs {
            let chunks = &doc_chunks[doc_idx];
            if chunks.len() == 1 && all_chunks[chunks[0]].single {
                doc_embs[doc_idx] = chunk_embs[chunks[0]].clone().unwrap_or_else(|| vec![0.0; MODEL_DIMS]);
            } else {
                let weighted: Vec<(Vec<f32>, usize)> = chunks
                    .iter()
                    .map(|&i| {
                        let emb = chunk_embs[i].clone().unwrap_or_else(|| vec![0.0; MODEL_DIMS]);
                        (emb, all_chunks[i].tokens)
                    })
                    .collect();
                doc_embs[doc_idx] = Self::weighted_average(&weighted);
            }
        }

        Ok(doc_embs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- weighted_average ---

    #[test]
    fn test_weighted_average_single() {
        let mut emb = vec![0.0f32; MODEL_DIMS];
        emb[0] = 1.0;
        emb[1] = 2.0;
        let result = EmbedModel::weighted_average(&[(emb.clone(), 10)]);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_average_equal_weights() {
        let mut a = vec![0.0f32; MODEL_DIMS];
        let mut b = vec![0.0f32; MODEL_DIMS];
        a[0] = 2.0;
        b[0] = 4.0;
        let result = EmbedModel::weighted_average(&[(a, 1), (b, 1)]);
        assert!((result[0] - 3.0).abs() < 1e-6); // (2+4)/2
    }

    #[test]
    fn test_weighted_average_unequal_weights() {
        let mut a = vec![0.0f32; MODEL_DIMS];
        let mut b = vec![0.0f32; MODEL_DIMS];
        a[0] = 0.0;
        b[0] = 10.0;
        // weight 1:3 → 0*0.25 + 10*0.75 = 7.5
        let result = EmbedModel::weighted_average(&[(a, 1), (b, 3)]);
        assert!((result[0] - 7.5).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_average_zero_weight() {
        let result = EmbedModel::weighted_average(&[]);
        assert_eq!(result.len(), MODEL_DIMS);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // --- make_batches ---

    #[test]
    fn test_make_batches_single_item() {
        let items = vec![(0, "text".to_string(), 100)];
        let batches = EmbedModel::make_batches(&items);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
    }

    #[test]
    fn test_make_batches_respects_token_budget() {
        // 3 items each 6000 tokens. Budget is 16384.
        // Batch of 2 → 2*6000=12000 ✓, batch of 3 → 3*6000=18000 ✗
        let items: Vec<_> = (0..3).map(|i| (i, format!("t{i}"), 6000)).collect();
        let batches = EmbedModel::make_batches(&items);
        assert_eq!(batches.len(), 2); // [2, 1]
    }

    #[test]
    fn test_make_batches_respects_max_size() {
        // 40 items with tiny tokens — should split at MAX_BATCH_SIZE (32)
        let items: Vec<_> = (0..40).map(|i| (i, format!("t{i}"), 10)).collect();
        let batches = EmbedModel::make_batches(&items);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 32);
        assert_eq!(batches[1].len(), 8);
    }

    #[test]
    fn test_make_batches_sorts_by_tokens() {
        // Items in descending order — batching sorts ascending
        let items = vec![
            (0, "big".to_string(), 500),
            (1, "small".to_string(), 10),
            (2, "medium".to_string(), 200),
        ];
        let batches = EmbedModel::make_batches(&items);
        // All fit in one batch (3 * 500 = 1500 < 16384)
        assert_eq!(batches.len(), 1);
        // Should be sorted: small(1), medium(2), big(0)
        assert_eq!(batches[0], vec![1, 2, 0]);
    }

    #[test]
    fn test_make_batches_empty() {
        let items: Vec<(usize, String, usize)> = vec![];
        let batches = EmbedModel::make_batches(&items);
        assert!(batches.is_empty());
    }
}
