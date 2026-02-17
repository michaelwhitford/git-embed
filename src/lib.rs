//! Library interface for benchmarks and tests.
//!
//! The binary crate (`main.rs`) owns the CLI; this re-exports internal
//! modules so that criterion benchmarks can exercise them directly.

pub mod git;
pub mod index;
pub mod model;
pub mod search;
