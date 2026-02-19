// Type definitions for GGUF loader
use std::collections::HashMap;
use crate::types::{GgufValue, Q8Block, EmbeddingMatrix};

#[derive(Debug)]
pub enum GgmlType {
  GgmlTypeF32     = 0,
  GgmlTypeF16     = 1,
  GgmlTypeQ4_0  = 2,
  GgmlTypeQ4_1    = 3,
  // GGML_TYPE_Q4_2 = 4, support has been removed
  // GGML_TYPE_Q4_3 = 5, support has been removed
  GgmlTypeQ5_0    = 6,
  GgmlTypeQ5_1    = 7,
  GgmlTypeQ8_0    = 8,
  GgmlTypeQ8_1    = 9,
  GgmlTypeQ2K    = 10,
  GgmlTypeQ3K    = 11,
  GgmlTypeQ4K    = 12,
  GgmlTypeQ5K    = 13,
  GgmlTypeQ6K    = 14,
  GgmlTypeQ8K    = 15,
  GgmlTypeIq2Xxs = 16,
  GgmlTypeIq2Xs  = 17,
  GgmlTypeIq3Xxs = 18,
  GgmlTypeIq1S   = 19,
  GgmlTypeIq4Nl  = 20,
  GgmlTypeIq3S   = 21,
  GgmlTypeIq2S   = 22,
  GgmlTypeIq4Xs  = 23,
  GgmlTypeI8      = 24,
  GgmlTypeI16     = 25,
  GgmlTypeI32     = 26,
  GgmlTypeI64     = 27,
  GgmlTypeF64     = 28,
  GgmlTypeIq1M   = 29,
  GgmlTypeBf16    = 30,
  // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
  // GGML_TYPE_Q4_0_4_8 = 32,
  // GGML_TYPE_Q4_0_8_8 = 33,
  GgmlTypeTq1_0   = 34,
  GgmlTypeTq2_0   = 35,
  // GGML_TYPE_IQ4_NL_4_4 = 36,
  // GGML_TYPE_IQ4_NL_4_8 = 37,
  // GGML_TYPE_IQ4_NL_8_8 = 38,
  GgmlTypeMxfp4   = 39, // MXFP4 (1 block)
  GgmlTypeCount   = 40,
}

pub struct Header {
  pub magic: [u8; 4],
  pub version: u32,
  pub tensor_count: u64,
  pub kv_count: u64,
}

pub struct Model {
  pub header: Header,
  pub config: ModelConfig,
  pub tensor_info: Vec<TensorInfo>,
  pub tensors: Vec<Tensor>,
  pub embedding: EmbeddingMatrix,
}

#[derive(Debug)]
pub struct ModelConfig {
  pub arch: String,
  pub context_length: u32,
  pub embedding_length: u32,
  pub head_count: u32,
  pub head_count_kv: u32,
  pub rope_freq_base: f32,
  pub metadata: HashMap<String, GgufValue>
}

#[derive(Debug)]
pub struct TensorInfo {
  pub name: String,
  pub n_dims: u32,
  pub dims: Vec<u64>,
  pub dtype: u32,
  pub offset: u64
}

#[derive(Debug)]
pub enum TensorData {
  F32(Vec<f32>),
  Q8_0(Vec<Q8Block>)
}

#[derive(Debug)]
pub struct Tensor {
  pub ggml_type: GgmlType,
  pub dimensions: Vec<u64>,
  pub data: TensorData
}
