// Common types used throughout the project

/// Quantized Q8_0 block (32 elements with a scale factor)
#[derive(Debug, Clone)]
pub struct Q8Block {
    pub scale: f32,
    pub values: [i8; 32],
}

/// Tensor data types
#[derive(Debug)]
pub enum TensorData {
    F32(Vec<f32>),
    Q8(Vec<Q8Block>),
}

/// Embedding matrix with dimensions
#[derive(Debug)]
pub struct EmbeddingMatrix {
    pub data: Vec<f32>,
    pub embedding_dim: usize,
    pub vocab_size: usize,
}

/// Tensor with metadata and data
#[derive(Debug)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub data: TensorData,
}

/// Model configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub arch: String,
    pub context_length: u32,
    pub embedding_length: u32,
}

/// Complete model with all tensors
#[derive(Debug)]
pub struct Model {
    pub config: Config,
    pub layers: Vec<TransformerLayer>,
    pub tokenizer_tokens: Vec<String>,
    pub tokenizer_merges: Vec<String>,
    pub tokenizer_pre: String,
    pub embedding: EmbeddingMatrix,
    pub output_norm: Vec<f32>,
    pub output_weight: Vec<f32>,
    pub eos_token_id: u64,
}

#[derive(Debug)]
pub struct AttentionWeights {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub output: Vec<f32>,
    pub norm: Vec<f32>,  // RMSNorm weights before attention
}

#[derive(Debug)]
pub struct ShortConvWeights {
    pub in_proj: Vec<f32>,
    pub conv: Vec<f32>,
    pub out_proj: Vec<f32>,
    pub norm: Vec<f32>,  // RMSNorm weights before shortconv (blk.N.attn_norm.weight)
}

#[derive(Debug)]
pub struct FfnWeights {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
    pub norm: Vec<f32>,  // RMSNorm weights before FFN
}

#[derive(Debug)]
pub enum LayerType {
    Attention(AttentionWeights),
    ShortConv(ShortConvWeights),  // Later
}

#[derive(Debug)]
pub struct TransformerLayer {
    pub layer_type: LayerType,
    pub ffn: FfnWeights,
}

#[derive(Debug, Clone)]
pub enum GgufValue {
  Uint8(u8),
  Int8(i8),
  Uint16(u16),
  Int16(i16),
  Uint32(u32),
  Int32(i32),
  Float32(f32),
  Bool(bool),
  String(String),
  Array(Vec<GgufValue>),
  Uint64(u64),
  Int64(i64),
  Float64(f64),
}