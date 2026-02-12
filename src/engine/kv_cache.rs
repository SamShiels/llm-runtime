
pub struct KVCache {
  k: Vec<f32>,
  v: Vec<f32>,
  
  kv_dim: usize,
}

impl KVCache {

pub fn new(kv_dim: usize, context_len: usize) -> Self {
    KVCache {
      k: Vec::with_capacity(context_len),
      v: Vec::with_capacity(context_len),
      kv_dim,
    }
  }

  pub fn append_k(&mut self, k_values: &[f32]) {
    self.k.extend_from_slice(k_values);
  }

  pub fn append_v(&mut self, v_values: &[f32]) {
    self.v.extend_from_slice(v_values);
  }

  pub fn keys(&self) -> &[f32] {
    &self.k
  }

  pub fn values(&self) -> &[f32] {
    &self.v
  }

  pub fn dim(&self) -> &usize {
    &self.kv_dim
  }
}