
pub struct KVCache {
  k: Vec<f32>,
  v: Vec<f32>,
  
  head_size: usize,
  context_len: usize,
  pub current_token_idx: usize
}

impl KVCache {

  pub fn new(head_size: usize, n_heads: usize, context_len: usize) -> Self {
    let total_capacity = n_heads * head_size * context_len;

    KVCache {
      k: vec![0.0; total_capacity],
      v: vec![0.0; total_capacity],
      head_size,
      context_len,
      current_token_idx: 0
    }
  }

  pub fn append_k_at(&mut self, head_index: usize, k_values: &[f32]) {
    let head_start = head_index * self.context_len * self.head_size;
    let token_offset = self.current_token_idx * self.head_size;
    let write_start = head_start + token_offset; 

    self.k[write_start..write_start + self.head_size].copy_from_slice(k_values);
  }

  pub fn append_v_at(&mut self, head_index: usize, v_values: &[f32]) {
    let head_start = head_index * self.context_len * self.head_size;
    let token_offset = self.current_token_idx * self.head_size;
    let write_start = head_start + token_offset; 

    self.v[write_start..write_start + self.head_size].copy_from_slice(v_values);
  }

  pub fn get_k_history(&self, head_index: usize) -> &[f32] {
    let head_start = head_index * self.context_len * self.head_size;
    let history_end = head_start + (self.current_token_idx + 1) * self.head_size;

    &self.k[head_start..history_end]
  }

  pub fn get_v_history(&self, head_index: usize) -> &[f32] {
    let head_start = head_index * self.context_len * self.head_size;
    let history_end = head_start + (self.current_token_idx + 1) * self.head_size;

    &self.v[head_start..history_end]
  }
}