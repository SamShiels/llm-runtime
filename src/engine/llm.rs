use crate::{engine::tokenizer::Tokenizer, loader::gguf::loader::Model};

// pub fn infer(model: &Model, query: String) -> String {
//   let tokeizer = Tokenizer::new(&model);
//   let token_ids = tokeizer.tokenize(&query.trim());

//   for token_id in token_ids {
//     let embedding = lookup_embedding(token_id, embedding_matrix, embed_dim);
//   };
// }

// fn lookup_embedding(token_id: u64, embedding_matrix: &[f32], embed_dim: usize) -> Vec<f32> {
//   let start = token_id as usize * embed_dim;
//   let end = start + embed_dim;

//   embedding_matrix[start..end].to_vec()
// }