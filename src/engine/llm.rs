use crate::{engine::{math::rms_normalize, tokenizer::Tokenizer}, types::Model};

pub fn infer(model: &Model, query: String) -> String {
  let tokeizer = Tokenizer::new(&model);
  let token_ids = tokeizer.tokenize(&query.trim());

  for token_id in token_ids {
    let embedding = lookup_embedding(token_id, &model.embedding.data, model.embedding.embedding_dim);

    //rms_normalize(&embedding, &norm_weight);
  };

  query
}

fn lookup_embedding(token_id: u64, embedding_matrix: &Vec<f32>, embed_dim: usize) -> Vec<f32> {
  let start = token_id as usize * embed_dim;
  let end = start + embed_dim;

  embedding_matrix[start..end].to_vec()
}
