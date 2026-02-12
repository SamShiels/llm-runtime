use crate::{engine::{kv_cache::KVCache, math::{matmul_vec, rms_normalize, softmax, swish}, tokenizer::Tokenizer}, types::{AttentionWeights, FfnWeights, LayerType, Model}};

pub fn infer(model: &Model, query: String) -> String {
  let tokeizer = Tokenizer::new(&model);
  let token_ids = tokeizer.tokenize(&query.trim());

  let mut kv_caches: Vec<KVCache> = model.layers.iter()
    .map(|layer| {
      let kv_dim = match &layer.layer_type {
        LayerType::Attention(attn) => attn.k.len() / model.config.embedding_length as usize
      };
      KVCache::new(kv_dim, model.config.context_length as usize)
    })
    .collect();

  let mut current_token = 0u64;
  for token_id in token_ids {
    current_token = token_id;

    run_transformer(current_token, &model, &mut kv_caches);
  };

  loop {
    let logits = run_transformer(current_token, &model, &mut kv_caches);

    let mut next_token = 0u64;
    let mut best_score = f32::NEG_INFINITY;
    for (i, &score) in logits.iter().enumerate() {
      if score > best_score {
        best_score = score;
        next_token = i as u64;
      }
    }

    let next_token_string = model.tokenizer_tokens[next_token as usize].clone();

    println!("{}", next_token_string);

    current_token = next_token;

    if next_token == model.eos_token_id { 
      break; 
    }
  }

  query
}

fn run_transformer(current_token: u64, model: &Model, kv_caches: &mut Vec<KVCache>) -> Vec<f32> {
  let mut embedding = lookup_embedding(current_token, &model.embedding.data, model.embedding.embedding_dim);

  for (i, layer) in model.layers.iter().enumerate() {
    match &layer.layer_type {
      LayerType::Attention(attn) => {
        let normed = rms_normalize(&embedding, &attn.norm);

        let attention_out = attention(&mut kv_caches[i], &attn, &normed);

        embedding = embedding.iter().zip(&attention_out).map(|(a, b)| a + b).collect();
      }
    }

    let normed = rms_normalize(&embedding, &layer.ffn.norm);

    let ffn_out = ffn(&layer.ffn, &normed);

    embedding = embedding.iter().zip(&ffn_out).map(|(a, b)| a + b).collect();
  }

  let normed = rms_normalize(&embedding, &model.output_norm);
  let logits = matmul_vec(&model.output_weight, &normed);


  logits
}

fn lookup_embedding(token_id: u64, embedding_matrix: &Vec<f32>, embed_dim: usize) -> Vec<f32> {
  let start = token_id as usize * embed_dim;
  let end = start + embed_dim;

  embedding_matrix[start..end].to_vec()
}

fn attention(kv_cache: &mut KVCache, attention_layer: &AttentionWeights, input: &[f32]) -> Vec<f32> {
  let q = matmul_vec(
    &attention_layer.q,
    input);

  let k = matmul_vec(
    &attention_layer.k,
    input);

  let v = matmul_vec(
    &attention_layer.v,
    input);

  kv_cache.append_k(&k);
  kv_cache.append_v(&v);

  let dim = *kv_cache.dim();

  let mut scores: Vec<f32> = Vec::with_capacity(dim);

  for k in kv_cache.keys().chunks(dim) {
    let dot: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum();
    let scale = 1.0 / (dim as f32).sqrt();
    scores.push(dot * scale);
  }

  softmax(&mut scores);

  let mut context = vec![0.0f32; dim];
  for (score, v) in scores.iter().zip(kv_cache.values().chunks(dim)) {
    for (c, vi) in context.iter_mut().zip(v.iter()) {
      *c += score * vi;
    }
  }

  matmul_vec(&attention_layer.output, &context)
}

fn ffn(weights: &FfnWeights, input: &[f32]) -> Vec<f32> {
  let gate = matmul_vec(&weights.gate, &input);
  let up = matmul_vec(&weights.up, &input);

  let activated: Vec<f32> = gate.iter().zip(&up)
    .map(|(g, u)| swish(*g) * u)
    .collect();

  matmul_vec(&weights.down, &activated)
}