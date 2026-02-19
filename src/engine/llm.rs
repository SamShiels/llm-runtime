use crate::{engine::{kv_cache::KVCache, math::{matmul_vec, rms_normalize, sigmoid, softmax, swish, rope_rotate}, tokenizer::Tokenizer}, types::{AttentionWeights, FfnWeights, LayerType, Model, ShortConvWeights}};

enum LayerState {
  Attn(KVCache),
  Conv(Vec<f32>)
}

pub fn infer(model: &Model, query: String) -> String {
  let tokenizer = Tokenizer::new(&model);
  for (i, token) in model.tokenizer_tokens.iter().enumerate() {
    if token.contains("user") || token.contains("assistant") || token.contains("system") {
        println!("{}: {:?}", i, token);
    }
}
  let mut token_ids = Vec::new();
  token_ids.push(1); // <s>
  // <|user|> — look up its ID from the vocabulary
  token_ids.push(1); // <s>
  token_ids.append(&mut tokenizer.tokenize("<|user|>\n"));
  token_ids.append(&mut tokenizer.tokenize(query.trim()));
  token_ids.push(2); // </s>
  token_ids.append(&mut tokenizer.tokenize("\n<|assistant|>\n"));

  let mut layer_states: Vec<LayerState> = model.layers.iter()
    .map(|layer| {
      match &layer.layer_type {
        LayerType::Attention(_) => {
          let head_dim = model.config.embedding_length as usize / model.config.head_count as usize;
          LayerState::Attn(KVCache::new(head_dim, model.config.head_count_kv as usize, model.config.context_length as usize))
        },
        LayerType::ShortConv(shortconv) => {
          let half_dim = model.config.embedding_length as usize; 
          let total_weights = shortconv.conv.len();
          let kernel_size = total_weights / half_dim; // Usually 3
          
          // Buffer needs to hold [kernel_size * half_dim]
          LayerState::Conv(vec![0.0f32; kernel_size * half_dim])
        }
      }
    })
    .collect();

  // Process all prompt tokens, saving logits from the last one.
  let mut logits = Vec::new();
  for &token_id in &token_ids {
    logits = run_transformer(token_id, &model, &mut layer_states);
  }

  // Generation loop: sample from logits, then run the chosen token.
  loop {
    let mut next_token = 0u64;
    let mut best_score = f32::NEG_INFINITY;
    for (i, &score) in logits.iter().enumerate() {
      if score > best_score {
        best_score = score;
        next_token = i as u64;
      }
    }

    if next_token == model.eos_token_id {
      break;
    }

    let next_token_string = Tokenizer::detokenize(&model.tokenizer_tokens[next_token as usize].clone());
    println!("{} {}", next_token_string, next_token);

    logits = run_transformer(next_token, &model, &mut layer_states);
  }

  query
}

fn run_transformer(current_token: u64, model: &Model, kv_caches: &mut Vec<LayerState>) -> Vec<f32> {
  let mut embedding = lookup_embedding(current_token, &model.embedding.data, model.embedding.embedding_dim);

  let n_heads = model.config.head_count as usize;
  let head_dims = model.embedding.embedding_dim / n_heads;

  // println!("Embedding before: {}", embedding[0]);
  for (i, layer) in model.layers.iter().enumerate() {
    match (&layer.layer_type, &mut kv_caches[i]) {
      (LayerType::Attention(attn_weights), LayerState::Attn(kv_cache)) => {
          let normed = rms_normalize(&embedding, &attn_weights.norm);
          //println!("Normed attn: {}", normed[0]);

          let attention_out = attention(kv_cache, &attn_weights, &normed, head_dims, n_heads, model.config.head_count_kv as usize, model.config.rope_freq_base);
          //println!("attention out: {}", attention_out[0]);

          embedding = embedding.iter().zip(&attention_out).map(|(a, b)| a + b).collect();
        },
      (LayerType::ShortConv(conv_weights), LayerState::Conv(conv_buffer)) => {
          let normed = rms_normalize(&embedding, &conv_weights.norm);
          //println!("Normed conv: {}", normed[0]);

          let conv_out = shortconv(conv_buffer, &conv_weights, &normed);
          //println!("conv out: {}", conv_out[0]);

          embedding = embedding.iter().zip(&conv_out).map(|(a, b)| a + b).collect();
        },
      _ => unreachable!(),
    }
    // println!("Embedding after: {}", embedding[0]);
    let normed = rms_normalize(&embedding, &layer.ffn.norm);
    //println!("Normed: {}", normed[0]);

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

fn attention(kv_cache: &mut KVCache, attention_layer: &AttentionWeights, input: &[f32], head_size: usize, n_heads: usize, n_kv_heads: usize, freq_base: f32) -> Vec<f32> {
  let q = matmul_vec(
    &attention_layer.q,
    input);

  let k = matmul_vec(
    &attention_layer.k,
    input);

  let v = matmul_vec(
    &attention_layer.v,
    input);

  // Write each KV head into the cache once
  for kv_h in 0..n_kv_heads {
    let mut k_head = k[kv_h * head_size..(kv_h + 1) * head_size].to_vec();
    rope_rotate(&mut k_head, kv_cache.current_token_idx, freq_base);
    kv_cache.append_k_at(kv_h, &k_head);
    kv_cache.append_v_at(kv_h, &v[kv_h * head_size..(kv_h + 1) * head_size]);
  }

  let mut all_heads_context = vec![0.0; input.len()];

  for h in 0..n_heads {
    let kv_h = h * n_kv_heads / n_heads;

    let mut q_head = q[h * head_size..(h + 1) * head_size].to_vec();
    rope_rotate(&mut q_head, kv_cache.current_token_idx, freq_base);

    let k_history = kv_cache.get_k_history(kv_h);
    let v_history = kv_cache.get_v_history(kv_h);

    let mut scores: Vec<f32> = Vec::new();
    for k_past in k_history.chunks(head_size) {
      let dot: f32 = q_head.iter().zip(k_past).map(|(a, b)| a * b).sum();
      scores.push(dot / (head_size as f32).sqrt());
    }

    softmax(&mut scores);

    let mut head_context = vec![0.0; head_size];
    for (score, v_past) in scores.iter().zip(v_history.chunks(head_size)) {
      for (i, val) in v_past.iter().enumerate() {
        head_context[i] += score * val;
      }
    }

    all_heads_context[h * head_size..(h + 1) * head_size].copy_from_slice(&head_context);
  }

  kv_cache.current_token_idx += 1;

  matmul_vec(&attention_layer.output, &all_heads_context)
}

fn shortconv(buffer: &mut Vec<f32>, shortconv_layer: &ShortConvWeights, input: &[f32]) -> Vec<f32> {
  let projected = matmul_vec(&shortconv_layer.in_proj, input);
  let dim = projected.len() / 3;

  let x = &projected[..dim];          // The part to be convolved
  let gate = &projected[dim..2*dim];  // The gate
  let value = &projected[2*dim..];

  // println!("Proj Len: {}, Dim: {}, Remainder: {}", projected.len(), dim, projected.len() % 3);

  let kernel_size = shortconv_layer.conv.len() / dim;

  // Shift buffer back one time step, dropping the oldest
  let len = buffer.len();

  buffer.copy_within(0..len - dim, dim);
  buffer[0..dim].copy_from_slice(x);

  // Depthwise conv: each channel c accumulates over kernel_size time steps
  let mut conv_out = vec![0.0f32; dim];
  for c in 0..dim {
    let mut sum = 0.0;
    for k in 0..kernel_size {
      // Data is at [k * dim + c] (Strided by dim in buffer)
      let val = buffer[k * dim + c];
      
      // Weight is at [c * kernel_size + k] (Strided by kernel_size in weights)
      let idx = c * kernel_size + (kernel_size - 1 - k);
      let weight = shortconv_layer.conv[idx];
      
      sum += val * weight;
    }
    conv_out[c] = sum;
  }

  // Sigmoid gating
  for i in 0..dim {
    let gated_conv = conv_out[i] * swish(gate[i]); // Or sigmoid, check config
    
    // Multiply by the previously ignored Value branch
    conv_out[i] = gated_conv + value[i]; 
  }

  matmul_vec(&shortconv_layer.out_proj, &conv_out)
}

fn ffn(weights: &FfnWeights, input: &[f32]) -> Vec<f32> {
  let gate = matmul_vec(&weights.gate, &input);
  let up = matmul_vec(&weights.up, &input);

  let activated: Vec<f32> = gate.iter().zip(&up)
    .map(|(g, u)| swish(*g) * u)
    .collect();

  matmul_vec(&weights.down, &activated)
}