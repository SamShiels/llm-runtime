use core::f32;

use crate::types::Q8Block;

pub fn dequantize(blocks: &[Q8Block]) -> Vec<f32> {
  let total_elements = blocks.len() * 32;
  let mut dequantized_values: Vec<f32> = Vec::with_capacity(total_elements);

  for block in blocks {
    let scale = block.scale;
    let quantized_values = &block.values;

    for i in quantized_values {
      let dequantized_value = *i as f32 * scale;
      dequantized_values.push(dequantized_value);
    }
  }

  dequantized_values
}

pub fn matmul_vec(
  matrix: &[f32],
  vector: &[f32],
) -> Vec<f32> {
  let in_dim = vector.len();
  if in_dim == 0 {
    println!("Trying to divide {} by zero!", matrix.len())
  }
  let out_dim = matrix.len() / in_dim;

  let mut output = vec![0.0; out_dim];

  for i in 0..out_dim {
    let mut sum = 0.0;
    for j in 0..in_dim {
      sum += matrix[i * in_dim + j] * vector[j];
    }
    output[i] = sum;
  }

  output
}

fn mean_sqr_vec(a: &[f32], size: usize) -> f32 {
  let sum: f32 = a.iter().map(|v| v * v).sum();
  sum / size as f32
}

pub fn rms_normalize(input: &[f32], weights: &[f32]) -> Vec<f32> {
  let mean = mean_sqr_vec(&input, input.len());
  let rms = f32::sqrt(mean + 1e-6);

  let a_norm = 
    input.iter()
    .zip(weights.iter())
    .map(|(v, w)| v / rms * w)
    .collect();

  a_norm
}

pub fn softmax(scores: &mut Vec<f32>){
  let max: f32 = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
  let sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();

  for s in scores.iter_mut() {
    *s = (*s - max).exp() / sum;
  }
} 

pub fn sigmoid(x: f32) -> f32 {
  if x >= 0.0 {
    1.0 / (1.0 + (-x).exp())
  } else {
    let exp_x = x.exp();
    exp_x / (1.0 + exp_x)
  }
}

pub fn swish(x: f32) -> f32 {
  sigmoid(x) * x
}