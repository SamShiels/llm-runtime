use crate::loader::gguf::loader::Q8_0Block;

pub fn dequantize(blocks: &[Q8_0Block]) -> Vec<f32> {
  let total_elements = blocks.len() * 32;
  let mut dequantized_values: Vec<f32> = Vec::with_capacity(total_elements);

  for block in blocks {
    let scale = block.scale;
    let quantized_values = &block.quantized_samples;

    for i in quantized_values {
      let dequantized_value = *i as f32 * scale;
      dequantized_values.push(dequantized_value);
    }
  }

  dequantized_values
}

pub fn matmul_vec(
  a: &[f32],
  b: &[f32],
  out_dimensions: usize,
  in_dimensions: usize
) -> Vec<f32> {
  let mut output = vec![0.0; out_dimensions];

  for i in 0..out_dimensions {
    let mut sum = 0.0;
    for j in 0..in_dimensions {
      sum += a[i * in_dimensions + j] * b[j];
    }
    output[i] = sum;
  }

  output
}