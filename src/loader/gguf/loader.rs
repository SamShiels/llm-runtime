// Implement function to load up a gguf file, and another to parse the result
use std::{collections::HashMap, io::{Error, ErrorKind}};
use tokio::fs::File;
use half::f16;

use crate::{loader::gguf::reader::GgufReader, types::{GgufValue, ShortConvWeights}};
use super::types::{Header, Model as LoaderModel, ModelConfig, TensorInfo, TensorData as LoaderTensorData, Tensor as LoaderTensor, GgmlType};
use crate::types::{Q8Block, Model, Config, EmbeddingMatrix, AttentionWeights, FfnWeights, LayerType, TransformerLayer};
use crate::engine::math;

pub async fn load() -> Result<Model, Error>{
  let loader_model = open_file().await?;

  // Convert config
  let config = Config {
    arch: loader_model.config.arch,
    context_length: loader_model.config.context_length,
    embedding_length: loader_model.config.embedding_length,
  };

  // Build a name→f32 map, dequantizing Q8 tensors as needed
  let mut tensor_map: HashMap<String, Vec<f32>> = HashMap::new();
  for (info, loader_tensor) in loader_model.tensor_info.iter().zip(loader_model.tensors.iter()) {
    let data = match &loader_tensor.data {
      LoaderTensorData::F32(values) => values.clone(),
      LoaderTensorData::Q8_0(blocks) => math::dequantize(blocks),
    };
    tensor_map.insert(info.name.clone(), data);
  }

  // Derive number of layers from tensor names (max blk.N.* index + 1)
  let num_layers = tensor_map.keys()
    .filter_map(|name| {
      let rest = name.strip_prefix("blk.")?;
      let idx_end = rest.find('.')?;
      rest[..idx_end].parse::<usize>().ok()
    })
    .max()
    .map(|m| m + 1)
    .unwrap_or(0);

  // Build transformer layers
  let mut layers: Vec<TransformerLayer> = Vec::with_capacity(num_layers);
  for i in 0..num_layers {
    let layer_type = if let Some(q) = tensor_map.remove(&format!("blk.{i}.attn_q.weight")) {
      LayerType::Attention(AttentionWeights {
        q,
        k:      tensor_map.remove(&format!("blk.{i}.attn_k.weight")).unwrap_or_default(),
        v:      tensor_map.remove(&format!("blk.{i}.attn_v.weight")).unwrap_or_default(),
        output: tensor_map.remove(&format!("blk.{i}.attn_output.weight")).unwrap_or_default(),
        norm:   tensor_map.remove(&format!("blk.{i}.attn_norm.weight")).unwrap_or_default(),
      })
    } else if let Some(conv) = tensor_map.remove(&format!("blk.{i}.shortconv.conv.weight"))  {
      LayerType::ShortConv(ShortConvWeights {
        conv,
        in_proj:  tensor_map.remove(&format!("blk.{i}.shortconv.in_proj.weight")).unwrap_or_default(),
        out_proj: tensor_map.remove(&format!("blk.{i}.shortconv.out_proj.weight")).unwrap_or_default(),
        norm:     tensor_map.remove(&format!("blk.{i}.attn_norm.weight")).unwrap_or_default(),
      })
    } else {
      // ShortConv layer - placeholder until implemented
      LayerType::Attention(AttentionWeights {
        q: Vec::new(), k: Vec::new(), v: Vec::new(), output: Vec::new(),
        norm: tensor_map.remove(&format!("blk.{i}.attn_norm.weight")).unwrap_or_default(),
      })
    };

    let ffn = FfnWeights {
      gate: tensor_map.remove(&format!("blk.{i}.ffn_gate.weight")).unwrap_or_default(),
      up:   tensor_map.remove(&format!("blk.{i}.ffn_up.weight")).unwrap_or_default(),
      down: tensor_map.remove(&format!("blk.{i}.ffn_down.weight")).unwrap_or_default(),
      norm: tensor_map.remove(&format!("blk.{i}.ffn_norm.weight")).unwrap_or_default(),
    };

    layers.push(TransformerLayer { layer_type, ffn });
  }

  let output_norm = tensor_map.remove("output_norm.weight")
      .or_else(|| tensor_map.remove("norm.weight"))
      .or_else(|| tensor_map.remove("token_embd_norm.weight")) // <--- THE FIX
      .expect("Critical: Could not find any final normalization tensor!");

  // 3. Get the Output Head (Map 'token_embd' to 'output_weight')
  // If 'output.weight' or 'lm_head.weight' is missing, CLONE the embedding weights.
  let output_weight = tensor_map.remove("output.weight")
      .or_else(|| tensor_map.remove("lm_head.weight"))
      .unwrap_or_else(|| loader_model.embedding.data.clone()); // <--- THE FIX (Weight Tying)

  let eos_token_id = loader_model.config.metadata
    .get("tokenizer.ggml.eos_token_id")
    .and_then(|v| match v {
      GgufValue::Uint32(id) => Some(*id as u64),
      GgufValue::Uint64(id) => Some(*id),
      _ => None,
    })
    .unwrap_or(2);

  // Extract tokenizer data from metadata
  let tokenizer_tokens = loader_model.config.metadata
    .get("tokenizer.ggml.tokens")
    .and_then(|v| {
      if let GgufValue::Array(arr) = v {
        Some(arr.iter().filter_map(|item| {
          if let GgufValue::String(s) = item {
            Some(s.clone())
          } else {
            None
          }
        }).collect())
      } else {
        None
      }
    })
    .unwrap_or_else(Vec::new);

  let tokenizer_merges = loader_model.config.metadata
    .get("tokenizer.ggml.merges")
    .and_then(|v| {
      if let GgufValue::Array(arr) = v {
        Some(arr.iter().filter_map(|item| {
          if let GgufValue::String(s) = item {
            Some(s.clone())
          } else {
            None
          }
        }).collect())
      } else {
        None
      }
    })
    .unwrap_or_else(Vec::new);

  let tokenizer_pre = loader_model.config.metadata
    .get("tokenizer.ggml.pre")
    .and_then(|v| {
      if let GgufValue::String(s) = v {
        Some(s.clone())
      } else {
        None
      }
    })
    .unwrap_or_else(|| String::new());

  Ok(Model {
    config,
    layers,
    tokenizer_tokens,
    tokenizer_merges,
    tokenizer_pre,
    embedding: loader_model.embedding,
    output_norm,
    output_weight,
    eos_token_id,
  })
}

impl Header {
  pub async fn new(reader: &mut GgufReader) -> Result<Self, Error> {
    let magic_vec = reader.read_exact_vec(4).await?;
    let magic: [u8; 4] = magic_vec
      .try_into()
      .map_err(|_| Error::new(ErrorKind::InvalidData, "invalid GGUF magic size"))?;
    let version = reader.read_u32().await?;
    let tensor_count = reader.read_u64().await?;
    let kv_count = reader.read_u64().await?;

    Ok(Self {
      magic,
      version,
      tensor_count,
      kv_count
    })
  }
}

async fn open_file() -> Result<LoaderModel, Error> {
  let file = File::open("models/LFM2.5-1.2B-Instruct-Q8_0.gguf").await?;

  let mut gguf_reader = GgufReader::new(file);
  let header = Header::new(&mut gguf_reader).await?;
  println!("Magic: {:?}", String::from_utf8(header.magic.to_vec()).unwrap());
  println!("Version: {}", header.version);
  println!("Tensor count: {}", header.tensor_count);
  println!("KV count: {}", header.kv_count);

  let config = read_kv(header.kv_count, &mut gguf_reader).await?;

  let mut tensor_info: Vec<TensorInfo> = Vec::new();

  for _ in 0..header.tensor_count {
    let info = read_tensor_info(&mut gguf_reader).await?;
    tensor_info.push(info);
  }

  println!("\n=== Tensor Names ===");
  for info in &tensor_info {
    println!("  {}", info.name);
  }

  let alignment = config.metadata.get("general.alignment")
    .and_then(|val| {
      if let GgufValue::Uint32(u) = val { Some(*u as u64) } else { None }
    })
    .unwrap_or(32);

  let mut current_pos = gguf_reader.get_position().await?;
  while current_pos % alignment != 0 {
      gguf_reader.read_u8().await?; // Discard padding byte
      current_pos += 1;
  }
  
  let tensor_start_position = current_pos;

  // Move the file pointer to the nearest 32-byte position ahead of where we are now
  gguf_reader.seek_pointer_absolute(tensor_start_position).await?;

  let mut tensors: Vec<LoaderTensor> = Vec::new();
  let mut embedding: Option<EmbeddingMatrix> = None;

  for i in 0..header.tensor_count {
    let info = &tensor_info[i as usize];
    let tensor = read_tensor(&mut gguf_reader, &info, &tensor_start_position).await?;

    // Extract embedding matrix
    if info.name == "token_embd.weight" {
      let embedding_dim = info.dims[0] as usize;
      let vocab_size = info.dims[1] as usize;

      let data = match &tensor.data {
        LoaderTensorData::F32(values) => values.clone(),
        LoaderTensorData::Q8_0(blocks) => math::dequantize(blocks),
      };

      embedding = Some(EmbeddingMatrix {
        data,
        embedding_dim,
        vocab_size,
      });
    }

    tensors.push(tensor);
  }

  let embedding = embedding.ok_or_else(|| Error::new(ErrorKind::InvalidData, "token_embd.weight not found"))?;

  Ok(LoaderModel {
    header,
    config,
    tensor_info,
    tensors,
    embedding,
  })
}

async fn read_kv(count: u64, gguf_reader: &mut GgufReader) -> Result<ModelConfig, Error> {
  let mut arch: Option<String> = None;
  let mut context_length: Option<u32> = None;
  let mut embedding_length: Option<u32> = None;
  let mut metadata: HashMap<String, GgufValue> = HashMap::new();

  for _ in 0..count {    
    let key = gguf_reader.read_utf8().await?;
    println!("key = {}", key);

    let value_type = gguf_reader.read_u32().await?;
    // println!("value_type = {}", value_type);

    let value = gguf_reader.get_gguf_value(value_type).await?;
    // println!("{:?}", value);

    if key.as_str() == "general.architecture" {
      if let GgufValue::String(s) = value {
        arch = Some(s);
      } else {
        return Err(Error::new(ErrorKind::InvalidData, "general.architecture not a string"));
      }
      continue;
    }

    if let Some(ref arch_name) = arch {
      if key == format!("{arch_name}.context_length") {
        if let GgufValue::Uint32(i) = value {
          context_length = Some(i);
          continue;
        } else {
          return Err(Error::new(ErrorKind::InvalidData, "context_length not 4 bytes"));
        }
      } else if key == format!("{arch_name}.embedding_length") {
        if let GgufValue::Uint32(i) = value {
          embedding_length = Some(i);
          continue;
        } else {
          return Err(Error::new(ErrorKind::InvalidData, "embedding_length not 4 bytes"));
        }
      }
    }
    metadata.insert(key.clone(), value);

  }

  let arch = arch.ok_or_else(|| Error::new(ErrorKind::InvalidData, "missing general.architecture"))?;
  let context_length = context_length.ok_or_else(|| Error::new(ErrorKind::InvalidData, "missing (arch).context_length"))?;
  let embedding_length = embedding_length.ok_or_else(|| Error::new(ErrorKind::InvalidData, "missing (arch).embedding_length"))?;

  let config = ModelConfig {
    arch,
    context_length,
    embedding_length,
    metadata
  };

  Ok(config)
}

async fn read_tensor_info(gguf_reader: &mut GgufReader) -> Result<TensorInfo, Error> {
  let tensor_name = gguf_reader.read_utf8().await?;

  let n_dims = gguf_reader.read_u32().await?;
  let mut dims = Vec::with_capacity(n_dims.try_into().unwrap());
  for _ in 0..n_dims {
    dims.push(gguf_reader.read_u64().await?);
  }
  let dtype = gguf_reader.read_u32().await?;
  let offset = gguf_reader.read_u64().await?;

  Ok(TensorInfo { name: tensor_name, n_dims, dims, dtype, offset })
}

async fn read_tensor(gguf_reader: &mut GgufReader, info: &TensorInfo, tensor_start_position: &u64) -> Result<LoaderTensor, Error> {
  let num_elements: u64 = info.dims.iter().product();
  let _ = gguf_reader.seek_pointer_absolute(tensor_start_position + info.offset).await;

  println!("{}", info.dtype);
  let num_bytes = match info.dtype {
      0 => num_elements * 4,
      8 => {
        let num_blocks = (num_elements + 31) / 32;
        num_blocks * 34
      }
      _ => panic!("Unsupported dtype: {}", info.dtype)
  };

  let data = gguf_reader.read_exact_vec(num_bytes as usize).await?;

  let ggml_type = match info.dtype {
    0 => GgmlType::GgmlTypeF32,
    8 => GgmlType::GgmlTypeQ8_0,
    _ => panic!("Unsupported dtype: {}", info.dtype)
  };

  let parsed_data = match info.dtype {
      0 => {
        let f32_data: Vec<f32> = data.chunks(4)
          .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
          .collect();

        LoaderTensorData::F32(f32_data)
      },
      8 => {
        let mut blocks: Vec<Q8Block> = Vec::new();
        for chunk in data.chunks(34) {
          let mut quantized = [0i8; 32];

          let scale_bytes: [u8; 2] = chunk[0..2].try_into().unwrap();
          for (i, &byte) in chunk[2..34].iter().enumerate() {
            quantized[i] = byte as i8;
          }

          let scale = f16::from_le_bytes(scale_bytes).to_f32();

          let block = Q8Block {
            values: quantized,
            scale
          };

          blocks.push(block);
        }

        LoaderTensorData::Q8_0(blocks)
      },
      _ => panic!("Unsupported dtype: {}", info.dtype)
  };

  Ok(LoaderTensor { ggml_type, dimensions: info.dims.clone(), data: parsed_data })
}