// Implement function to load up a gguf file, and another to parse the result
use std::{collections::HashMap, io::{Error, ErrorKind}};
use tokio::fs::File;
use half::f16;

use crate::{loader::gguf::reader::GgufReader, types::GgufValue};
use super::types::{Header, Model as LoaderModel, ModelConfig, TensorInfo, TensorData as LoaderTensorData, Tensor as LoaderTensor, GgmlType};
use crate::types::{Q8Block, Model, Config, Tensor, TensorData, EmbeddingMatrix};
use crate::engine::math;

pub async fn load() -> Result<Model, Error>{
  let loader_model = open_file().await?;

  // Convert config
  let config = Config {
    arch: loader_model.config.arch,
    context_length: loader_model.config.context_length,
    embedding_length: loader_model.config.embedding_length,
  };

  // Convert tensors - tensors and tensor_info are in same order
  let tensors: Vec<Tensor> = loader_model.tensors.into_iter()
    .enumerate()
    .map(|(i, loader_tensor)| {
      let info = &loader_model.tensor_info[i];

      // Convert tensor data
      let data = match loader_tensor.data {
        LoaderTensorData::Q8_0(blocks) => TensorData::Q8(blocks),
        LoaderTensorData::F32(values) => TensorData::F32(values),
      };

      Tensor {
        name: info.name.clone(),
        dims: info.dims.clone(),
        data,
      }
    })
    .collect();

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
    tensors,
    tokenizer_tokens,
    tokenizer_merges,
    tokenizer_pre,
    embedding: loader_model.embedding,
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

  let alignment = config.metadata.get("general.alignment")
    .and_then(|val| {
      if let GgufValue::Uint32(u) = val { Some(*u as u64) } else { None }
    })
    .unwrap_or(32);

  // Move the file pointer to the nearest 32-byte position ahead of where we are now
  let file_position = gguf_reader.get_position().await?;
  let tensor_start_position = ((file_position as f64 / alignment as f64).ceil() * alignment as f64) as u64;
  let file_pointer_movement = tensor_start_position;
  gguf_reader.seek_pointer_absolute(file_pointer_movement).await?;

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

          for (i, &byte) in chunk[0..32].iter().enumerate() {
            quantized[i] = byte as i8;
          }
          let scale_bytes: [u8; 2] = chunk[32..34].try_into().unwrap();
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