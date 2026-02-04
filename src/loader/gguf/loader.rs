use core::num;
// Implement function to load up a gguf file, and another to parse the result
use std::{collections::HashMap, io::{Error, ErrorKind}, pin::Pin};
use tokio::fs::File;

use crate::loader::gguf::reader::{GgufReader, GgufValue};

#[derive(Debug)]
enum GgmlType {
  GgmlTypeF32     = 0,
  GgmlTypeF16     = 1,
  GgmlTypeQ4_0  = 2,
  GgmlTypeQ4_1    = 3,
  // GGML_TYPE_Q4_2 = 4, support has been removed
  // GGML_TYPE_Q4_3 = 5, support has been removed
  GgmlTypeQ5_0    = 6,
  GgmlTypeQ5_1    = 7,
  GgmlTypeQ8_0    = 8,
  GgmlTypeQ8_1    = 9,
  GgmlTypeQ2K    = 10,
  GgmlTypeQ3K    = 11,
  GgmlTypeQ4K    = 12,
  GgmlTypeQ5K    = 13,
  GgmlTypeQ6K    = 14,
  GgmlTypeQ8K    = 15,
  GgmlTypeIq2Xxs = 16,
  GgmlTypeIq2Xs  = 17,
  GgmlTypeIq3Xxs = 18,
  GgmlTypeIq1S   = 19,
  GgmlTypeIq4Nl  = 20,
  GgmlTypeIq3S   = 21,
  GgmlTypeIq2S   = 22,
  GgmlTypeIq4Xs  = 23,
  GgmlTypeI8      = 24,
  GgmlTypeI16     = 25,
  GgmlTypeI32     = 26,
  GgmlTypeI64     = 27,
  GgmlTypeF64     = 28,
  GgmlTypeIq1M   = 29,
  GgmlTypeBf16    = 30,
  // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
  // GGML_TYPE_Q4_0_4_8 = 32,
  // GGML_TYPE_Q4_0_8_8 = 33,
  GgmlTypeTq1_0   = 34,
  GgmlTypeTq2_0   = 35,
  // GGML_TYPE_IQ4_NL_4_4 = 36,
  // GGML_TYPE_IQ4_NL_4_8 = 37,
  // GGML_TYPE_IQ4_NL_8_8 = 38,
  GgmlTypeMxfp4   = 39, // MXFP4 (1 block)
  GgmlTypeCount   = 40,
}

pub async fn load() -> Result<Model, Error>{
  // let file_contents = match open_file().await {
  //   Ok(model) => model,
  //   Err(err) => {
  //     print!("failed to open GGUF file: {err}");
  //   }
  // }

  // println!("{:?}", file_contents.tensors);

  Ok(open_file().await?)
}

pub struct Header {
  pub magic: [u8; 4],
  pub version: u32,
  pub tensor_count: u64,
  pub kv_count: u64,
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

pub struct Model {
  pub header: Header,
  pub config: ModelConfig,
  pub tensor_info: Vec<TensorInfo>,
  pub tensors: Vec<Tensor>
}

#[derive(Debug)]
pub struct ModelConfig {
  pub arch: String,
  pub context_length: u32,
  pub embedding_length: u32,
  pub metadata: HashMap<String, super::reader::GgufValue>
}

#[derive(Debug)]
pub struct TensorInfo {
  name: String,
  n_dims: u32,
  dims: Vec<u64>,
  dtype: u32,
  offset: u64
}

#[derive(Debug)]
pub struct Q8_0Block {
  quantized_samples: [i8; 32],
  scale: f32,
}

#[derive(Debug)]
pub enum TensorData {
  F32(Vec<f32>),
  Q8_0(Vec<Q8_0Block>)
}

#[derive(Debug)]
pub struct Tensor {
  ggml_type: GgmlType,
  dimensions: Vec<u64>,
  data: TensorData
}

async fn open_file() -> Result<Model, Error> {
  let file = File::open("models/tinyllama.gguf").await?;

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

  // Move the file pointer to the nearest 32-byte position ahead of where we are now
  let file_position = gguf_reader.get_position().await?;
  let tensor_start_position = ((file_position as f64 / 32.0).ceil() * 32.0) as u64;
  let file_pointer_movement = tensor_start_position;
  gguf_reader.seek_pointer_absolute(file_pointer_movement).await?;

  let mut tensors: Vec<Tensor> = Vec::new();

  for i in 0..header.tensor_count {
    let info = &tensor_info[i as usize];
    let tensor = read_tensor(&mut gguf_reader, &info).await?;

    tensors.push(tensor);
  }

  Ok(Model {
    header,
    config,
    tensor_info,
    tensors
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
    println!("value_type = {}", value_type);

    let value = gguf_reader.get_gguf_value(value_type).await?;
    //println!("{:?}", value);

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

  //println!("{:?}", config);

  Ok(config)
}

async fn read_tensor_info(gguf_reader: &mut GgufReader) -> Result<TensorInfo, Error> {
  let tensor_name = gguf_reader.read_utf8().await?;

  println!("Tensor = {}", tensor_name);
  let n_dims = gguf_reader.read_u32().await?;
  println!("n_dims = {}", n_dims);
  let mut dims = Vec::with_capacity(n_dims.try_into().unwrap());
  for _ in 0..n_dims {
    dims.push(gguf_reader.read_u64().await?);
  }
  println!("dims = {:?}", dims);
  let dtype = gguf_reader.read_u32().await?;
  println!("dtype = {}", dtype);
  let offset = gguf_reader.read_u64().await?;
  println!("offset = {}", offset);

  Ok(TensorInfo { name: tensor_name, n_dims, dims, dtype, offset })
}

// async fn read_block(gguf_reader: &mut GgufReader, dims: Vec<u64>, cur_dimension: u32, num_dimension: u32) -> Pin<Box<dyn Future<Output = Result<Q8_0Block, Error>> + '_>> {
//   Box::pin(async move {
//     let dimension_length = dims[cur_dimension];
//     for i in 0..dimension_length {
//       read_block(gguf_reader, dims, cur_dimension, num_dimension).await?;
//     }
//   })
// }

async fn read_tensor(gguf_reader: &mut GgufReader, info: &TensorInfo) -> Result<Tensor, Error> {
  let num_elements: u64 = info.dims.iter().product();
  let _ = gguf_reader.seek_pointer_absolute(info.offset).await;

  let num_bytes = match info.dtype {
      0 => num_elements * 4,
      8 => {
        let num_blocks = (num_elements + 31) / 32;
        num_blocks * 36
      }
      _ => panic!("Unsupported dtype: {}", info.dtype)
  };

  let data = gguf_reader.read_exact_vec(num_bytes as usize).await?;

  let ggml_type = match info.dtype {
    0 => GgmlType::GgmlTypeF32,
    8 => GgmlType::GgmlTypeQ8_0,
    _ => panic!("Unsupported dtype: {}", info.dtype)
  };

  println!("{}", data.len());

  // Just assume it's Q8_0 for now. We will fill out more dtypes later
  
  let parsed_data = match info.dtype {
      0 => {
        let f32_data: Vec<f32> = data.chunks(4)
          .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
          .collect();

        TensorData::F32(f32_data)
      },
      8 => {
        let mut blocks: Vec<Q8_0Block> = Vec::new();
        for chunk in data.chunks(36) {
          let mut quantized = [0i8; 32];

          for (i, &byte) in chunk[0..32].iter().enumerate() {
            quantized[i] = byte as i8;
          }
          let scale_bytes: [u8; 4] = chunk[32..36].try_into().unwrap();
          let scale = f32::from_le_bytes(scale_bytes);

          let block = Q8_0Block {
            quantized_samples: quantized,
            scale
          };

          blocks.push(block);
        }

        TensorData::Q8_0(blocks)
      },
      _ => panic!("Unsupported dtype: {}", info.dtype)
  };

  Ok(Tensor { ggml_type, dimensions: info.dims.clone(), data: parsed_data })
}