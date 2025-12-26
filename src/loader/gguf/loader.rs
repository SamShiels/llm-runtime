// Implement function to load up a gguf file, and another to parse the result
use std::{collections::HashMap, io::{Error, ErrorKind}};
use tokio::fs::File;

use crate::loader::gguf::reader::{GgufReader, GgufValue};

pub async fn load() {
  let file_contents = match open_file().await {
    Ok(bytes) => bytes,
    Err(err) => {
      print!("failed to open GGUF file: {err}");
      return;
    }
  };

  let length = file_contents.len();
  println!("{length}");
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

#[derive(Debug)]
struct ModelConfig {
  pub arch: String,
  pub context_length: u32,
  pub embedding_length: u32,
  pub metadata: HashMap<String, super::reader::GgufValue>
}

async fn open_file() -> Result<Vec<u8>, Error> {
  let file = File::open("models/tinyllama.gguf").await?;

  let mut gguf_reader = GgufReader::new(file);
  let header = Header::new(&mut gguf_reader).await?;
  println!("Magic: {:?}", String::from_utf8(header.magic.to_vec()).unwrap());
  println!("Version: {}", header.version);
  println!("Tensor count: {}", header.tensor_count);
  println!("KV count: {}", header.kv_count);

  read_kv(header.kv_count, &mut gguf_reader).await?;

  Ok(vec![])
}

async fn read_kv(count: u64, gguf_reader: &mut GgufReader) -> Result<ModelConfig, Error> {
  let mut arch: Option<String> = None;
  let mut context_length: Option<u32> = None;
  let mut embedding_length: Option<u32> = None;
  let mut metadata: HashMap<String, GgufValue> = HashMap::new();

  for k in 0..count {
    //println!("key index = {}", k);
    
    let key = gguf_reader.read_utf8().await?;
    println!("key = {}", key);

    let value_type = gguf_reader.read_u32().await?;
    println!("value_type = {}", value_type);

    let value = gguf_reader.get_value(value_type).await?;
    
    if k != 17 && k != 16 && k != 15 {
      println!("value = {:?}", value);
    }

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

  println!("{:?}", config);

  Ok(config)
}
