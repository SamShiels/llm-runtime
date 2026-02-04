use std::{io::{Error, ErrorKind, SeekFrom}, pin::Pin};

use tokio::{fs::File, io::{AsyncReadExt, AsyncSeekExt}};

#[derive(Debug, Clone)]
pub enum GgufValue {
  Uint8(u8),
  Int8(i8),
  Uint16(u16),
  Int16(i16),
  Uint32(u32),
  Int32(i32),
  Float32(f32),
  Bool(bool),
  String(String),
  Array(Vec<GgufValue>),
  Uint64(u64),
  Int64(i64),
  Float64(f64),
}

pub struct GgufReader {
  file: File
}

impl GgufReader {
  pub fn new(file: File) -> Self {
    Self {
      file
    }
  }

  pub async fn get_position(&mut self) -> Result<u64, Error> {
    self.file.stream_position().await
  }

  pub async fn read_f32(&mut self) -> Result<f32, Error> {
    let mut buffer = [0u8; 4];
    self.file.read_exact(&mut buffer).await?;
    Ok(f32::from_le_bytes(buffer))
  }

  pub async fn read_u32(&mut self) -> Result<u32, Error> {
    let mut buffer = [0u8; 4];
    self.file.read_exact(&mut buffer).await?;
    Ok(u32::from_le_bytes(buffer))
  }

  pub async fn read_u64(&mut self) -> Result<u64, Error> {
    let mut buffer = [0u8; 8];
    self.file.read_exact(&mut buffer).await?;
    Ok(u64::from_le_bytes(buffer))
  }

  pub async fn read_exact_vec(&mut self, size: usize) -> Result<Vec<u8>, Error> {
    let mut v = vec![0u8; size];
    self.file.read_exact(&mut v).await?;
    Ok(v)
  }

  pub async fn read_prefixed_bytes(&mut self) -> Result<Vec<u8>, Error> {
    let length = self.read_u64().await?;
    let length = usize::try_from(length).map_err(|_| Error::new(ErrorKind::InvalidData, "length too large"))?;

    self.read_exact_vec(length).await
  }

  pub async fn read_utf8(&mut self) -> Result<String, Error> {
    let bytes = self.read_prefixed_bytes().await?;
    String::from_utf8(bytes)
      .map_err(|_| Error::new(ErrorKind::InvalidData, "invalid utf8"))
  }

  pub async fn seek_pointer(&mut self, movement: i64) -> Result<(), Error> {
    self.file.seek(SeekFrom::Current(movement)).await?;
    Ok(())
  }

  pub async fn seek_pointer_absolute(&mut self, position: u64) -> Result<(), Error> {
    self.file.seek(SeekFrom::Start(position)).await?;
    Ok(())
  }

  async fn read_fixed<const N: usize>(&mut self) -> Result<Vec<u8>, Error> {
    let mut buf = vec![0u8; N];
    self.file.read_exact(&mut buf).await?;

    Ok(buf)
  }

  async fn read_string(&mut self) -> Result<Vec<u8>, Error> {
    self.read_prefixed_bytes().await
  }

  fn read_array(&mut self) -> Pin<Box<dyn Future<Output = Result<Vec<GgufValue>, Error>> + '_>> {
    Box::pin(async move {
      let elem_type = self.read_u32().await?;
      let elem_count = self.read_u64().await?;
      let count = usize::try_from(elem_count).map_err(|_| Error::new(ErrorKind::InvalidData, "array too large"))?;

      let mut items: Vec<GgufValue> = Vec::with_capacity(count);

      for _ in 0..count {
        let element = self.get_gguf_value(elem_type).await?;
        items.push(element);
      }

      Ok(items)
    })
  }

  pub fn get_gguf_value(&mut self, value_type: u32) -> Pin<Box<dyn Future<Output = Result<GgufValue, Error>> + '_>> {
    Box::pin(async move {
      let value = match value_type {
        0 => { // GGUF_TYPE_UINT8
          let bytes = self.read_fixed::<1>().await?;
          GgufValue::Uint8(u8::from_le_bytes([bytes[0]]))
        }
        1 => { // INT8
          let bytes = self.read_fixed::<1>().await?;
          GgufValue::Int8(i8::from_le_bytes([bytes[0]]))
        }
        2 => { // UINT16
          let bytes = self.read_fixed::<2>().await?;
          let arr: [u8; 2] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| Error::new(ErrorKind::InvalidData, "invalid uint16 length"))?;
          GgufValue::Uint16(u16::from_le_bytes(arr))
        }
        3 => { // INT16
          let bytes = self.read_fixed::<2>().await?;
          let arr: [u8; 2] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| Error::new(ErrorKind::InvalidData, "invalid int16 length"))?;
          GgufValue::Int16(i16::from_le_bytes(arr))
        }
        4 => { // UINT32
          let bytes = self.read_fixed::<4>().await?;
          let arr: [u8; 4] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| Error::new(ErrorKind::InvalidData, "invalid uint32 length"))?;
          GgufValue::Uint32(u32::from_le_bytes(arr))
        }
        5 => { // INT32
          let bytes = self.read_fixed::<4>().await?;
          let arr: [u8; 4] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| Error::new(ErrorKind::InvalidData, "invalid int32 length"))?;
          GgufValue::Int32(i32::from_le_bytes(arr))
        }
        6 => { // FLOAT32
          let bytes = self.read_fixed::<4>().await?;
          let arr: [u8; 4] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| Error::new(ErrorKind::InvalidData, "invalid float32 length"))?;
          GgufValue::Float32(f32::from_le_bytes(arr))
        }
        7 => { // BOOL (0 or 1)
          let bytes = self.read_fixed::<1>().await?;
          let val = match bytes[0] {
            0 => false,
            1 => true,
            _ => return Err(Error::new(ErrorKind::InvalidData, "invalid bool value")),
          };
          GgufValue::Bool(val)
        }
        8 => { // STRING: u64 length + bytes
          let s = self.read_utf8().await?;
          GgufValue::String(s)
        }
        9 => { // ARRAY: u32 elem type + u64 count + elements
          let elem_type = self.read_u32().await?;
          let elem_count = self.read_u64().await?;
          let count = usize::try_from(elem_count)
            .map_err(|_| Error::new(ErrorKind::InvalidData, "array too large"))?;
          let mut items = Vec::with_capacity(count);
          for _ in 0..count {
            let element = self.get_gguf_value(elem_type).await?;
            items.push(element);
          }
          GgufValue::Array(items)
        }
        10 => { // UINT64
          let bytes = self.read_fixed::<8>().await?;
          let arr: [u8; 8] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| Error::new(ErrorKind::InvalidData, "invalid uint64 length"))?;
          GgufValue::Uint64(u64::from_le_bytes(arr))
        }
        11 => { // INT64
          let bytes = self.read_fixed::<8>().await?;
          let arr: [u8; 8] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| Error::new(ErrorKind::InvalidData, "invalid int64 length"))?;
          GgufValue::Int64(i64::from_le_bytes(arr))
        }
        12 => { // FLOAT64
          let bytes = self.read_fixed::<8>().await?;
          let arr: [u8; 8] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| Error::new(ErrorKind::InvalidData, "invalid float64 length"))?;
          GgufValue::Float64(f64::from_le_bytes(arr))
        }
        _ => {
          println!("unknown value type");
          return Err(Error::new(ErrorKind::InvalidData, "unknown value type"));
        }
      };

      Ok(value)
    })
  }
}
