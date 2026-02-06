// Implement BPE tokenizer

use std::{collections::HashMap, usize};
use regex::Regex;

use crate::loader::gguf::{loader::Model, reader::GgufValue};

pub struct Tokenizer {
  vocabulary: HashMap<String, u64>,
  merges: HashMap<(String, String), usize>,
  byte_to_unicode: HashMap<u8, String>,
  pre_tokenization: String // should this be a &str?
}

impl Tokenizer {

  pub fn new(model: &Model) -> Tokenizer {
    let tokens_gguf= model.config.metadata.get("tokenizer.ggml.tokens").unwrap();
    let mut vocab = HashMap::new();

    if let GgufValue::Array(token_list) = tokens_gguf {
      for (i, token) in token_list.iter().enumerate() {
        if let GgufValue::String(s) = token {
          vocab.insert(s.clone(), i as u64);
        }
      }
    }

    let merges_gguf= model.config.metadata.get("tokenizer.ggml.merges").unwrap();
    let mut merges = HashMap::new();

    if let GgufValue::Array(merge_list) = merges_gguf {
      for (i, merge) in merge_list.iter().enumerate() {
        if let GgufValue::String(s) = merge {
          let pair: Vec<&str> = s.split(" ").collect();

          merges.insert((pair[0].to_string(), pair[1].to_string()), i);
        }
      }
    }

    let pre_tokenization_gguf = model.config.metadata.get("tokenizer.ggml.pre").unwrap();
    let pattern_str = if let GgufValue::String(s) = pre_tokenization_gguf {
        match s.as_str() {
            "lfm2" | "llama3" => r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+"#.to_string(),
            "gpt2" => r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#.to_string(),
            _ => s.to_string() // If it's already a regex, use it
        }
    } else {
        panic!("pre-tokenization not a string")
    };

    let map = get_byte_to_unicode_map();

    Tokenizer { vocabulary: vocab, merges, pre_tokenization: pattern_str, byte_to_unicode: map }
  }

  pub fn tokenize(&self, query: &str) -> Vec<u64> {
    let pattern = Regex::new(&self.pre_tokenization).unwrap();

    let mut token_ids = Vec::new();
    for m in pattern.find_iter(query) {
      let chunk = m.as_str();
      let byte_tokens: Vec<String> = chunk.as_bytes()
        .iter()
        .map(|b| self.byte_to_unicode.get(b).unwrap().clone())
        .collect();

      let tokens = self.merge(byte_tokens);

      println!("{:?}", tokens);

      for token in tokens {
        if let Some(&id) = self.vocabulary.get(&token) {
          token_ids.push(id);
        }
      }
    }

    token_ids
  }

  fn merge(&self, mut words: Vec<String>) -> Vec<String> {
    if words.len() < 2 {
      return words;
    }

    loop {
      let mut best_pair: Option<(usize, usize)> = None;
      let mut highest_priority = usize::MAX;

      for i in 0..words.len() - 1 {
        let pair = (words[i].clone(), words[i+1].clone());
        if let Some(&rank) = self.merges.get(&pair) {
          if rank < highest_priority {
            highest_priority = rank;
            best_pair = Some((i, i+1));
          }
        }
      };

      match best_pair {
        Some(i) => {
          let new_token = format!("{}{}", words[i.0], words[i.1]);
          words[i.0] = new_token;
          words.remove(i.1);
        }
        None => break,
      }
    }

    words
  }
}

fn get_byte_to_unicode_map() -> HashMap<u8, String> {
  let mut map = HashMap::new();
  let mut n = 0;

  for b in 0..=255 {
    let is_printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);

    let c = if is_printable {
      b as char
    } else {
      char::from_u32(256 + n).expect("Invalid unicode!")
    };

    if !is_printable { n += 1; }
    map.insert(b, c.to_string());
  }

  map
}