// Implement BPE tokenizer

use std::{collections::HashMap, f32::INFINITY, fmt::format};

use crate::loader::gguf::{loader::Model, reader::GgufValue};

pub struct Tokenizer {
  vocabulary: HashMap<String, u32>,
  merges: Vec<(String, String)>
}

impl Tokenizer {

  pub fn new(model: &Model) -> Tokenizer {
    let tokens_gguf= model.config.metadata.get("tokenizer.ggml.tokens").unwrap();
    let mut vocab = HashMap::new();

    if let GgufValue::Array(token_list) = tokens_gguf {
      for (i, token) in token_list.iter().enumerate() {
        if let GgufValue::String(s) = token {
          vocab.insert(s.clone(), i as u32);
        }
      }
    }

    let merges_gguf= model.config.metadata.get("tokenizer.ggml.merges").unwrap();
    let mut merges = Vec::new();

    if let GgufValue::Array(merge_list) = merges_gguf {
      for (_, merge) in merge_list.iter().enumerate() {
        if let GgufValue::String(s) = merge {
          let pair: Vec<&str> = s.split(" ").collect();

          merges.push((pair[0].to_string(), pair[1].to_string()));
        }
      }
    }

    Tokenizer { vocabulary: vocab, merges }
  }

  pub fn tokenize(&self, query: &str) -> Vec<u64> {

    let tokens = self.merge(query);

    println!("{:?}", tokens);

    Vec::new()
  }

  fn merge(&self, query: &str) -> Vec<String> {
    let mut tokens: Vec<String> = query.chars().map(|c| c.to_string()).collect();

    loop {
      let mut found_merge = false;
      let mut highest_priority = INFINITY;

      let mut removable_at: Option<usize> = None;

      for i in 0..tokens.len() - 2 {
        let token_1 = &tokens[i];
        let token_2 = &tokens[i + 1];

        let position = self.merges.iter().position(|(a, b)| a == token_1 && b == token_2);

        if let Some(index) = position {
          let index_float = index as f32;
          if index_float < highest_priority {
            highest_priority = index_float;

            removable_at = Some(i);
          }

          found_merge = true;
        }
      };

      if found_merge == true {
        if let Some(index) = removable_at {
          let merged = format!("{}{}", tokens[index], tokens[index + 1]);
          tokens[index] = merged;
          tokens.remove(index + 1);
        }
      }
      else {
        break;
      }
    }

    tokens
  }
}