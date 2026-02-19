// Implement BPE tokenizer

use std::{collections::HashMap, usize};
use crate::types::Model;

pub struct Tokenizer {
  vocabulary: HashMap<String, u64>,
  merges: HashMap<(String, String), usize>,
  special_tokens: Vec<String> // should this be a &str?
}

impl Tokenizer {

  pub fn new(model: &Model) -> Tokenizer {
    let mut vocab = HashMap::new();
    for (i, token) in model.tokenizer_tokens.iter().enumerate() {
      vocab.insert(token.clone(), i as u64);
    }

    let mut merges = HashMap::new();
    for (i, merge) in model.tokenizer_merges.iter().enumerate() {
      let pair: Vec<&str> = merge.split(" ").collect();

      merges.insert((pair[0].to_string(), pair[1].to_string()), i);
    }

    // let pattern_str = match model.tokenizer_pre.as_str() {
    //     "lfm2" | "llama3" => r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+"#.to_string(),
    //     "gpt2" => r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#.to_string(),
    //     _ => model.tokenizer_pre.to_string() // If it's already a regex, use it
    // };

    let special_tokens: Vec<String> = model.tokenizer_tokens.iter()
      .zip(model.tokenizer_token_types.iter())
      .filter(|(_, t)| **t == 3)
      .map(|(token, _)| token.clone())
      .collect();

    Tokenizer { vocabulary: vocab, merges, special_tokens }
  }

  pub fn tokenize(&self, text: &str) -> Vec<u64> {
    let mut token_ids = Vec::new();
    
    // First, split off any special tokens
    let segments = Tokenizer::split_on_special_tokens(text, &self.special_tokens);
    
    for segment in segments {
      if let Some(&id) = self.vocabulary.get(&segment) {
        // It's a special token, emit directly
        token_ids.push(id);
      } else {
          // Replace spaces with ▁, prepend ▁
        let processed = segment.replace(' ', "▁");
        
        // Split into individual characters
        let chars: Vec<String> = processed.chars().map(|c| c.to_string()).collect();
        
        // Run BPE merges (your existing merge fn should work here)
        let merged = self.merge(chars);
        
        for token in merged {
          if let Some(&id) = self.vocabulary.get(&token) {
              token_ids.push(id);
          }
        }
      }
    }
    
    token_ids
  }

  fn split_on_special_tokens(text: &str, special_tokens: &[String]) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut i = 0;
    let bytes = text.as_bytes();

    while i < text.len() {
        let remaining = &text[i..];
        
        // Try to match a special token (longest first)
        let mut matched: Option<String> = None;
        for token in special_tokens {
            if remaining.starts_with(token.as_str()) {
                match &matched {
                    Some(prev) if token.len() <= prev.len() => {},
                    _ => matched = Some(token.clone()),
                }
            }
        }

        if let Some(token) = matched {
            if !current.is_empty() {
                segments.push(current.clone());
                current.clear();
            }
            segments.push(token.clone());
            i += token.len();
        } else {
            current.push(text[i..].chars().next().unwrap());
            i += text[i..].chars().next().unwrap().len_utf8();
        }
    }

    if !current.is_empty() {
        segments.push(current);
    }

    segments
  }

  pub fn get_token_by_id(&self, token_str: &str) -> Option<u64> {
    self.vocabulary.get(token_str).copied()
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

  pub fn detokenize(token: &str) -> String {
    // Handle byte tokens like <0x0A>, <0xFF>, etc.
    if token.starts_with("<0x") && token.ends_with(">") {
        let hex = &token[3..token.len() - 1];
        if let Ok(byte) = u8::from_str_radix(hex, 16) {
            return String::from(byte as char);
        }
    }
    
    // Replace ▁ with space
    token.replace("▁", " ")
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