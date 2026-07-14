use std::fs::File;
use std::io::{Read, BufReader};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TokenizerMode {
    SentencePiece,
    TikToken,
}

pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    vocab_scores: Vec<f32>,
    mode: TokenizerMode,
    _max_token_length: u32,
}

// Pure Rust, dependency-free Base64 decoder for loading .tiktoken files
fn base64_decode(s: &str) -> Vec<u8> {
    let mut table = [0u8; 256];
    for (i, &c) in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".iter().enumerate() {
        table[c as usize] = i as u8;
    }
    
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'=' { break; }
        let b0 = table[bytes[i] as usize] as u32;
        let b1 = table[bytes[i+1] as usize] as u32;
        let b2 = if bytes[i+2] == b'=' { 0 } else { table[bytes[i+2] as usize] as u32 };
        let b3 = if bytes[i+3] == b'=' { 0 } else { table[bytes[i+3] as usize] as u32 };
        
        let val = (b0 << 18) | (b1 << 12) | (b2 << 6) | b3;
        out.push(((val >> 16) & 0xFF) as u8);
        if bytes[i+2] != b'=' {
            out.push(((val >> 8) & 0xFF) as u8);
        }
        if bytes[i+3] != b'=' {
            out.push((val & 0xFF) as u8);
        }
        i += 4;
    }
    out
}

impl Tokenizer {
    pub fn new(tokenizer_path: &str, vocab_size: usize) -> Self {
        if tokenizer_path.ends_with(".tiktoken") {
            // Load TikToken text format
            let file = File::open(tokenizer_path).expect("Failed to open tokenizer file");
            let reader = BufReader::new(file);
            let mut vocab = vec![Vec::new(); vocab_size];
            let mut vocab_scores = vec![0.0f32; vocab_size];
            
            use std::io::BufRead;
            for line in reader.lines() {
                let line = line.expect("Failed to read line");
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() == 2 {
                    let token_bytes = base64_decode(parts[0]);
                    let rank = parts[1].parse::<usize>().expect("Failed to parse rank");
                    if rank < vocab_size {
                        vocab[rank] = token_bytes;
                        // TikToken BPE merges lowest rank first.
                        // By storing negative rank as score, we can use the same max-score BPE merge loop!
                        vocab_scores[rank] = -(rank as f32);
                    }
                }
            }
            
            Self {
                vocab,
                vocab_scores,
                mode: TokenizerMode::TikToken,
                _max_token_length: 128,
            }
        } else {
            // Load standard Llama SentencePiece binary format (.bin)
            let file = File::open(tokenizer_path).expect("Failed to open tokenizer file");
            let mut reader = BufReader::new(file);
            let mut max_token_length_bytes = [0u8; 4];
            reader.read_exact(&mut max_token_length_bytes).expect("Failed to read header");
            let max_token_length = u32::from_le_bytes(max_token_length_bytes);
            let mut vocab = Vec::with_capacity(vocab_size);
            let mut vocab_scores = Vec::with_capacity(vocab_size);
            for i in 0..vocab_size {
                let mut score_bytes = [0u8; 4];
                if reader.read_exact(&mut score_bytes).is_err() {
                    for _ in i..vocab_size {
                        vocab.push(b"<unk>".to_vec());
                        vocab_scores.push(0.0);
                    }
                    break;
                }
                let score = f32::from_le_bytes(score_bytes);
                vocab_scores.push(score);
                let mut len_bytes = [0u8; 4];
                reader.read_exact(&mut len_bytes).expect("Failed to read len");
                let len = i32::from_le_bytes(len_bytes) as usize;
                let mut token_bytes = vec![0u8; len];
                reader.read_exact(&mut token_bytes).expect("Failed to read token bytes");
                vocab.push(token_bytes);
            }
            Self {
                vocab,
                vocab_scores,
                mode: TokenizerMode::SentencePiece,
                _max_token_length: max_token_length,
            }
        }
    }

    // Decode a single token ID back to text bytes
    pub fn decode(&self, _prev_token: usize, token: usize) -> String {
        if self.mode == TokenizerMode::TikToken {
            if token == 151644 {
                return "<|im_start|>".to_string();
            }
            if token == 151645 {
                return "<|im_end|>".to_string();
            }
        } else {
            if self.vocab.len() == 32003 {
                if token == 32000 {
                    return "<|system|>".to_string();
                }
                if token == 32001 {
                    return "<|user|>".to_string();
                }
                if token == 32002 {
                    return "<|assistant|>".to_string();
                }
            }
        }
        if token >= self.vocab.len() {
            return String::new();
        }
        let mut bytes = self.vocab[token].clone();
        
        // Handle special byte representations for SentencePiece
        if self.mode == TokenizerMode::SentencePiece {
            if bytes.len() == 6 && bytes.starts_with(b"<0x") && bytes.ends_with(b">") {
                if let Ok(hex_str) = std::str::from_utf8(&bytes[3..5]) {
                    if let Ok(byte_val) = u8::from_str_radix(hex_str, 16) {
                        bytes = vec![byte_val];
                    }
                }
            }
        }
 
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn encode_segment(&self, text: &str) -> Vec<usize> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut byte_tokens: Vec<(usize, usize)> = Vec::new(); // (vocab_id, byte_length)
        
        match self.mode {
            TokenizerMode::TikToken => {
                for &byte in text.as_bytes() {
                    let id = self.find_token(&[byte]).unwrap_or_else(|| {
                        panic!("Byte {} not found in TikToken vocabulary", byte);
                    });
                    byte_tokens.push((id, 1));
                }
            }
            TokenizerMode::SentencePiece => {
                for &byte in text.as_bytes() {
                    if let Some(id) = self.find_token(&[byte]) {
                        byte_tokens.push((id, 1));
                    } else {
                        let fallback_str = format!("<0x{:02X}>", byte).into_bytes();
                        if let Some(id) = self.find_token(&fallback_str) {
                            byte_tokens.push((id, 1));
                        } else {
                            panic!("Byte not found in tokenizer vocabulary: {}", byte);
                        }
                    }
                }
            }
        }

        // Greedy BPE Merge Loop (shared by both modes!)
        loop {
            let mut best_score = -1e10f32;
            let mut best_idx = None;
            let mut best_id = None;

            for i in 0..byte_tokens.len()-1 {
                let mut pair_bytes = Vec::new();
                pair_bytes.extend_from_slice(&self.vocab[byte_tokens[i].0]);
                pair_bytes.extend_from_slice(&self.vocab[byte_tokens[i+1].0]);

                if let Some(merged_id) = self.find_token(&pair_bytes) {
                    let score = self.vocab_scores[merged_id];
                    if score > best_score {
                        best_score = score;
                        best_idx = Some(i);
                        best_id = Some(merged_id);
                    }
                }
            }

            if let Some(idx) = best_idx {
                let merged_id = best_id.unwrap();
                byte_tokens[idx] = (merged_id, byte_tokens[idx].1 + byte_tokens[idx+1].1);
                byte_tokens.remove(idx + 1);
            } else {
                break;
            }
        }

        let mut tokens = Vec::new();
        for (id, _) in byte_tokens {
            tokens.push(id);
        }
        tokens
    }

    // Main encode method that parses special control tokens for both TikToken and SentencePiece
    pub fn encode(&self, text: &str, bos: bool) -> Vec<usize> {
        let special_mappings = if self.mode == TokenizerMode::TikToken {
            vec![
                ("<|im_start|>", 151644),
                ("<|im_end|>", 151645),
            ]
        } else if self.vocab.len() == 32003 {
            vec![
                ("</s>", 2),
            ]
        } else {
            Vec::new()
        };

        if !special_mappings.is_empty() {
            let mut tokens = Vec::new();
            if bos && self.mode == TokenizerMode::SentencePiece {
                tokens.push(1);
            }
            let mut current_pos = 0;
            
            while current_pos < text.len() {
                let text_slice = &text[current_pos..];
                
                let mut matched = false;
                for &(tag, id) in &special_mappings {
                    if text_slice.starts_with(tag) {
                        tokens.push(id);
                        current_pos += tag.len();
                        matched = matched || true;
                        break;
                    }
                }
                
                if matched {
                    continue;
                }
                
                let mut next_special = None;
                for &(tag, _) in &special_mappings {
                    if let Some(pos) = text_slice.find(tag) {
                        next_special = Some(next_special.map_or(pos, |curr: usize| curr.min(pos)));
                    }
                }
                
                let segment_len = next_special.unwrap_or(text_slice.len());
                let segment = &text_slice[..segment_len];
                
                tokens.extend(self.encode_segment(segment));
                current_pos += segment_len;
            }
            return tokens;
        }

        let mut tokens = Vec::new();
        if bos && self.mode == TokenizerMode::SentencePiece {
            tokens.push(1); 
        }
        tokens.extend(self.encode_segment(text));
        tokens
    }

    // Find a token in vocabulary matching raw bytes
    fn find_token(&self, text: &[u8]) -> Option<usize> {
        self.vocab.iter().position(|t| t == text)
    }
}