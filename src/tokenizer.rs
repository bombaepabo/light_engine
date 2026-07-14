use std::fs::File;
use std::io::{Read, BufReader};

pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    vocab_scores: Vec<f32>,
    _max_token_length: u32,
}

impl Tokenizer {
    pub fn new(tokenizer_path: &str, vocab_size: usize) -> Self {
        let file = File::open(tokenizer_path).expect("Failed to open tokenizer file");
        let mut reader = BufReader::new(file);
        let mut max_token_length_bytes = [0u8; 4];
        reader.read_exact(&mut max_token_length_bytes).expect("Failed to read header");
        let max_token_length = u32::from_le_bytes(max_token_length_bytes);
        let mut vocab = Vec::with_capacity(vocab_size);
        let mut vocab_scores = Vec::with_capacity(vocab_size);
        for i in 0..vocab_size {
            // 1. Read score (4 bytes float). If we hit EOF, pad the rest and break.
            let mut score_bytes = [0u8; 4];
            if reader.read_exact(&mut score_bytes).is_err() {
                // Pad the remaining tokens with placeholders
                for _ in i..vocab_size {
                    vocab.push(b"<unk>".to_vec());
                    vocab_scores.push(0.0);
                }
                break;
            }
            let score = f32::from_le_bytes(score_bytes);
            vocab_scores.push(score);
            // 2. Read token length (4 bytes int)
            let mut len_bytes = [0u8; 4];
            reader.read_exact(&mut len_bytes).expect("Failed to read len");
            let len = i32::from_le_bytes(len_bytes) as usize;
            // 3. Read token string bytes
            let mut token_bytes = vec![0u8; len];
            reader.read_exact(&mut token_bytes).expect("Failed to read token bytes");
            vocab.push(token_bytes);
        }
        Self {
            vocab,
            vocab_scores,
            _max_token_length: max_token_length,
        }
    }

    // Decode a single token ID back to text bytes
    pub fn decode(&self, _prev_token: usize, token: usize) -> String {
        let mut bytes = self.vocab[token].clone();
        
        // Handle special byte representations (raw byte tokens are represented as "<0xXX>")
        if bytes.len() == 6 && bytes.starts_with(b"<0x") && bytes.ends_with(b">") {
            if let Ok(hex_str) = std::str::from_utf8(&bytes[3..5]) {
                if let Ok(byte_val) = u8::from_str_radix(hex_str, 16) {
                    bytes = vec![byte_val];
                }
            }
        }

        String::from_utf8_lossy(&bytes).into_owned()
    }

    // Find a token in vocabulary matching raw bytes
    fn find_token(&self, text: &[u8]) -> Option<usize> {
        self.vocab.iter().position(|t| t == text)
    }

    // Encode a text prompt into token IDs
    pub fn encode(&self, text: &str, bos: bool) -> Vec<usize> {
        let mut tokens = Vec::new();

        // Prepend BOS (Beginning of String) token if requested
        if bos {
            tokens.push(1); // ID 1 is BOS in Llama-2
        }

        if text.is_empty() {
            return tokens;
        }

         // Initialize with individual bytes
        let mut byte_tokens: Vec<(usize, usize)> = Vec::new(); // (vocab_id, byte_length)
        for &byte in text.as_bytes() {
            if let Some(id) = self.find_token(&[byte]) {
                byte_tokens.push((id, 1));
            } else {
                // Fall back to hex representation for control bytes (e.g. "<0x0A>" for newline)
                let fallback_str = format!("<0x{:02X}>", byte).into_bytes();
                if let Some(id) = self.find_token(&fallback_str) {
                    byte_tokens.push((id, 1));
                } else {
                    panic!("Byte not found in tokenizer vocabulary: {}", byte);
                }
            }
        }

        // Greedy BPE Merge Loop
        loop {
            let mut best_score = -1e10f32;
            let mut best_idx = None;
            let mut best_id = None;

            for i in 0..(byte_tokens.len() - 1) {
                let mut pair_bytes = Vec::new();
                let left_id = byte_tokens[i].0;
                pair_bytes.extend_from_slice(&self.vocab[left_id]);

                let right_id = byte_tokens[i+1].0;
                pair_bytes.extend_from_slice(&self.vocab[right_id]);

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

        for (id, _) in byte_tokens {
            tokens.push(id);
        }

        tokens
    }
}