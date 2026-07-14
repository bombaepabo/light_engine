pub mod model;
pub mod loader;
pub mod tokenizer;
pub mod math;
pub mod state;

use std::fs::File;
use std::io::{Write, stdout};
use memmap2::Mmap;
use crate::state::{RunState, KVCache};

// Simple Argmax Sampler: Finds the index of the highest logit score
fn sample_argmax(logits: &[f32]) -> usize {
    let mut max_val = logits[0];
    let mut max_idx = 0;
    for i in 1..logits.len() {
        if logits[i] > max_val {
            max_val = logits[i];
            max_idx = i;
        }
    }
    max_idx
}

fn main() {
    // let model_path = "tinyllama_1.1b_q8.bin";
    // let tokenizer_path = "tokenizer.bin";
    let model_path = "qwen_3b_q8.bin";
    let tokenizer_path = "qwen.tiktoken"; // Maximum number of tokens to generate
    
    // Configurable parameters
    let steps = 250; // Maximum number of tokens to generate

    println!("Loading model: {}...", model_path);
    let file = File::open(model_path).expect("Failed to open model file");
    let mmap = unsafe { Mmap::map(&file).expect("Failed to memory map file") };

    let config = loader::read_config(&mmap);
    let weights = loader::load_weights(&mmap, &config);
    
    let tokenizer = tokenizer::Tokenizer::new(tokenizer_path, config.vocab_size);
    println!("Tokenizer: {}", tokenizer_path);
    println!("Vocab size: {}, Layers: {}, Dim: {}, Heads: {}, KV Heads: {}\n",
        config.vocab_size, config.n_layers, config.dim, config.n_heads, config.n_kv_heads);
    println!("Model loaded and ready!\n");

    // Loop the chat interface infinitely
    loop {
        // 1. Prompt User for Input
        print!("Enter prompt: ");
        stdout().flush().unwrap();
        
        let mut prompt_input = String::new();
        let bytes_read = std::io::stdin().read_line(&mut prompt_input).expect("Failed to read line");
        if bytes_read == 0 {
            break;
        }
        let prompt_input = prompt_input.trim();
        
        // Break out of the chat loop if the user types exit/quit
        if prompt_input == "exit" || prompt_input == "quit" {
            println!("Goodbye!");
            break;
        }
        
        if prompt_input.is_empty() {
            continue;
        }

        // Auto-detect model type by vocab size and apply the correct prompt template
        let is_qwen = config.vocab_size == 151936;
        let prompt = if is_qwen {
            // Qwen-2.5: ChatML format with Qwen identity
            format!(
                "<|im_start|>system\nYou are Qwen, a helpful assistant created by Alibaba Cloud.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                prompt_input
            ).replace("\r", "")
        } else {
            // TinyLlama / Llama-2 chat format
            format!(
                "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
                prompt_input
            )
        };

        // Pre-allocate temporary buffers (resets state for each turn)
        let mut state = RunState::new(&config);
        let mut kv_cache = KVCache::new(&config);

        // 2. Encode the prompt into token IDs
        // Llama needs an explicit BOS token; Qwen handles context via ChatML tags
        let prompt_tokens = tokenizer.encode(&prompt, !is_qwen);
        
        let mut next_token = 0;
        
        // 3. Stage 1: Feed the prompt tokens into the model to build the KV Cache
        for pos in 0..prompt_tokens.len() {
            let current_token = prompt_tokens[pos];
            weights.forward(current_token, pos, &config, &mut state, &mut kv_cache);
            
            next_token = if pos < prompt_tokens.len() - 1 {
                prompt_tokens[pos + 1]
            } else {
                sample_argmax(&state.logits)
            };
        }

        // 4. Print the first generated token (skip control tags)
        let piece = tokenizer.decode(prompt_tokens[prompt_tokens.len() - 1], next_token);
        let mut tail = piece.clone(); // rolling buffer of recent output

        // 5. Stage 2: Generation Loop
        // We keep a rolling tail buffer of the last ~20 chars so we can detect
        // stop sequences (<|im_end|>, <|im_start|>, <|user|>, <|system|>, <|assistant|>, </s>) 
        // even if they are split across multiple tokens.
        let stop_seqs = [
            "<|im_end|>", 
            "<|im_start|>", 
            "<|user|>", 
            "<|system|>", 
            "<|assistant|>",
            "</s>",
            "<|end|>",
            "<|endoftext|>"
        ];

        let is_stop = |buf: &str| stop_seqs.iter().any(|s| buf.contains(s));

        if !is_stop(&tail) {
            print!("{}", piece);
            stdout().flush().unwrap();
        }

        let eos_id = if is_qwen { 151645 } else { 2 };

        'generate: for pos in prompt_tokens.len()..steps {
            weights.forward(next_token, pos, &config, &mut state, &mut kv_cache);

            let prev_token = next_token;
            next_token = sample_argmax(&state.logits);

            // Stop on EOS token or special ChatML tags
            if next_token == eos_id || next_token == 151643 || next_token == 32000 || next_token == 32001 || next_token == 32002 {
                break 'generate;
            }

            let piece = tokenizer.decode(prev_token, next_token);

            // Append to rolling tail and keep only the last 20 chars
            tail.push_str(&piece);
            if tail.len() > 20 {
                let trim = tail.len() - 20;
                tail.drain(..trim);
            }

            // Stop if any stop sequence appears in the rolling tail
            if is_stop(&tail) {
                break 'generate;
            }

            print!("{}", piece);
            stdout().flush().unwrap();
        }
        println!("\n");
    }
}