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
    let model_path = "tinyllama_1.1b_q8.bin";
    let tokenizer_path = "tokenizer.bin";
    
    // Configurable parameters
    let steps = 150; // Maximum number of tokens to generate

    println!("Loading Llama-2 Model...");
    let file = File::open(model_path).expect("Failed to open model file");
    let mmap = unsafe { Mmap::map(&file).expect("Failed to memory map file") };

    let config = loader::read_config(&mmap);
    let weights = loader::load_weights(&mmap, &config);
    
    let tokenizer = tokenizer::Tokenizer::new(tokenizer_path, config.vocab_size);
    println!("Model and Tokenizer loaded successfully!\n");

    // Loop the chat interface infinitely
    loop {
        // 1. Prompt User for Input
        print!("Enter prompt: ");
        stdout().flush().unwrap();
        
        let mut prompt_input = String::new();
        std::io::stdin().read_line(&mut prompt_input).expect("Failed to read line");
        let prompt_input = prompt_input.trim();
        
        // Break out of the chat loop if the user types exit/quit
        if prompt_input == "exit" || prompt_input == "quit" {
            println!("Goodbye!");
            break;
        }
        
        if prompt_input.is_empty() {
            continue;
        }

        // Automatically format the raw input into the ChatML template
        let prompt = format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt_input
        );

        // Pre-allocate temporary buffers (resets state for each turn)
        let mut state = RunState::new(&config);
        let mut kv_cache = KVCache::new(&config);

        // 2. Encode the prompt into token IDs
        let prompt_tokens = tokenizer.encode(&prompt, true);
        
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

        // 4. Print the first generated token
        let piece = tokenizer.decode(prompt_tokens[prompt_tokens.len() - 1], next_token);
        print!("{}", piece);
        stdout().flush().unwrap();

        // 5. Stage 2: Generation Loop
        for pos in prompt_tokens.len()..steps {
            weights.forward(next_token, pos, &config, &mut state, &mut kv_cache);
            
            let prev_token = next_token;
            next_token = sample_argmax(&state.logits);
            
            if next_token == 2 {
                break;
            }
            
            let piece = tokenizer.decode(prev_token, next_token);
            print!("{}", piece);
            stdout().flush().unwrap();
        }
        println!("\n");
    }
}