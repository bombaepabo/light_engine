#[path = "../tokenizer.rs"]
pub mod tokenizer;

use tokenizer::Tokenizer;
use std::path::Path;

fn main() {
    println!("=== Testing Dual-Mode Tokenizer ===");

    // 1. Test SentencePiece Mode using tokenizer.bin (Llama-2)
    let sp_path = "tokenizer.bin";
    if Path::new(sp_path).exists() {
        println!("Loading Llama SentencePiece tokenizer from {}...", sp_path);
        let tokenizer = Tokenizer::new(sp_path, 32000);
        let prompt = "Hello world! This is a test.";
        println!("Input text: {:?}", prompt);
        
        let tokens = tokenizer.encode(prompt, true);
        println!("Encoded tokens (BOS=true): {:?}", tokens);
        
        let decoded = tokenizer.decode(0, tokens[0]); // decode first
        println!("First decoded token: {:?}", decoded);

        let mut reconstructed = String::new();
        for &t in &tokens {
            reconstructed.push_str(&tokenizer.decode(0, t));
        }
        println!("Reconstructed text: {:?}", reconstructed);
        println!("SentencePiece Test: Passed!\n");
    } else {
        println!("Skipping SentencePiece test: {} not found.", sp_path);
    }

    // 2. Test TikToken Mode (Qwen) if the file exists
    let qwen_path = "qwen.tiktoken";
    if Path::new(qwen_path).exists() {
        println!("Loading Qwen TikToken tokenizer from {}...", qwen_path);
        let tokenizer = Tokenizer::new(qwen_path, 151936);
        let prompt = "Hello world! This is a test.";
        println!("Input text: {:?}", prompt);
        
        let tokens = tokenizer.encode(prompt, false);
        println!("Encoded tokens (BOS=false): {:?}", tokens);
        
        let mut reconstructed = String::new();
        for &t in &tokens {
            reconstructed.push_str(&tokenizer.decode(0, t));
        }
        println!("Reconstructed text: {:?}", reconstructed);
        
        if reconstructed == prompt {
            println!("TikToken Test: Passed!\n");
        } else {
            println!("TikToken Test: Failed (Mismatched reconstruction)!\n");
        }
    } else {
        println!("Skipping Qwen TikToken test: {} not found yet.", qwen_path);
        println!("To run the Qwen test, place your qwen.tiktoken file in this directory.");
    }
}
