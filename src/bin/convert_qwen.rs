use memmap2::Mmap;
use safetensors::SafeTensors;
use std::fs::File;
use std::io::{Write, BufWriter};
use std::path::Path;

// Helper to convert bfloat16 bytes directly to f32 floats.
fn bf16_to_f32(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let bits = (u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16;
        out.push(f32::from_bits(bits));
    }
    out
}

fn main() {
    println!("=== Qwen-2.5-3B-Instruct Safetensors to Bin Converter ===");

    let file1_path = "model-00001-of-00002.safetensors";
    let file2_path = "model-00002-of-00002.safetensors";

    if !Path::new(file1_path).exists() || !Path::new(file2_path).exists() {
        panic!("Missing safetensors files! Please make sure both model-00001 and model-00002 are downloaded.");
    }

    println!("Memory-mapping safetensors files...");
    let file1 = File::open(file1_path).unwrap();
    let mmap1 = unsafe { Mmap::map(&file1).unwrap() };
    let safetensors1 = SafeTensors::deserialize(&mmap1).unwrap();

    let file2 = File::open(file2_path).unwrap();
    let mmap2 = unsafe { Mmap::map(&file2).unwrap() };
    let safetensors2 = SafeTensors::deserialize(&mmap2).unwrap();

    // Helper closure to find a tensor in either file
    let get_tensor = |name: &str| {
        if let Ok(t) = safetensors1.tensor(name) {
            t
        } else if let Ok(t) = safetensors2.tensor(name) {
            t
        } else {
            panic!("Tensor not found in either file: {}", name);
        }
    };

    // Qwen-2.5-3B Config
    let dim: i32 = 2048;
    let hidden_dim: i32 = 11008;
    let n_layers: i32 = 36;
    let n_heads: i32 = 16;
    let n_kv_heads: i32 = 2; // actual Qwen-2.5-3B has 2 KV heads (GQA)
    let vocab_size: i32 = 151936; // padded
    let seq_len: i32 = 2048;      
    let head_size = (dim / n_heads) as usize;

    let output_path = "qwen_3b.bin";
    println!("Creating output file {}...", output_path);
    let out_file = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(out_file);

    // 1. Write the 28-byte header config (7 integers)
    println!("Writing header...");
    writer.write_all(&dim.to_le_bytes()).unwrap();
    writer.write_all(&hidden_dim.to_le_bytes()).unwrap();
    writer.write_all(&n_layers.to_le_bytes()).unwrap();
    writer.write_all(&n_heads.to_le_bytes()).unwrap();
    writer.write_all(&n_kv_heads.to_le_bytes()).unwrap();
    writer.write_all(&vocab_size.to_le_bytes()).unwrap();
    writer.write_all(&seq_len.to_le_bytes()).unwrap();

    // Helper to convert and write a tensor
    let mut write_tensor = |name: &str, pad_to_vocab: bool| {
        let t = get_tensor(name);
        let mut float_data = bf16_to_f32(t.data());
        
        if pad_to_vocab {
            let target_len = (vocab_size * dim) as usize;
            if float_data.len() < target_len {
                float_data.resize(target_len, 0.0f32);
            }
        }

        // Convert f32 slice to raw bytes and write
        let bytes = unsafe {
            std::slice::from_raw_parts(float_data.as_ptr() as *const u8, float_data.len() * 4)
        };
        writer.write_all(bytes).unwrap();
        println!(" - Wrote: {} (floats: {})", name, float_data.len());
    };

    // 2. Write Token Embedding Table
    write_tensor("model.embed_tokens.weight", true);

    // 3. Write layers sequentially
    for l in 0..n_layers {
        println!("Processing Layer {}/{}...", l + 1, n_layers);
        write_tensor(&format!("model.layers.{}.input_layernorm.weight", l), false);
        
        // Attention weights
        write_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", l), false);
        write_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", l), false);
        write_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", l), false);
        write_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", l), false);

        // Attention Biases (floats, need to convert from bf16)
        write_tensor(&format!("model.layers.{}.self_attn.q_proj.bias", l), false);
        write_tensor(&format!("model.layers.{}.self_attn.k_proj.bias", l), false);
        write_tensor(&format!("model.layers.{}.self_attn.v_proj.bias", l), false);

        // FFN layer norm and MLPs
        write_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", l), false);
        write_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", l), false);
        write_tensor(&format!("model.layers.{}.mlp.down_proj.weight", l), false);
        write_tensor(&format!("model.layers.{}.mlp.up_proj.weight", l), false);
    }

    // 4. Write final RMS Norm
    write_tensor("model.norm.weight", false);

    // 5. Compute and write Qwen RoPE frequencies (base frequency is 1,000,000)
    println!("Computing and writing Qwen RoPE frequency tables...");
    let base = 1000000.0f32;
    let mut inv_freq = Vec::with_capacity(head_size / 2);
    for i in (0..head_size).step_by(2) {
        inv_freq.push(1.0f32 / base.powf((i as f32) / (head_size as f32)));
    }

    let mut freqs_cos = Vec::with_capacity((seq_len as usize) * (head_size / 2));
    let mut freqs_sin = Vec::with_capacity((seq_len as usize) * (head_size / 2));
    for t in 0..seq_len {
        for &f in &inv_freq {
            let val = (t as f32) * f;
            freqs_cos.push(val.cos());
            freqs_sin.push(val.sin());
        }
    }

    // Write RoPE tables
    let cos_bytes = unsafe {
        std::slice::from_raw_parts(freqs_cos.as_ptr() as *const u8, freqs_cos.len() * 4)
    };
    writer.write_all(cos_bytes).unwrap();

    let sin_bytes = unsafe {
        std::slice::from_raw_parts(freqs_sin.as_ptr() as *const u8, freqs_sin.len() * 4)
    };
    writer.write_all(sin_bytes).unwrap();
    println!(" - Wrote: RoPE cos/sin tables");

    println!("Success! Created qwen_3b.bin");
}
