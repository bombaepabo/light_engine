use std::fs::File;
use std::io::Write;
use memmap2::Mmap;

// Quantizes a slice of f32s to q8_0 format.
// Returns a tuple of: (scales: Vec<f32>, quantized_weights: Vec<i8>)
fn quantize_q8(data: &[f32], block_size: usize) -> (Vec<f32>, Vec<i8>) {
    assert_eq!(data.len() % block_size, 0, "Data size must be a multiple of block size");
    let n_blocks = data.len() / block_size;
    
    let mut scales = Vec::with_capacity(n_blocks);
    let mut quantized = Vec::with_capacity(data.len());
    
    for b in 0..n_blocks {
        let block_slice = &data[b * block_size .. (b + 1) * block_size];
        
        // Find absolute maximum
        let mut max_val = 0.0f32;
        for &val in block_slice {
            let abs_val = val.abs();
            if abs_val > max_val {
                max_val = abs_val;
            }
        }
        
        // Calculate scale factor (avoid divide by zero)
        let scale = if max_val == 0.0 { 1e-5 } else { max_val / 127.0 };
        scales.push(scale);
        
        // Quantize floats to i8
        for &val in block_slice {
            let q = (val / scale).round();
            // Clamp value to safe i8 range [-127, 127]
            let q_clamped = q.clamp(-127.0, 127.0) as i8;
            quantized.push(q_clamped);
        }
    }
    
    (scales, quantized)
}

fn main() {
    let input_path = "tinyllama_1.1b.bin";
    let output_path = "tinyllama_1.1b_q8.bin";
    let block_size = 32;

    println!("Reading {}...", input_path);
    let file = File::open(input_path).expect("Failed to open input file");
    let mmap = unsafe { Mmap::map(&file).expect("Failed to memory map file") };

    // Read the 28-byte header config (7 integers)
    let mut header_bytes = [0u8; 28];
    header_bytes.copy_from_slice(&mmap[0..28]);
    
    let mut header = [0i32; 7];
    for i in 0..7 {
        header[i] = i32::from_le_bytes(header_bytes[i*4..(i+1)*4].try_into().unwrap());
    }

    let dim = header[0] as usize;
    let hidden_dim = header[1] as usize;
    let n_layers = header[2] as usize;
    let n_heads = header[3] as usize;
    let n_kv_heads = header[4] as usize;
    let raw_vocab_size = header[5];
    let vocab_size = raw_vocab_size.abs() as usize;
    let seq_len = header[6] as usize;
    let head_size = dim / n_heads;

    println!("Config:\n - Dim: {}\n - Hidden Dim: {}\n - Layers: {}\n - Vocab: {}", dim, hidden_dim, n_layers, vocab_size);

    // Cast the rest of the file to f32
    let weights_f32 = unsafe {
        let ptr = mmap.as_ptr().add(28) as *const f32;
        let len = (mmap.len() - 28) / 4;
        std::slice::from_raw_parts(ptr, len)
    };

    let mut offset = 0;
    // Slice all weights sequentially manually (prevents borrow checker issues)
    let token_embedding_table = &weights_f32[offset .. offset + vocab_size * dim];
    offset += vocab_size * dim;
    let rms_att_weight_block = &weights_f32[offset .. offset + n_layers * dim];
    offset += n_layers * dim;
    
    let wq_block = &weights_f32[offset .. offset + n_layers * dim * (n_heads * head_size)];
    offset += n_layers * dim * (n_heads * head_size);
    
    let wk_block = &weights_f32[offset .. offset + n_layers * dim * (n_kv_heads * head_size)];
    offset += n_layers * dim * (n_kv_heads * head_size);
    
    let wv_block = &weights_f32[offset .. offset + n_layers * dim * (n_kv_heads * head_size)];
    offset += n_layers * dim * (n_kv_heads * head_size);
    
    let wo_block = &weights_f32[offset .. offset + n_layers * (n_heads * head_size) * dim];
    offset += n_layers * (n_heads * head_size) * dim;
    
    let rms_ffn_weight_block = &weights_f32[offset .. offset + n_layers * dim];
    offset += n_layers * dim;
    
    let w1_block = &weights_f32[offset .. offset + n_layers * hidden_dim * dim];
    offset += n_layers * hidden_dim * dim;
    
    let w2_block = &weights_f32[offset .. offset + n_layers * dim * hidden_dim];
    offset += n_layers * dim * hidden_dim;
    
    let w3_block = &weights_f32[offset .. offset + n_layers * hidden_dim * dim];
    offset += n_layers * hidden_dim * dim;
    
    let rms_final_weight = &weights_f32[offset .. offset + dim];
    offset += dim;
    
    let freq_cis_real = &weights_f32[offset .. offset + seq_len * (head_size / 2)];
    offset += seq_len * (head_size / 2);
    
    let freq_cis_imag = &weights_f32[offset .. offset + seq_len * (head_size / 2)];
    offset += seq_len * (head_size / 2);
    let wcls = if offset < weights_f32.len() {
        let start = offset;
        offset += vocab_size * dim;
        Some(&weights_f32[start..offset])
    } else {
        None
    };

    println!("Quantizing projection and FFN layers to Q8_0 (block size 32)...");
    
    // Quantize the weight matrices
    let (wq_scales, wq_weights) = quantize_q8(wq_block, block_size);
    let (wk_scales, wk_weights) = quantize_q8(wk_block, block_size);
    let (wv_scales, wv_weights) = quantize_q8(wv_block, block_size);
    let (wo_scales, wo_weights) = quantize_q8(wo_block, block_size);
    
    let (w1_scales, w1_weights) = quantize_q8(w1_block, block_size);
    let (w2_scales, w2_weights) = quantize_q8(w2_block, block_size);
    let (w3_scales, w3_weights) = quantize_q8(w3_block, block_size);

    println!("Writing to {}...", output_path);
    let mut out_file = File::create(output_path).expect("Failed to create output file");
    
    // 1. Write header
    out_file.write_all(&header_bytes).unwrap();
    
    // Helper to write floats
    let write_floats = |file: &mut File, data: &[f32]| {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        file.write_all(bytes).unwrap();
    };

    // Helper to write quantized tensor (scales first, then weights)
    let write_q8_tensor = |file: &mut File, scales: &[f32], weights: &[i8]| {
        // Write scales
        let scale_bytes = unsafe {
            std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4)
        };
        file.write_all(scale_bytes).unwrap();

        // Write i8 weights
        let weight_bytes = unsafe {
            std::slice::from_raw_parts(weights.as_ptr() as *const u8, weights.len())
        };
        file.write_all(weight_bytes).unwrap();
    };

    // 2. Write token embedding and rms_att_weight
    write_floats(&mut out_file, token_embedding_table);
    write_floats(&mut out_file, rms_att_weight_block);

    // 3. Write quantized attention weights
    write_q8_tensor(&mut out_file, &wq_scales, &wq_weights);
    write_q8_tensor(&mut out_file, &wk_scales, &wk_weights);
    write_q8_tensor(&mut out_file, &wv_scales, &wv_weights);
    write_q8_tensor(&mut out_file, &wo_scales, &wo_weights);

    // 4. Write rms_ffn_weight
    write_floats(&mut out_file, rms_ffn_weight_block);

    // 5. Write quantized FFN weights
    write_q8_tensor(&mut out_file, &w1_scales, &w1_weights);
    write_q8_tensor(&mut out_file, &w2_scales, &w2_weights);
    write_q8_tensor(&mut out_file, &w3_scales, &w3_weights);

    // 6. Write final_rms and freqs
    write_floats(&mut out_file, rms_final_weight);
    write_floats(&mut out_file, freq_cis_real);
    write_floats(&mut out_file, freq_cis_imag);

    if let Some(wcls_data) = wcls {
        write_floats(&mut out_file, wcls_data);
    }

    println!("Success! Created {}", output_path);
}